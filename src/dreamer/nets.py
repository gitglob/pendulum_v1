"""DreamerV3 building blocks: symlog, twohot, and the image encoder/decoder.

These small pieces are most of what makes DreamerV3 work across wildly
different reward scales without per-task tuning, so they are worth stating
plainly:

* **symlog** squashes large magnitudes while staying linear near zero and
  invertible, so a reward of 1000 and a reward of 0.01 both land in a range a
  network can regress.
* **twohot** turns a scalar target into a distribution over a fixed grid of
  bins, splitting mass between the two bins it falls between. Reward and value
  become *classification* problems, which do not chase a moving regression
  target the way MSE heads do.

Images are deliberately NOT symlogged: they are scaled to [-0.5, 0.5] and
trained with MSE, as in the paper.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|) -- compresses scale, keeps sign, invertible."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `symlog`."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class TwoHotHead(nn.Module):
    """Predicts a scalar as a distribution over `bins` points in symlog space.

    Args:
        in_dim: width of the input feature.
        config: the `dreamer` config block -- `units`, `layers`, `bins`,
            `bin_limit` (the grid spans [-bin_limit, +bin_limit] in symlog
            space, so +-20 covers returns up to symexp(20)).
    """

    def __init__(self, in_dim: int, config: dict[str, Any]):
        super().__init__()
        # Zero-initialised output: the head starts as a uniform distribution
        # over bins rather than an arbitrary confident guess.
        self.mlp = mlp(in_dim, config["bins"], config, out_scale=0.0)
        limit = config["bin_limit"]
        self.register_buffer("bin_values", torch.linspace(-limit, limit, config["bins"]))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def mean(self, features: torch.Tensor) -> torch.Tensor:
        """Expected value under the predicted distribution, in real units."""
        probs = F.softmax(self.logits(features), dim=-1)
        return symexp((probs * self.bin_values).sum(dim=-1))

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against the twohot encoding of `target`."""
        log_probs = F.log_softmax(self.logits(features), dim=-1)
        return -(twohot(symlog(target), self.bin_values) * log_probs).sum(dim=-1)


def twohot(x: torch.Tensor, bin_values: torch.Tensor) -> torch.Tensor:
    """Encode `x` as mass split between its two neighbouring bins.

    Args:
        x: scalars already in symlog space, any shape.
        bin_values: the monotonically increasing bin grid, [bins].

    Returns:
        One distribution per input, [..., bins]. A value exactly on a bin gets
        all its mass there; outside the grid it saturates at the end bin.
    """
    x = x.clamp(bin_values[0], bin_values[-1])
    # Index of the bin just below x, and how far between the two bins x sits.
    above = torch.sum((bin_values <= x.unsqueeze(-1)).long(), dim=-1) - 1
    above = above.clamp(0, bin_values.numel() - 2)
    lower, upper = bin_values[above], bin_values[above + 1]
    weight_upper = ((x - lower) / (upper - lower)).unsqueeze(-1)

    encoding = torch.zeros(*x.shape, bin_values.numel(), device=x.device)
    encoding.scatter_(-1, above.unsqueeze(-1), 1.0 - weight_upper)
    encoding.scatter_(-1, (above + 1).unsqueeze(-1), weight_upper)
    return encoding


def mlp(
    in_dim: int, out_dim: int, config: dict[str, Any], out_scale: float = 1.0
) -> nn.Sequential:
    """The paper's MLP: `layers` hidden layers of `units`, LayerNorm + SiLU.

    Args:
        out_scale: multiplies the output layer's initial weights. DreamerV3
            initialises prediction heads at (near) zero so they start neutral;
            for the actor this matters a lot, because a head that starts with
            large outputs saturates its tanh and never recovers -- the gradient
            through a saturated tanh is ~0, so the policy stays stuck.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(config["layers"]):
        layers += [nn.Linear(dim, config["units"]), nn.LayerNorm(config["units"]), nn.SiLU()]
        dim = config["units"]
    output = nn.Linear(dim, out_dim)
    with torch.no_grad():
        output.weight.mul_(out_scale)
        output.bias.zero_()
    layers.append(output)
    return nn.Sequential(*layers)


class ImageEncoder(nn.Module):
    """Four stride-2 convolutions: (frames, 64, 64) -> a flat embedding."""

    def __init__(self, obs_shape: tuple[int, ...], config: dict[str, Any]):
        super().__init__()
        depth = config["cnn_depth"]
        channels = [obs_shape[0], depth, depth * 2, depth * 4, depth * 8]
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [
                nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                # LayerNorm over channel+spatial dims, as in the paper's
                # normalisation of every conv layer.
                nn.GroupNorm(1, channels[i + 1]),
                nn.SiLU(),
            ]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, *obs_shape)).flatten(1).shape[1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: [batch, frames, 64, 64] scaled to [-0.5, 0.5] -> [batch, out_dim]."""
        return self.net(obs).flatten(start_dim=1)


class ImageDecoder(nn.Module):
    """Mirror of `ImageEncoder`: latent features -> predicted frame stack."""

    def __init__(self, in_dim: int, obs_shape: tuple[int, ...], config: dict[str, Any]):
        super().__init__()
        depth = config["cnn_depth"]
        self.start_shape = (depth * 8, 4, 4)
        self.linear = nn.Linear(in_dim, depth * 8 * 4 * 4)
        channels = [depth * 8, depth * 4, depth * 2, depth, obs_shape[0]]
        layers: list[nn.Module] = []
        for i in range(4):
            layers.append(
                nn.ConvTranspose2d(channels[i], channels[i + 1], 4, stride=2, padding=1)
            )
            if i < 3:
                layers += [nn.GroupNorm(1, channels[i + 1]), nn.SiLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predicts the image in [-0.5, 0.5]; no output activation (MSE loss)."""
        x = self.linear(features).reshape(-1, *self.start_shape)
        return self.net(x)


def to_image(obs: torch.Tensor) -> torch.Tensor:
    """uint8 frames in [0, 255] -> float in [-0.5, 0.5], the decoder's target."""
    return obs.float() / 255.0 - 0.5
