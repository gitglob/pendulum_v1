"""Latent representation for PETS-from-pixels.

Planning directly over images is hopeless -- one CEM search would have to
imagine 500 candidates x 20 particles x 15 steps = 150,000 frames -- and the
analytic reward needs an angle that pixels do not hand over. So this module
learns a compact stand-in for the state:

    frame stack --[conv encoder]--> latent  --[ensemble]--> next latent
                                       \\--[reward head]--> reward

The autoencoder is trained by reconstruction, the reward head against the
rewards the environment already returns. Neither uses the true simulator state,
so the resulting agent genuinely learns from pixels alone. The dynamics over
those latents are then the *unchanged* `ProbabilisticEnsemble`, and CEM plans
in latent space at the same cost as it did in state space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Compresses a stacked-frame observation to a latent vector and back.

    Args:
        obs_shape: observation shape `(frames, height, width)`; the frame stack
            is fed to the convolution as channels.
        config: the `encoder` config block -- `latent_dim` and `channels`.
        device: torch device the module lives on.
    """

    def __init__(self, obs_shape: tuple[int, ...], config: dict[str, Any], device: str):
        super().__init__()
        self.device = device
        self.obs_shape = obs_shape
        frames, height, width = obs_shape
        self.latent_dim = config["latent_dim"]

        encoder_layers: list[nn.Module] = []
        in_channels = frames
        for out_channels in config["channels"]:
            encoder_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            ]
            in_channels = out_channels
        self.conv = nn.Sequential(*encoder_layers)

        # Infer the flattened conv output rather than deriving it by hand, so
        # image_size and channels stay free to change in the config.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, frames, height, width)).numel()
        self.spatial = (in_channels, height >> len(config["channels"]), width >> len(config["channels"]))

        # LayerNorm bounds the latent's scale. An unbounded latent is free to
        # drift to a large, oddly-scaled space, which makes the dynamics
        # ensemble's job much harder for no representational gain.
        self.to_latent = nn.Sequential(
            nn.Linear(n_flat, self.latent_dim), nn.LayerNorm(self.latent_dim)
        )
        self.from_latent = nn.Linear(self.latent_dim, n_flat)

        # The decoder rebuilds only the NEWEST frame, not the whole stack. The
        # encoder still reads all 3 (velocity needs them), but if the latent had
        # to describe the whole window, its dynamics would be dominated by the
        # window shifting along -- a large, action-independent motion that
        # drowns the small action-dependent part planning depends on. Targeting
        # the newest frame makes the latent a *current state*, whose dynamics
        # are the Markov ones the ensemble should be learning.
        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(config["channels"]))
        for i, out_channels in enumerate(reversed_channels[1:] + [1]):
            decoder_layers += [
                nn.ConvTranspose2d(
                    reversed_channels[i], out_channels, kernel_size=4, stride=2, padding=1
                ),
                nn.SiLU() if i < len(reversed_channels) - 1 else nn.Sigmoid(),
            ]
        self.deconv = nn.Sequential(*decoder_layers)

        self.to(device)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Map observations to latents.

        Args:
            obs: frame stacks, [batch, frames, H, W], already scaled to [0, 1].

        Returns:
            Latents, [batch, latent_dim].
        """
        return self.to_latent(self.conv(obs).flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct observations from latents; inverse of `encode`."""
        x = self.from_latent(latent).reshape(-1, *self.spatial)
        return self.deconv(x)

    @torch.no_grad()
    def encode_numpy(self, obs: np.ndarray) -> np.ndarray:
        """Encode uint8 observations straight from the replay buffer.

        Args:
            obs: frame stacks, [batch, frames, H, W], uint8 in [0, 255].

        Returns:
            Latents as a numpy array, [batch, latent_dim].
        """
        return self.encode(to_float(obs, self.device)).cpu().numpy()


class TransitionHead(nn.Module):
    """Predicts the latent delta of one step -- the *reason* the latent is learnable.

    Reconstruction alone gives the encoder no incentive to make its latent
    predictable over time, and measurement showed exactly that failure: an
    autoencoder latent whose one-step dynamics error was 3000x worse than the
    true state's, leaving the planner with no signal to rank plans by. Training
    this head jointly pushes that gradient back into the encoder, so the latent
    has to be something a dynamics model can actually follow.

    It is a training-time device only; PETS still plans through the
    `ProbabilisticEnsemble`.
    """

    def __init__(self, latent_dim: int, act_dim: int, config: dict[str, Any], device: str):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim + act_dim
        for hidden in config["transition_hidden"]:
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, latent: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Predicted *next* latent (the head itself outputs the delta)."""
        return latent + self.net(torch.cat([latent, act], dim=-1))


class RewardHead(nn.Module):
    """Predicts the reward of taking `action` in the state a latent describes.

    Trained on rewards the environment returns, which are part of the ordinary
    RL signal -- this is what lets the planner score imagined trajectories
    without access to the true angle.
    """

    def __init__(self, latent_dim: int, act_dim: int, config: dict[str, Any], device: str):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim + act_dim
        for hidden in config["reward_hidden"]:
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, latent: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Reward for each (latent, action) row; trailing dim is squeezed off."""
        return self.net(torch.cat([latent, act], dim=-1)).squeeze(-1)


def to_float(obs: np.ndarray | torch.Tensor, device: str) -> torch.Tensor:
    """uint8 observations in [0, 255] -> float tensor in [0, 1] on `device`."""
    tensor = torch.as_tensor(obs, device=device)
    return tensor.float().div_(255.0)


def train_encoder(
    autoencoder: ConvAutoencoder,
    reward_head: RewardHead,
    transition_head: TransitionHead,
    obs: np.ndarray,
    act: np.ndarray,
    next_obs: np.ndarray,
    rewards: np.ndarray,
    config: dict[str, Any],
    generator: torch.Generator,
) -> dict[str, float]:
    """Fit the autoencoder, reward head and transition head jointly.

    Three objectives shape the latent, and all three are needed:
      * reconstruction keeps it from collapsing to a constant,
      * reward prediction makes it keep what determines control,
      * one-step latent prediction makes it *predictable*, which is what the
        planner ultimately relies on.

    Args:
        autoencoder: updated in place.
        reward_head: updated in place.
        transition_head: updated in place; a training-time device that shapes
            the latent, not the model PETS plans with.
        obs: frame stacks collected so far, [n, frames, H, W] uint8.
        act: the action taken at each, [n, act_dim].
        next_obs: the resulting frame stacks, same shape as `obs`.
        rewards: the reward received at each, [n].
        config: the `encoder` config block -- `lr`, `batch_size`,
            `epochs_per_retrain`, `reward_weight`, `latent_weight`.
        generator: torch RNG for batch shuffling.

    Returns:
        Scalars for wandb: `encoder/recon_mse`, `encoder/reward_mae` and
        `encoder/latent_mse` (the last is the one that predicts whether
        planning will work at all).
    """
    device = autoencoder.device
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        list(autoencoder.parameters())
        + list(reward_head.parameters())
        + list(transition_head.parameters()),
        lr=config["lr"],
    )
    n = obs.shape[0]
    batch_size = min(config["batch_size"], n)
    recon_mse = reward_mae = latent_mse = 0.0

    for _ in range(config["epochs_per_retrain"]):
        perm = torch.randperm(n, generator=generator, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size].cpu().numpy()
            # Frames are converted per batch: the full uint8 buffer would be
            # 4x larger as float, for no benefit.
            frames = to_float(obs[idx], device)
            next_frames = to_float(next_obs[idx], device)

            latent = autoencoder.encode(frames)
            next_latent = autoencoder.encode(next_frames)

            # Planning feeds these heads *predicted* latents, which are always
            # slightly off. Training them on jittered latents makes them smooth
            # enough to survive that: without it the reward head was accurate
            # on true latents (MAE 0.2) yet off by more than 2 on latents only
            # three model steps old, which drowned the signal CEM ranks by.
            jitter = config["latent_noise"] * torch.randn_like(latent)
            recon_loss = ((autoencoder.decode(latent) - frames[:, -1:]) ** 2).mean()
            reward_error = reward_head(latent + jitter, act_t[idx]) - reward_t[idx]
            # Detached target: the transition head must chase the encoder, not
            # drag both towards whatever latent is easiest to predict.
            latent_error = (
                transition_head(latent + jitter, act_t[idx]) - next_latent.detach()
            )

            loss = (
                recon_loss
                + config["reward_weight"] * (reward_error**2).mean()
                + config["latent_weight"] * (latent_error**2).mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            recon_mse = float(recon_loss.detach())
            reward_mae = float(reward_error.detach().abs().mean())
            latent_mse = float((latent_error.detach() ** 2).mean())

    return {
        "encoder/recon_mse": recon_mse,
        "encoder/reward_mae": reward_mae,
        "encoder/latent_mse": latent_mse,
    }
