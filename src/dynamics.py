"""Probabilistic dynamics ensemble -- the learned model behind PETS.

Each member is an MLP that maps (obs, action) to a Gaussian over the
observation *delta*: mean and log-variance. Members are trained by Gaussian
NLL on their own bootstrap resample of the replay buffer, so disagreement
between them expresses epistemic uncertainty about the dynamics.

All members live in batched weight tensors and are evaluated in a single pass:
the planner calls this model tens of times per action, so per-member kernel
launches would dominate the runtime.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class EnsembleLinear(nn.Module):
    """Linear layer holding `n_members` independent weight matrices.

    Args:
        n_members: number of ensemble members, each with its own weights.
        in_features: input width of every member.
        out_features: output width of every member.
    """

    def __init__(self, n_members: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, out_features))
        # Truncated-normal init as in the PETS reference implementation.
        std = 1.0 / (2.0 * math.sqrt(in_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every member's weights in one batched matmul.

        Args:
            x: inputs, [n_members, batch, in_features]. Each member sees its own
                slice along dim 0, so callers must materialise that dimension
                (an expanded, stride-0 view will not do -- `baddbmm` needs it
                contiguous).

        Returns:
            [n_members, batch, out_features].
        """
        return torch.baddbmm(self.bias, x, self.weight)


class ProbabilisticEnsemble(nn.Module):
    """Ensemble of MLPs, each predicting a Gaussian over the observation delta.

    Args:
        obs_dim: observation width (3 for Pendulum: cos, sin, theta_dot).
        act_dim: action width (1 for Pendulum).
        config: the `model` block of the YAML config -- `ensemble_size` and
            `hidden_layers` are read here, the training keys in `train_model`.
        device: torch device string the whole ensemble lives on.
        projection: optional map applied to every predicted next state during
            planning rollouts, used to keep them on the manifold the model was
            trained on. The pixel agent passes LayerNorm, matching how its
            encoder produces latents; the state agent needs none.
    """

    def __init__(self, obs_dim: int, act_dim: int, config: dict[str, Any], device: str,
                 projection: Callable[[torch.Tensor], torch.Tensor] | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_members = config["ensemble_size"]
        self.device = device
        self.projection = projection

        layers: list[nn.Module] = []
        in_dim = obs_dim + act_dim
        for hidden in config["hidden_layers"]:
            layers += [EnsembleLinear(self.n_members, in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        # Two heads in one output: mean delta, then log-variance.
        layers.append(EnsembleLinear(self.n_members, in_dim, 2 * obs_dim))
        self.net = nn.Sequential(*layers)

        # Learned soft bounds on the predicted log-variance (PETS appendix A.1).
        self.max_logvar = nn.Parameter(torch.full((obs_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim,), -10.0))

        # Input normalization statistics, refit from the buffer at every retrain.
        self.register_buffer("input_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("input_std", torch.ones(obs_dim + act_dim))

        self.to(device)

    def fit_normalizer(self, inputs: torch.Tensor) -> None:
        """Refit input normalization statistics.

        Args:
            inputs: concatenated (obs, act) rows, [batch, obs_dim + act_dim],
                taken from the training split only so holdout data stays unseen.
        """
        self.input_mean = inputs.mean(dim=0)
        # Guard against constant columns; an unclamped std would blow up inputs.
        self.input_std = inputs.std(dim=0).clamp_min(1e-6)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict each member's Gaussian over the observation delta.

        Args:
            obs: observations, [n_members, batch, obs_dim].
            act: actions taken there, [n_members, batch, act_dim].

        Returns:
            `(mean_delta, logvar)`, both [n_members, batch, obs_dim]. The
            prediction is of the *delta*, so the next observation is
            `obs + mean_delta`; `logvar` is soft-clamped between the learned
            bounds.
        """
        x = (torch.cat([obs, act], dim=-1) - self.input_mean) / self.input_std
        mean, logvar = self.net(x).split(self.obs_dim, dim=-1)
        # Soft clamp keeps the variance in a sane range while staying differentiable.
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.no_grad()
    def propagate(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Sample the next observation for planning particles (TS-infinity).

        Args:
            obs: current particle states, [n_members, chunk, obs_dim]. `chunk`
                is however many particles each member owns; a particle stays
                with the same member for the whole horizon, which is the variant
                the PETS paper found best and needs no per-step member gather.
            act: actions applied to those particles, [n_members, chunk, act_dim].

        Returns:
            Next observations sampled from each member's predicted Gaussian,
            same shape as `obs`.
        """
        mean, logvar = self(obs, act)
        delta = mean + torch.randn_like(mean) * logvar.exp().sqrt()
        next_obs = obs + delta
        # Keep long rollouts on the manifold the model was trained on. Without
        # this the pixel agent's latent rollout error grew from 0.003 at one
        # step to ~3500 by fifteen, which left the planner ranking noise.
        return self.projection(next_obs) if self.projection else next_obs

    def _nll_loss(
        self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian negative log-likelihood, summed over members.

        Args:
            obs: observations, [n_members, batch, obs_dim] -- each member gets
                its own bootstrap rows, which is why dim 0 is not shared.
            act: actions taken there, [n_members, batch, act_dim].
            target: ground-truth deltas (`next_obs - obs`), same shape as `obs`.

        Returns:
            Scalar loss: NLL averaged within each member and summed across them,
            plus a small penalty on the learned log-variance bounds.
        """
        mean, logvar = self(obs, act)
        inv_var = torch.exp(-logvar)
        # Gaussian NLL, dropping the constant term; mean over batch and dims.
        loss = (((mean - target) ** 2) * inv_var + logvar).mean(dim=(1, 2)).sum()
        # Small penalty pulling the learned variance bounds inwards.
        return loss + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def _holdout_mse(
        self, obs: torch.Tensor, act: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Mean one-step prediction error of the ensemble mean, for monitoring.

        Args:
            obs: holdout observations repeated per member,
                [n_members, batch, obs_dim].
            act: the actions taken there, [n_members, batch, act_dim].
            target: ground-truth deltas, [1, batch, obs_dim] -- one shared copy,
                since every member is scored against the same holdout rows.

        Returns:
            Mean squared error of the averaged member predictions.
        """
        mean, _ = self(obs, act)
        return float(((mean.mean(dim=0) - target[0]) ** 2).mean())


def _rollout_starts(
    n: int, horizon: int, episode_ends: np.ndarray | None, device: str
) -> torch.Tensor:
    """Indices from which `horizon` steps stay inside a single episode."""
    if horizon <= 1:
        return torch.empty(0, dtype=torch.long, device=device)

    valid = np.ones(n, dtype=bool)
    valid[max(0, n - horizon - 1) :] = False
    if episode_ends is not None:
        # Drop any start whose window would step across a reset.
        ends = np.flatnonzero(episode_ends[:n])
        for offset in range(horizon):
            crossing = ends - offset
            valid[crossing[crossing >= 0]] = False
    return torch.as_tensor(np.flatnonzero(valid), dtype=torch.long, device=device)


def multi_step_loss(
    model: ProbabilisticEnsemble,
    obs_t: torch.Tensor,
    act_t: torch.Tensor,
    starts: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """NLL of a `horizon`-step rollout, not just of one step.

    One-step accuracy is a poor proxy for what a planner needs. Measured on the
    pixel agent: a model with a one-step error of 0.003 drifted to 0.34 by
    fifteen steps, and its predicted returns correlated with the truth at 0.02
    -- planning on noise. Rolling the model out during training penalises the
    compounding directly.

    Args:
        model: the ensemble being trained.
        obs_t: all states in temporal order, [n, dim].
        act_t: the action taken at each, [n, act_dim].
        starts: indices to roll out from, [batch]; callers must guarantee
            `start + horizon` stays inside one episode.
        horizon: number of steps to unroll.

    Returns:
        Scalar loss, summed over members and averaged over steps.
    """
    members = model.n_members
    # [members, batch, dim] -- every member rolls out the same starts.
    state = obs_t[starts].unsqueeze(0).expand(members, -1, -1).contiguous()
    loss = torch.zeros((), device=obs_t.device)

    for step in range(horizon):
        idx = starts + step
        act = act_t[idx].unsqueeze(0).expand(members, -1, -1).contiguous()
        mean, logvar = model(state, act)
        target = (obs_t[idx + 1] - obs_t[idx]).unsqueeze(0)

        inv_var = torch.exp(-logvar)
        loss = loss + (((mean - target) ** 2) * inv_var + logvar).mean(dim=(1, 2)).sum()

        # Feed the model its own prediction, which is what planning will do.
        state = state + mean
        if model.projection is not None:
            state = model.projection(state)

    return loss / horizon + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())


def train_model(
    model: ProbabilisticEnsemble,
    obs: np.ndarray,
    act: np.ndarray,
    next_obs: np.ndarray,
    config: dict[str, Any],
    generator: torch.Generator,
    episode_ends: np.ndarray | None = None,
) -> dict[str, float]:
    """Refit the ensemble on the whole buffer; returns losses for logging.

    Each member sees its own bootstrap resample of the training split, which is
    what makes the members disagree in a useful way. Optimizer state is not
    carried across calls: every retrain starts a fresh Adam on the grown buffer.

    Args:
        model: the ensemble to fit, updated in place.
        obs: observations of every transition collected so far, [n, obs_dim].
        act: the action taken in each, [n, act_dim].
        next_obs: the resulting observation, [n, obs_dim]. The training target
            is the delta `next_obs - obs`, not this array directly.
        config: the `model` block of the YAML config -- `holdout_ratio`, `lr`,
            `weight_decay`, `batch_size`, `epochs_per_retrain`.
        generator: torch RNG for the holdout split and the bootstrap draws,
            kept separate from the planner's so runs stay reproducible.

    Returns:
        Scalars for wandb: final `model/train_nll`, `model/buffer_size`, and
        `model/holdout_mse` when the buffer is big enough to hold data out.
    """
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    # The model predicts the delta, not the absolute next observation.
    target_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device) - obs_t

    n = obs_t.shape[0]
    perm = torch.randperm(n, generator=generator, device=device)
    n_holdout = min(int(n * config["holdout_ratio"]), n - 1)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]
    n_train = train_idx.numel()

    model.fit_normalizer(torch.cat([obs_t[train_idx], act_t[train_idx]], dim=-1))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    batch_size = min(config["batch_size"], n_train)
    horizon = config.get("rollout_horizon", 1)
    # Rollout training needs contiguous transitions, so it draws its own starts
    # rather than using the shuffled bootstrap indices.
    rollout_starts = _rollout_starts(n, horizon, episode_ends, device)

    final_loss = 0.0
    for _ in range(config["epochs_per_retrain"]):
        # Independent bootstrap resample per member, reshuffled every epoch.
        boot = torch.randint(
            n_train, (model.n_members, n_train), generator=generator, device=device
        )
        for start in range(0, n_train, batch_size):
            idx = train_idx[boot[:, start : start + batch_size]]
            loss = model._nll_loss(obs_t[idx], act_t[idx], target_t[idx])

            if horizon > 1 and rollout_starts.numel():
                pick = torch.randint(
                    rollout_starts.numel(), (batch_size,),
                    generator=generator, device=device,
                )
                loss = loss + multi_step_loss(
                    model, obs_t, act_t, rollout_starts[pick], horizon
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())

    stats = {"model/train_nll": final_loss, "model/buffer_size": float(n)}
    if n_holdout > 0:
        # Every member scores the same holdout set, so repeat it per member
        # (contiguous: baddbmm cannot take a stride-0 batch dimension).
        expand = (model.n_members, -1, -1)
        stats["model/holdout_mse"] = model._holdout_mse(
            obs_t[holdout_idx].unsqueeze(0).expand(*expand).contiguous(),
            act_t[holdout_idx].unsqueeze(0).expand(*expand).contiguous(),
            target_t[holdout_idx].unsqueeze(0),
        )
    return stats
