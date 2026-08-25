"""CEM model-predictive control -- the "planning" half of PETS.

At every environment step the planner searches for an action sequence that
maximises predicted return under the learned dynamics, executes only the first
action, and re-plans at the next step.

The planner is agnostic to what it plans *in*: the state agent rolls out
observations and scores them with Pendulum's known analytic reward
(`pendulum_reward`, the standard setup for PETS on these benchmarks, stated as
an assumption in the README), while the pixel agent rolls out autoencoder
latents and scores them with a learned reward head. Both arrive here as a
`reward_fn` handed to `CEMPlanner`.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .dynamics import ProbabilisticEnsemble


def pendulum_reward(
    obs: torch.Tensor, act: torch.Tensor, max_torque: float
) -> torch.Tensor:
    """Pendulum-v1's exact reward: -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2).

    `obs` is [cos(theta), sin(theta), theta_dot], so theta comes back via atan2,
    which already returns a value in (-pi, pi] -- the same range Pendulum's own
    `angle_normalize` produces, so no extra wrapping is needed.

    Note this is a function of the state *before* the transition and of the
    action taken there, matching `PendulumEnv.step`; the rollout below therefore
    accumulates reward before advancing the model.
    """
    theta = torch.atan2(obs[..., 1], obs[..., 0])
    theta_dot = obs[..., 2]
    u = act[..., 0].clamp(-max_torque, max_torque)
    return -(theta**2 + 0.1 * theta_dot**2 + 0.001 * u**2)


class CEMPlanner:
    """Cross-entropy-method planner over open-loop action sequences.

    Args:
        model: dynamics the plans are rolled out through. It works in whatever
            space the caller trained it on -- observations for the state agent,
            autoencoder latents for the pixel agent.
        config: full config; the `planner` block is read here.
        action_low / action_high: bounds candidate actions are clipped to.
        seed: seeds this planner's own RNG, so CEM sampling is reproducible
            without disturbing the exploration or model-training streams.
        reward_fn: scores `(state, action)` for imagined states, returning one
            reward per row. `pendulum_reward` for the state agent; the learned
            reward head for the pixel agent, which has no angle to read.
    """

    def __init__(
        self,
        model: ProbabilisticEnsemble,
        config: dict[str, Any],
        action_low: float,
        action_high: float,
        seed: int,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.device = model.device
        planner_config = config["planner"]
        self.horizon = planner_config["horizon"]
        self.popsize = planner_config["popsize"]
        self.elites = planner_config["elites"]
        self.iters = planner_config["iters"]
        self.alpha = planner_config["alpha"]
        self.n_particles = planner_config["n_particles"]
        self.warm_start = planner_config["warm_start"]

        if self.n_particles % model.n_members:
            raise ValueError(
                f"n_particles ({self.n_particles}) must be a multiple of "
                f"ensemble_size ({model.n_members})"
            )

        self.act_dim = model.act_dim
        self.action_low = action_low
        self.action_high = action_high
        self.max_torque = action_high
        # A wide-enough starting distribution to cover the action range.
        self.init_std = (action_high - action_low) / 2.0

        # Dedicated generator: CEM sampling is reproducible without disturbing
        # the streams used for exploration and model training.
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_solution: torch.Tensor | None = None

    def reset(self) -> None:
        """Forget the warm-start solution (call between episodes)."""
        self.last_solution = None

    @torch.no_grad()
    def _rollout_returns(self, obs: np.ndarray, actions: torch.Tensor) -> torch.Tensor:
        """Predicted return of each candidate sequence, averaged over particles.

        actions: [popsize, horizon, act_dim] -> returns: [popsize]
        """
        n_members = self.model.n_members
        per_member = self.n_particles // n_members
        chunk = per_member * self.popsize

        start = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # One row per (particle, candidate), grouped so each ensemble member
        # owns a contiguous chunk for the whole horizon (TS-infinity).
        state = start.expand(n_members, chunk, self.model.obs_dim).contiguous()

        # [popsize, H, A] -> [n_particles, popsize, H, A] -> member blocks.
        # Particle-major order matters: it puts every candidate in every
        # member's block, so each candidate is scored by the whole ensemble.
        expanded = (
            actions.unsqueeze(0)
            .expand(self.n_particles, self.popsize, self.horizon, self.act_dim)
            .reshape(n_members, chunk, self.horizon, self.act_dim)
        )

        returns = torch.zeros(n_members, chunk, device=self.device)
        for t in range(self.horizon):
            act = expanded[:, :, t]
            # Reward depends on the pre-transition state, so score then step.
            returns += self.reward_fn(state, act)
            state = self.model.propagate(state, act)

        return returns.reshape(self.n_particles, self.popsize).mean(dim=0)

    @torch.no_grad()
    def plan(self, obs: np.ndarray) -> np.ndarray:
        """Return the first action of the best plan found for `obs`."""
        shape = (self.horizon, self.act_dim)
        if self.warm_start and self.last_solution is not None:
            # Shift last step's plan forward one step and repeat its tail.
            mean = torch.cat([self.last_solution[1:], self.last_solution[-1:]])
        else:
            mean = torch.zeros(shape, device=self.device)
        std = torch.full(shape, self.init_std, device=self.device)

        for _ in range(self.iters):
            noise = torch.randn(
                (self.popsize, *shape), generator=self.generator, device=self.device
            )
            candidates = (mean + std * noise).clamp(self.action_low, self.action_high)

            returns = self._rollout_returns(obs, candidates)
            elite = candidates[returns.topk(self.elites).indices]

            # Momentum on the refit keeps the search from collapsing too early.
            mean = self.alpha * mean + (1 - self.alpha) * elite.mean(dim=0)
            std = self.alpha * std + (1 - self.alpha) * elite.std(dim=0)

        self.last_solution = mean
        return mean[0].cpu().numpy()
