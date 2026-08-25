"""DreamerV3 agent: a world model, and an actor-critic trained inside it.

The division of labour is the whole point of the algorithm:

* `WorldModel` learns to predict from real experience only.
* `ActorCritic` never touches the environment. It trains on trajectories the
  world model imagines, which is why Dreamer is sample-efficient, and it
  distils that into a feed-forward policy, which is why acting is cheap -- one
  forward pass, not a search like PETS' CEM.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import nn

from .nets import ImageDecoder, ImageEncoder, TwoHotHead, mlp, to_image
from .rssm import RSSM, State


class WorldModel(nn.Module):
    """Encoder + RSSM + decoder + reward head + continue head.

    Args:
        obs_shape: observation shape `(frames, height, width)`.
        act_dim: action width.
        config: the `dreamer` config block.
        device: torch device.
    """

    def __init__(self, obs_shape: tuple[int, ...], act_dim: int,
                 config: dict[str, Any], device: str):
        super().__init__()
        self.device = device
        self.config = config
        self.encoder = ImageEncoder(obs_shape, config)
        self.rssm = RSSM(self.encoder.out_dim, act_dim, config, device)
        feature_dim = self.rssm.feature_dim
        self.decoder = ImageDecoder(feature_dim, obs_shape, config)
        self.reward_head = TwoHotHead(feature_dim, config)
        # Predicts whether the episode continues. Pendulum never terminates --
        # its 200-step limit is truncation, not termination -- so this target is
        # always 1 here; it is kept because the imagination discounting below is
        # written against it.
        self.continue_head = mlp(feature_dim, 1, config)
        self.to(device)

    def loss(
        self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, State, dict[str, float]]:
        """One world-model training step over a batch of sequences.

        Args:
            obs: uint8 frame stacks, [batch, time, frames, H, W].
            actions: actions taken into each step, [batch, time, act_dim].
            rewards: rewards received at each step, [batch, time].

        Returns:
            `(loss, posterior_states, metrics)`. The states are handed to the
            actor-critic as imagination starting points.
        """
        batch, time = obs.shape[:2]
        images = to_image(obs.reshape(batch * time, *obs.shape[2:]))
        embeds = self.encoder(images).reshape(batch, time, -1)

        states, prior_logits, posterior_logits = self.rssm.observe(embeds, actions)
        features = self.rssm.features(states).reshape(batch * time, -1)

        # Image reconstruction: plain MSE in [-0.5, 0.5] space, no symlog.
        recon = self.decoder(features)
        image_loss = ((recon - images) ** 2).sum(dim=(1, 2, 3)).mean()
        # Reward and continuation are classification problems (twohot / BCE).
        reward_loss = self.reward_head.loss(features, rewards.reshape(-1)).mean()
        continue_loss = F.binary_cross_entropy_with_logits(
            self.continue_head(features).squeeze(-1),
            torch.ones(batch * time, device=self.device),
        )

        dynamics_kl, representation_kl = self.rssm.kl_losses(
            prior_logits, posterior_logits, self.config["free_bits"]
        )

        loss = (
            image_loss
            + reward_loss
            + continue_loss
            + self.config["beta_dynamics"] * dynamics_kl
            + self.config["beta_representation"] * representation_kl
        )
        metrics = {
            "world/image_loss": float(image_loss.detach()),
            "world/reward_loss": float(reward_loss.detach()),
            "world/dynamics_kl": float(dynamics_kl.detach()),
            "world/representation_kl": float(representation_kl.detach()),
        }
        return loss, states, metrics


class ActorCritic(nn.Module):
    """Policy and value function, trained only on imagined rollouts.

    Args:
        feature_dim: width of the RSSM feature the heads read.
        act_dim: action width.
        max_action: actions are squashed to [-max_action, max_action].
        config: the `dreamer` config block.
        device: torch device.
    """

    def __init__(self, feature_dim: int, act_dim: int, max_action: float,
                 config: dict[str, Any], device: str):
        super().__init__()
        self.config = config
        self.device = device
        self.act_dim = act_dim
        self.max_action = max_action
        # Continuous action head: mean and (softplus) std of a squashed Normal.
        # Small output init keeps the tanh away from saturation at step 0.
        self.actor = mlp(
            feature_dim, 2 * act_dim, config, out_scale=config["actor_out_scale"]
        )
        self.critic = TwoHotHead(feature_dim, config)
        # Slow-moving copy of the critic supplies bootstrap targets, so the
        # value function is not chasing its own instantaneous estimate.
        self.slow_critic = TwoHotHead(feature_dim, config)
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for param in self.slow_critic.parameters():
            param.requires_grad_(False)
        # Running percentile range used to normalise returns, so one entropy
        # scale works whatever the reward magnitude, plus a running mean/scale
        # for the advantage itself (the repo's second normaliser). Without the
        # latter the very first updates see advantages of order the return
        # magnitude and drive the actor straight into tanh saturation.
        self.register_buffer("return_range", torch.ones(()))
        self.register_buffer("advantage_mean", torch.zeros(()))
        self.register_buffer("advantage_scale", torch.ones(()))
        self.to(device)

    def distribution(self, features: torch.Tensor) -> dist.Distribution:
        mean, std = self.actor(features).chunk(2, dim=-1)
        std = F.softplus(std) + self.config["min_std"]
        base = dist.Normal(torch.tanh(mean) * self.max_action, std)
        return dist.Independent(base, 1)

    def act(self, features: torch.Tensor, sample: bool = True) -> torch.Tensor:
        distribution = self.distribution(features)
        action = distribution.sample() if sample else distribution.mean
        return action.clamp(-self.max_action, self.max_action)

    def update_slow_critic(self) -> None:
        """EMA step of the target critic."""
        rate = self.config["slow_critic_rate"]
        with torch.no_grad():
            for slow, fast in zip(self.slow_critic.parameters(), self.critic.parameters()):
                slow.mul_(1 - rate).add_(rate * fast)

    def lambda_returns(
        self, rewards: torch.Tensor, values: torch.Tensor, continues: torch.Tensor
    ) -> torch.Tensor:
        """Discounted lambda-returns along an imagined trajectory.

        Args:
            rewards: reward of each imagined step, [batch, horizon].
            values: value of every state including the final one,
                [batch, horizon + 1]; the last entry is the bootstrap.
            continues: continuation probability per step, [batch, horizon].

        Returns:
            Returns for each step, [batch, horizon], computed backwards so each
            step mixes its own reward with the discounted return that follows.
        """
        discount = self.config["discount"] * continues
        lam = self.config["lambda"]
        returns = torch.zeros_like(rewards)
        # The tail bootstraps purely off the value of the state after the last
        # imagined action.
        accumulator = values[:, -1]
        for t in reversed(range(rewards.shape[1])):
            accumulator = rewards[:, t] + discount[:, t] * (
                (1 - lam) * values[:, t + 1] + lam * accumulator
            )
            returns[:, t] = accumulator
        return returns

    def loss(
        self, world_model: WorldModel, start: State
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Imagine from `start`, then score the actor and critic on it.

        Returns:
            `(actor_loss, critic_loss, metrics)`.
        """
        horizon = self.config["imagination_horizon"]
        # states has horizon + 1 entries; actions has horizon.
        states, actions = world_model.rssm.imagine(
            start, lambda features: self.act(features), horizon
        )
        features = world_model.rssm.features(states)
        flat = features.reshape(-1, features.shape[-1])
        # Features of the states actions were actually taken from.
        acting_features = features[:, :horizon]
        acting_flat = acting_features.reshape(-1, acting_features.shape[-1])

        rewards = world_model.reward_head.mean(acting_flat).reshape(
            acting_features.shape[:2]
        )
        continues = torch.sigmoid(
            world_model.continue_head(acting_flat).squeeze(-1)
        ).reshape(acting_features.shape[:2])
        with torch.no_grad():
            # Values for all horizon + 1 states, so the tail can bootstrap.
            values = self.slow_critic.mean(flat).reshape(features.shape[:2])
            returns = self.lambda_returns(rewards, values, continues)

        # Normalise returns by their 5-95 percentile spread, floored at 1, so
        # the entropy bonus keeps a consistent meaning across reward scales.
        low, high = torch.quantile(
            returns.detach(), torch.tensor([0.05, 0.95], device=self.device)
        )
        self.return_range.mul_(1 - self.config["return_ema_rate"]).add_(
            self.config["return_ema_rate"] * (high - low)
        )
        scale = torch.clamp(self.return_range, min=1.0)

        distribution = self.distribution(acting_features.detach())
        log_probs = distribution.log_prob(actions.detach())
        entropy = distribution.entropy()

        # REINFORCE on a twice-normalised advantage: first by the return scale,
        # then by the advantage's own running mean and spread, so the gradient
        # magnitude is O(1) no matter what the rewards look like.
        advantage = (returns - values[:, :horizon]) / scale
        ema = self.config["advantage_ema_rate"]
        self.advantage_mean.mul_(1 - ema).add_(ema * advantage.mean())
        self.advantage_scale.mul_(1 - ema).add_(ema * advantage.std())
        normalized = (advantage - self.advantage_mean) / torch.clamp(
            self.advantage_scale, min=1.0
        )

        actor_loss = -(log_probs * normalized.detach()).mean()
        actor_loss = actor_loss - self.config["entropy_scale"] * entropy.mean()

        critic_loss = self.critic.loss(
            acting_flat.detach(), returns.detach().reshape(-1)
        ).mean()

        metrics = {
            "actor/loss": float(actor_loss.detach()),
            "actor/entropy": float(entropy.mean().detach()),
            "critic/loss": float(critic_loss.detach()),
            "imagined/return": float(returns.mean().detach()),
            "imagined/reward": float(rewards.mean().detach()),
            "actor/return_range": float(scale.detach()),
            "actor/advantage_scale": float(self.advantage_scale),
            "actor/action_abs_mean": float(actions.detach().abs().mean()),
        }
        return actor_loss, critic_loss, metrics


class DreamerAgent:
    """Stateful policy used for acting: encode, advance the RSSM, act.

    Carries the recurrent state between steps, which is why the shared harness
    calls `reset()` at every episode boundary.
    """

    def __init__(self, world_model: WorldModel, actor_critic: ActorCritic,
                 act_dim: int, sample: bool):
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.act_dim = act_dim
        self.sample = sample
        self.state: State | None = None
        self.last_action: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None
        self.last_action = None

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        device = self.world_model.device
        if self.state is None:
            self.state = self.world_model.rssm.initial(1)
            self.last_action = torch.zeros(1, self.act_dim, device=device)

        image = to_image(torch.as_tensor(obs, device=device).unsqueeze(0))
        embed = self.world_model.encoder(image)
        self.state, _, _ = self.world_model.rssm.step(
            self.state, self.last_action, embed
        )
        action = self.actor_critic.act(
            self.world_model.rssm.features(self.state), sample=self.sample
        )
        self.last_action = action
        return action.squeeze(0).cpu().numpy()
