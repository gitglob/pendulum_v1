"""PETS on Pendulum-v1: probabilistic ensemble + CEM model-predictive control.

The model-based agent this project reports on. It alternates between fitting a
dynamics ensemble on everything collected so far and acting for one episode by
planning through that model at every step (MPC). Every setting comes from
`config/pets.yaml`.

Two input types share this loop, differing only in how an observation becomes
something to plan over (see `Representation` below):

  state  -- plan directly on [cos, sin, theta_dot], score with the known reward
  pixels -- encode a 3-frame stack to a latent, plan there, score with a
            learned reward head

Usage:
    .venv/bin/python -m src.pets_train
    .venv/bin/python -m src.pets_train --obs-type pixels
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .cli import build_parser, load_and_override
from .config import resolve_device
from .dynamics import ProbabilisticEnsemble, train_model
from .encoder import (
    ConvAutoencoder,
    RewardHead,
    TransitionHead,
    to_float,
    train_encoder,
)
from .evaluate import benchmark_inference, evaluate, make_env
from .metrics import PausableTimer, RunMetrics, collect_versions, steps_to_threshold
from .obs import apply_pixel_wrappers, is_pixels, render_mode_for
from .planner import CEMPlanner, pendulum_reward
from .wandb_utils import finish_run, init_run, log_eval_point, log_scalars


class StateRepresentation:
    """Plan directly on the observation, scored by Pendulum's known reward."""

    def __init__(self, obs_shape: tuple[int, ...], act_dim: int,
                 config: dict[str, Any], device: str, max_torque: float):
        self.plan_dim = obs_shape[0]
        self.buffer_dtype = np.float32
        self.reward_fn = partial(pendulum_reward, max_torque=max_torque)
        self.projection = None  # observations are already the natural space

    def fit(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray,
            rewards: np.ndarray, generator: torch.Generator) -> dict[str, float]:
        """Nothing to learn: the observation is already the planning space."""
        return {}

    def encode(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def state_dict(self) -> dict[str, Any]:
        return {}


class PixelRepresentation:
    """Encode frame stacks to latents, and learn the reward the planner needs.

    Neither the autoencoder nor the reward head ever sees the simulator state,
    so an agent built on this learns from pixels alone.
    """

    def __init__(self, obs_shape: tuple[int, ...], act_dim: int,
                 config: dict[str, Any], device: str, max_torque: float):
        encoder_config = config["encoder"]
        self.config = encoder_config
        self.device = device
        self.autoencoder = ConvAutoencoder(obs_shape, encoder_config, device)
        self.reward_head = RewardHead(
            self.autoencoder.latent_dim, act_dim, encoder_config, device
        )
        # Shapes the latent during training; PETS still plans with the ensemble.
        self.transition_head = TransitionHead(
            self.autoencoder.latent_dim, act_dim, encoder_config, device
        )
        self.plan_dim = self.autoencoder.latent_dim
        self.buffer_dtype = np.uint8
        self.reward_fn = self._reward
        # The encoder ends in LayerNorm, so a latent outside that set is
        # off-distribution by construction: project rollouts back onto it.
        self.projection = partial(
            torch.nn.functional.layer_norm, normalized_shape=(self.plan_dim,)
        )

    def _reward(self, latent: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Reward head applied to planning particles, [members, chunk, dim]."""
        return self.reward_head(latent, act)

    def fit(self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray,
            rewards: np.ndarray, generator: torch.Generator) -> dict[str, float]:
        return train_encoder(
            self.autoencoder, self.reward_head, self.transition_head,
            obs, act, next_obs, rewards, self.config, generator,
        )

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode a batch of frame stacks, or a single one, to latents."""
        if obs.ndim == 3:  # one observation: add and drop the batch axis
            return self.autoencoder.encode_numpy(obs[None])[0]
        return self._encode_batched(obs)

    @torch.no_grad()
    def _encode_batched(self, obs: np.ndarray, chunk: int = 512) -> np.ndarray:
        """Encode the whole buffer without materialising it as float at once."""
        latents = [
            self.autoencoder.encode(to_float(obs[i : i + chunk], self.device))
            for i in range(0, obs.shape[0], chunk)
        ]
        return torch.cat(latents).cpu().numpy()

    def state_dict(self) -> dict[str, Any]:
        return {
            "autoencoder": self.autoencoder.state_dict(),
            "reward_head": self.reward_head.state_dict(),
        }


def make_train_env(config: dict[str, Any], run_dir: Path, seed: int) -> gym.Env:
    """Training env, optionally recording video, with the configured obs type."""
    record_video = config["video"]["enabled"] and config["video"]["every_episodes"]
    render_mode = render_mode_for(config, "rgb_array" if record_video else None)
    env = gym.make(config["env"]["id"], render_mode=render_mode)

    if record_video:
        # Wrapped beneath the pixel pipeline so it records the real frames.
        every = config["video"]["every_episodes"]
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(run_dir / "videos"),
            episode_trigger=lambda ep: ep % every == 0,
            name_prefix=f"{config['algo']}-seed{seed}",
            disable_logger=True,
        )
    return apply_pixel_wrappers(env, config) if is_pixels(config) else env


def train_one_seed(config: dict[str, Any], seed: int) -> RunMetrics:
    total_steps = config["train"]["total_env_steps"]
    warmup_steps = config["train"]["warmup_steps"]
    retrain_every = config["train"]["retrain_every_episodes"]
    device = resolve_device(config["train"]["device"])
    threshold = config["benchmark"]["threshold"]
    eval_freq = config["eval"]["freq_env_steps"]
    obs_type = config["obs"]["type"]

    print(f"[pets/{obs_type}] seed {seed}: {total_steps:,} env steps on {device}")
    torch.manual_seed(seed)
    run = init_run(config, seed)

    run_dir = Path(config["output"]["dir"]) / obs_type / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(config, run_dir, seed)
    # Curve evals may render live for watching; the final eval and the latency
    # benchmark always use a plain env so they are not throttled to render fps.
    eval_env = make_env(
        config, render_mode="human" if config["eval"]["render"] else None
    )
    measure_env = make_env(config)
    train_env.action_space.seed(seed)

    obs_shape = train_env.observation_space.shape
    act_dim = train_env.action_space.shape[0]
    max_torque = float(train_env.action_space.high[0])

    representation_cls = PixelRepresentation if is_pixels(config) else StateRepresentation
    representation = representation_cls(obs_shape, act_dim, config, device, max_torque)

    model = ProbabilisticEnsemble(
        representation.plan_dim, act_dim, config["model"], device,
        projection=representation.projection,
    )
    planner = CEMPlanner(
        model, config, -max_torque, max_torque, seed, representation.reward_fn
    )
    model_generator = torch.Generator(device=device).manual_seed(seed)

    # Replay buffer: every transition is kept, the budget is known up front.
    # For pixels these are uint8 frame stacks -- ~180 MB each at 15k steps.
    buf_obs = np.zeros((total_steps, *obs_shape), dtype=representation.buffer_dtype)
    buf_next = np.zeros((total_steps, *obs_shape), dtype=representation.buffer_dtype)
    buf_act = np.zeros((total_steps, act_dim), dtype=np.float32)
    buf_reward = np.zeros(total_steps, dtype=np.float32)
    # Marks episode ends so rollout training never unrolls across a reset.
    buf_done = np.zeros(total_steps, dtype=bool)

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return planner.plan(representation.encode(obs))

    learning_curve: list[dict[str, float]] = []
    timer = PausableTimer()

    def record_curve_point(env_steps: int) -> None:
        with timer.pause():
            mean, std, _ = evaluate(
                act_fn, eval_env, config, config["eval"]["curve_episodes"], seed
            )
        learning_curve.append(
            {"env_steps": int(env_steps), "return_mean": mean, "return_std": std}
        )
        log_eval_point(run, env_steps, mean, std)
        print(f"  {env_steps:>6,} steps | return {mean:8.1f} +/- {std:5.1f}", flush=True)

    # Baseline point: an untrained model, so the curve starts where learning did.
    record_curve_point(env_steps=0)

    timer.start()
    env_steps = 0
    episode = 0
    obs, _ = train_env.reset(seed=seed)

    # Outer loop = one episode; the model is refit at episode boundaries, while
    # the planner re-plans from the current state at every single step.
    while env_steps < total_steps:
        planning = env_steps >= warmup_steps
        if planning and episode % retrain_every == 0:
            # Learn the planning space first, then the dynamics over it.
            stats = representation.fit(
                buf_obs[:env_steps], buf_act[:env_steps], buf_next[:env_steps],
                buf_reward[:env_steps], model_generator,
            )
            stats.update(train_model(
                model,
                representation.encode(buf_obs[:env_steps]),
                buf_act[:env_steps],
                representation.encode(buf_next[:env_steps]),
                config["model"],
                model_generator,
                episode_ends=buf_done[:env_steps],
            ))
            log_scalars(run, env_steps, stats)
        planner.reset()

        done = False
        while not done and env_steps < total_steps:
            action = act_fn(obs) if planning else train_env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = train_env.step(action)

            buf_obs[env_steps] = obs
            buf_act[env_steps] = action
            buf_next[env_steps] = next_obs
            buf_reward[env_steps] = reward
            done = terminated or truncated
            buf_done[env_steps] = done
            env_steps += 1
            obs = next_obs

            if env_steps % eval_freq == 0:
                record_curve_point(env_steps)

        episode += 1
        obs, _ = train_env.reset()

    train_wall_clock_s = timer.stop()

    final_mean, final_std, _ = evaluate(
        act_fn, measure_env, config, config["eval"]["final_episodes"], seed
    )
    inference = benchmark_inference(act_fn, measure_env, config, seed)
    torch.save(
        {"model": model.state_dict(), **representation.state_dict()},
        run_dir / "model.pt",
    )

    metrics = RunMetrics(
        algo=config["algo"],
        env_id=config["env"]["id"],
        seed=seed,
        device=device,
        obs_type=obs_type,
        total_env_steps=env_steps,
        threshold=threshold,
        env_steps_to_threshold=steps_to_threshold(learning_curve, threshold),
        train_wall_clock_s=train_wall_clock_s,
        eval_wall_clock_s=timer.paused_s,
        final_return_mean=final_mean,
        final_return_std=final_std,
        final_return_episodes=config["eval"]["final_episodes"],
        inference_ms_per_action=inference,
        learning_curve=learning_curve,
        hyperparams={
            **config["model"], **config["planner"], **config["train"],
            **config.get("encoder", {}), **config["obs"],
        },
        versions=collect_versions(),
    )
    metrics.save(run_dir / "metrics.json")
    finish_run(run, metrics)

    train_env.close()
    eval_env.close()
    measure_env.close()

    reached = metrics.env_steps_to_threshold
    print(
        f"[pets/{obs_type}] seed {seed}: final return {final_mean:.1f} +/- {final_std:.1f}"
        f" | train {train_wall_clock_s:.1f}s (eval overhead {timer.paused_s:.1f}s) | "
        f"threshold {threshold:g} at "
        + (f"{reached:,} steps" if reached is not None else "NOT REACHED")
    )
    return metrics


def main() -> None:
    args = build_parser(__doc__, "config/pets.yaml").parse_args()
    config = load_and_override(args)
    for seed in args.seeds or config["benchmark"]["seeds"]:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()
