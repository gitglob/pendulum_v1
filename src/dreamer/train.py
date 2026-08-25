"""DreamerV3 on Pendulum-v1 from pixels.

Alternates between collecting real experience with the current policy and
training on replayed sequences: the world model learns to predict, then the
actor and critic learn from trajectories imagined inside it, never touching the
environment. Every setting comes from `config/dreamer.yaml`.

Usage:
    .venv/bin/python -m src.dreamer.train
    .venv/bin/python -m src.dreamer.train --timesteps 2000 --no-video --no-wandb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from ..common.cli import build_parser, load_and_override
from ..common.config import resolve_device
from ..common.evaluate import benchmark_inference, evaluate, make_env
from ..common.metrics import (
    PausableTimer,
    RunMetrics,
    collect_versions,
    steps_to_threshold,
)
from ..common.obs import apply_pixel_wrappers, is_pixels, render_mode_for
from ..common.wandb_utils import finish_run, init_run, log_eval_point, log_scalars
from .agent import ActorCritic, DreamerAgent, WorldModel


class SequenceReplay:
    """Replay buffer holding whole episodes, sampled as fixed-length chunks.

    The world model is trained on sequences, not independent transitions: the
    RSSM has to build up recurrent state before its predictions mean anything.

    Args:
        capacity: total env steps to hold.
        obs_shape: observation shape per step.
        act_dim: action width.
        length: chunk length sampled for training.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], act_dim: int, length: int):
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.length = length
        self.size = 0
        # Start index of each completed episode, so chunks never straddle a reset.
        self.episode_starts: list[int] = []
        self._current_start = 0

    def add(self, obs: np.ndarray, act: np.ndarray, reward: float) -> None:
        self.obs[self.size] = obs
        self.act[self.size] = act
        self.reward[self.size] = reward
        self.size += 1

    def end_episode(self) -> None:
        if self.size - self._current_start >= self.length:
            self.episode_starts.append(self._current_start)
        self._current_start = self.size

    def ready(self) -> bool:
        return bool(self.episode_starts)

    def sample(self, batch_size: int, rng: np.random.Generator, device: str):
        """Draw `batch_size` chunks of `length` consecutive steps.

        Returns:
            `(obs, actions, rewards)` tensors on `device`, shaped
            [batch, length, ...].
        """
        starts = []
        for _ in range(batch_size):
            episode_start = self.episode_starts[rng.integers(len(self.episode_starts))]
            # Episodes are a fixed 200 steps here; clamp anyway so a shorter
            # trailing episode can never run past its own end.
            episode_end = min(episode_start + 200, self.size)
            latest = max(episode_start, episode_end - self.length)
            starts.append(rng.integers(episode_start, latest + 1))

        index = np.stack([np.arange(s, s + self.length) for s in starts])
        as_tensor = lambda x: torch.as_tensor(x, device=device)
        return (
            as_tensor(self.obs[index]),
            as_tensor(self.act[index]),
            as_tensor(self.reward[index]),
        )


def make_train_env(config: dict[str, Any], run_dir: Path, seed: int) -> gym.Env:
    """Training env with the pixel pipeline, optionally recording video."""
    record_video = config["video"]["enabled"] and config["video"]["every_episodes"]
    env = gym.make(
        config["env"]["id"], render_mode=render_mode_for(config, "rgb_array")
    )
    if record_video:
        every = config["video"]["every_episodes"]
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(run_dir / "videos"),
            episode_trigger=lambda ep: ep % every == 0,
            name_prefix=f"{config['algo']}-seed{seed}",
            disable_logger=True,
        )
    return apply_pixel_wrappers(env, config)


def train_one_seed(config: dict[str, Any], seed: int) -> RunMetrics:
    if not is_pixels(config):
        raise SystemExit(
            "dreamer is a pixel agent here: --obs-type state would need an MLP "
            "encoder/decoder, which is not implemented"
        )

    train_config = config["train"]
    dreamer_config = config["dreamer"]
    total_steps = train_config["total_env_steps"]
    warmup_steps = train_config["warmup_steps"]
    device = resolve_device(train_config["device"])
    threshold = config["benchmark"]["threshold"]
    eval_freq = config["eval"]["freq_env_steps"]
    obs_type = config["obs"]["type"]

    print(f"[dreamer/{obs_type}] seed {seed}: {total_steps:,} env steps on {device}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    run = init_run(config, seed)

    run_dir = Path(config["output"]["dir"]) / obs_type / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(config, run_dir, seed)
    eval_env = make_env(config)
    measure_env = make_env(config)
    train_env.action_space.seed(seed)

    obs_shape = train_env.observation_space.shape
    act_dim = train_env.action_space.shape[0]
    max_action = float(train_env.action_space.high[0])

    world_model = WorldModel(obs_shape, act_dim, dreamer_config, device)
    actor_critic = ActorCritic(
        world_model.rssm.feature_dim, act_dim, max_action, dreamer_config, device
    )
    model_opt = torch.optim.Adam(
        world_model.parameters(), lr=dreamer_config["model_lr"],
        eps=dreamer_config["adam_eps"],
    )
    actor_opt = torch.optim.Adam(
        actor_critic.actor.parameters(), lr=dreamer_config["actor_lr"],
        eps=dreamer_config["adam_eps"],
    )
    critic_opt = torch.optim.Adam(
        actor_critic.critic.parameters(), lr=dreamer_config["critic_lr"],
        eps=dreamer_config["adam_eps"],
    )

    replay = SequenceReplay(
        total_steps, obs_shape, act_dim, train_config["batch_length"]
    )
    # Exploration acts by sampling the policy; evaluation uses its mode.
    explore_agent = DreamerAgent(world_model, actor_critic, act_dim, sample=True)
    act_fn = DreamerAgent(world_model, actor_critic, act_dim, sample=False)

    # One gradient step per this many env steps (train_ratio counts replayed
    # steps, and one batch covers batch_size * batch_length of them).
    steps_per_batch = train_config["batch_size"] * train_config["batch_length"]
    env_steps_per_train = max(1, steps_per_batch // train_config["train_ratio"])

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

    def train_step() -> dict[str, float]:
        obs, actions, rewards = replay.sample(
            train_config["batch_size"], rng, device
        )
        model_loss, states, metrics = world_model.loss(obs, actions, rewards)
        model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            world_model.parameters(), dreamer_config["grad_clip"]
        )
        model_opt.step()

        # Imagine from every posterior state in the batch, detached so actor
        # gradients never leak into the world model.
        start = {k: v.reshape(-1, v.shape[-1]).detach() for k, v in states.items()}
        actor_loss, critic_loss, ac_metrics = actor_critic.loss(world_model, start)

        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        (actor_loss + critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            actor_critic.parameters(), dreamer_config["grad_clip"]
        )
        actor_opt.step()
        critic_opt.step()
        actor_critic.update_slow_critic()

        return {**metrics, **ac_metrics, "world/loss": float(model_loss.detach())}

    # Baseline point: an untrained agent, so the curve starts where learning did.
    record_curve_point(env_steps=0)

    timer.start()
    env_steps = 0
    obs, _ = train_env.reset(seed=seed)
    explore_agent.reset()

    while env_steps < total_steps:
        if env_steps < warmup_steps:
            action = train_env.action_space.sample()
        else:
            action = explore_agent(obs)

        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        replay.add(obs, action, reward)
        env_steps += 1
        obs = next_obs

        if terminated or truncated:
            replay.end_episode()
            obs, _ = train_env.reset()
            explore_agent.reset()

        if (
            env_steps >= warmup_steps
            and replay.ready()
            and env_steps % env_steps_per_train == 0
        ):
            stats = train_step()
            if env_steps % (env_steps_per_train * 100) == 0:
                log_scalars(run, env_steps, stats)

        if env_steps % eval_freq == 0:
            record_curve_point(env_steps)

    train_wall_clock_s = timer.stop()

    final_mean, final_std, _ = evaluate(
        act_fn, measure_env, config, config["eval"]["final_episodes"], seed
    )
    inference = benchmark_inference(act_fn, measure_env, config, seed)
    torch.save(
        {"world_model": world_model.state_dict(),
         "actor_critic": actor_critic.state_dict()},
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
        hyperparams={**dreamer_config, **train_config, **config["obs"]},
        versions=collect_versions(),
    )
    metrics.save(run_dir / "metrics.json")
    finish_run(run, metrics)

    train_env.close()
    eval_env.close()
    measure_env.close()

    reached = metrics.env_steps_to_threshold
    print(
        f"[dreamer/{obs_type}] seed {seed}: final return {final_mean:.1f} +/- "
        f"{final_std:.1f} | train {train_wall_clock_s:.1f}s (eval overhead "
        f"{timer.paused_s:.1f}s) | threshold {threshold:g} at "
        + (f"{reached:,} steps" if reached is not None else "NOT REACHED")
    )
    return metrics


def main() -> None:
    args = build_parser(__doc__, "config/dreamer.yaml").parse_args()
    config = load_and_override(args)
    for seed in args.seeds or config["benchmark"]["seeds"]:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()
