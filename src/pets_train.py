"""PETS on Pendulum-v1: probabilistic ensemble + CEM model-predictive control.

The model-based agent this project reports on. It alternates between fitting a
dynamics ensemble on everything collected so far and acting for one episode by
planning through that model at every step (MPC). Every setting comes from
`config/pets.yaml`.

Usage:
    .venv/bin/python -m src.pets_train
    .venv/bin/python -m src.pets_train --timesteps 600 --no-video --no-wandb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from .cli import build_parser, load_and_override
from .config import resolve_device
from .dynamics import ProbabilisticEnsemble, train_model
from .evaluate import benchmark_inference, evaluate, make_env
from .metrics import PausableTimer, RunMetrics, collect_versions, steps_to_threshold
from .planner import CEMPlanner
from .wandb_utils import finish_run, init_run, log_eval_point, log_scalars


def make_train_env(config: dict[str, Any], run_dir: Path, seed: int) -> gym.Env:
    """Training env, optionally wrapped in gymnasium's video recorder."""
    record_video = config["video"]["enabled"] and config["video"]["every_episodes"]
    env = gym.make(
        config["env"]["id"], render_mode="rgb_array" if record_video else None
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
    return env


def train_one_seed(config: dict[str, Any], seed: int) -> RunMetrics:
    total_steps = config["train"]["total_env_steps"]
    warmup_steps = config["train"]["warmup_steps"]
    device = resolve_device(config["train"]["device"])
    threshold = config["benchmark"]["threshold"]
    eval_freq = config["eval"]["freq_env_steps"]

    print(f"[pets] seed {seed}: {total_steps:,} env step budget on {device}")
    torch.manual_seed(seed)
    run = init_run(config, seed)

    run_dir = Path(config["output"]["dir"]) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(config, run_dir, seed)
    # Curve evals may render live for watching; the final eval and the latency
    # benchmark always use a plain env so they are not throttled to render fps.
    eval_env = make_env(
        config, render_mode="human" if config["eval"]["render"] else None
    )
    measure_env = make_env(config)
    train_env.action_space.seed(seed)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    model = ProbabilisticEnsemble(obs_dim, act_dim, config["model"], device)
    planner = CEMPlanner(
        model,
        config,
        float(train_env.action_space.low[0]),
        float(train_env.action_space.high[0]),
        seed,
    )
    model_generator = torch.Generator(device=device).manual_seed(seed)

    # Replay buffer: every transition is kept, the budget is known up front.
    buf_obs = np.zeros((total_steps, obs_dim), dtype=np.float32)
    buf_act = np.zeros((total_steps, act_dim), dtype=np.float32)
    buf_next = np.zeros((total_steps, obs_dim), dtype=np.float32)

    learning_curve: list[dict[str, float]] = []
    timer = PausableTimer()

    def record_curve_point(env_steps: int) -> None:
        with timer.pause():
            mean, std, _ = evaluate(
                planner.plan, eval_env, config, config["eval"]["curve_episodes"], seed
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
    obs, _ = train_env.reset(seed=seed)

    # Outer loop = one episode; the model is refit at episode boundaries, while
    # the planner re-plans from the current state at every single step.
    while env_steps < total_steps:
        planning = env_steps >= warmup_steps
        if planning:
            stats = train_model(
                model,
                buf_obs[:env_steps],
                buf_act[:env_steps],
                buf_next[:env_steps],
                config["model"],
                model_generator,
            )
            log_scalars(run, env_steps, stats)
        planner.reset()

        done = False
        while not done and env_steps < total_steps:
            action = planner.plan(obs) if planning else train_env.action_space.sample()
            next_obs, _, terminated, truncated, _ = train_env.step(action)

            buf_obs[env_steps] = obs
            buf_act[env_steps] = action
            buf_next[env_steps] = next_obs
            env_steps += 1
            obs = next_obs
            done = terminated or truncated

            if env_steps % eval_freq == 0:
                record_curve_point(env_steps)

        obs, _ = train_env.reset()

    train_wall_clock_s = timer.stop()

    final_mean, final_std, _ = evaluate(
        planner.plan, measure_env, config, config["eval"]["final_episodes"], seed
    )
    inference = benchmark_inference(planner.plan, measure_env, config, seed)
    torch.save(model.state_dict(), run_dir / "model.pt")

    metrics = RunMetrics(
        algo=config["algo"],
        env_id=config["env"]["id"],
        seed=seed,
        device=device,
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
        hyperparams={**config["model"], **config["planner"], **config["train"]},
        versions=collect_versions(),
    )
    metrics.save(run_dir / "metrics.json")
    finish_run(run, metrics)

    train_env.close()
    eval_env.close()
    measure_env.close()

    reached = metrics.env_steps_to_threshold
    print(
        f"[pets] seed {seed}: final return {final_mean:.1f} +/- {final_std:.1f} | "
        f"train {train_wall_clock_s:.1f}s (eval overhead {timer.paused_s:.1f}s) | "
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
