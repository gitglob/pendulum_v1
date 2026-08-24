"""Model-free PPO baseline on Pendulum-v1 (stable-baselines3).

This is the comparison baseline for the model-based (PETS) agent, not the
subject of the report. Every setting comes from `config/ppo.yaml`.

Usage:
    .venv/bin/python -m src.ppo_train
    .venv/bin/python -m src.ppo_train --config config/ppo.yaml --no-wandb
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder

from .config import load_config, resolve_device
from .evaluate import benchmark_inference, evaluate, make_env
from .metrics import PausableTimer, RunMetrics, collect_versions, steps_to_threshold
from .wandb_utils import finish_run, init_run, log_eval_point


class CurveCallback(BaseCallback):
    """Evaluate on a schedule and record the sample-efficiency curve.

    The training timer is paused for the duration of each evaluation so that
    reported wall-clock training time does not depend on how often we evaluate.
    """

    def __init__(self, eval_env, config: dict[str, Any], timer: PausableTimer,
                 seed: int, run: Any | None, run_dir: Path):
        super().__init__()
        self.eval_env = eval_env
        self.config = config
        self.timer = timer
        self.seed = seed
        self.run = run
        self.run_dir = run_dir
        self.eval_freq = config["eval"]["freq_env_steps"]
        self.n_episodes = config["eval"]["curve_episodes"]
        self.learning_curve: list[dict[str, float]] = []

    def _act_fn(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(obs, deterministic=True)[0]

    def record(self, env_steps: int) -> None:
        with self.timer.pause():
            mean, std, _ = evaluate(
                self._act_fn, self.eval_env, self.config, self.n_episodes, self.seed
            )
        self.learning_curve.append(
            {"env_steps": int(env_steps), "return_mean": mean, "return_std": std}
        )
        log_eval_point(self.run, env_steps, mean, std)
        print(f"  {env_steps:>7,} steps | return {mean:8.1f} +/- {std:5.1f}", flush=True)

    def _on_step(self) -> bool:
        last = self.learning_curve[-1]["env_steps"] if self.learning_curve else 0
        if self.num_timesteps - last >= self.eval_freq:
            self.record(self.num_timesteps)
        return True


def train_one_seed(config: dict[str, Any], seed: int) -> RunMetrics:
    timesteps = config["train"]["total_timesteps"]
    device = resolve_device(config["train"]["device"])
    threshold = config["benchmark"]["threshold"]

    print(f"[ppo] seed {seed}: training for {timesteps:,} env steps on {device}")
    set_random_seed(seed)
    run = init_run(config, seed)

    run_dir = Path(config["output"]["dir"]) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_envs = config["train"]["n_envs"]
    record_video = config["video"]["enabled"] and config["video"]["every_env_steps"]
    train_env = make_vec_env(
        config["env"]["id"],
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"render_mode": "rgb_array"} if record_video else None,
    )
    if record_video:
        # VecVideoRecorder counts vec-env steps, so divide by n_envs to trigger
        # on the env-step interval the config asks for.
        every = max(1, config["video"]["every_env_steps"] // n_envs)
        train_env = VecVideoRecorder(
            train_env,
            video_folder=str(run_dir / "videos"),
            record_video_trigger=lambda step: step % every == 0,
            video_length=config["video"]["length"],
            name_prefix=f"{config['algo']}-seed{seed}",
        )

    # Curve evals may render live for watching; the final eval and the latency
    # benchmark always use a plain env so they are not throttled to render fps.
    eval_env = make_env(
        config, render_mode="human" if config["eval"]["render"] else None
    )
    measure_env = make_env(config)
    model = PPO(
        env=train_env, seed=seed, device=device, verbose=0, **config["hyperparams"]
    )

    timer = PausableTimer()
    callback = CurveCallback(eval_env, config, timer, seed, run, run_dir)
    # Baseline point: an untrained policy, so the curve starts where learning did.
    callback.model = model
    callback.record(env_steps=0)

    timer.start()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    train_wall_clock_s = timer.stop()

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return model.predict(obs, deterministic=True)[0]

    final_mean, final_std, _ = evaluate(
        act_fn, measure_env, config, config["eval"]["final_episodes"], seed
    )
    inference = benchmark_inference(act_fn, measure_env, config, seed)
    model.save(run_dir / "model.zip")

    metrics = RunMetrics(
        algo=config["algo"],
        env_id=config["env"]["id"],
        seed=seed,
        device=device,
        total_env_steps=int(model.num_timesteps),
        threshold=threshold,
        env_steps_to_threshold=steps_to_threshold(callback.learning_curve, threshold),
        train_wall_clock_s=train_wall_clock_s,
        eval_wall_clock_s=timer.paused_s,
        final_return_mean=final_mean,
        final_return_std=final_std,
        final_return_episodes=config["eval"]["final_episodes"],
        inference_ms_per_action=inference,
        learning_curve=callback.learning_curve,
        hyperparams={**config["hyperparams"], **config["train"]},
        versions=collect_versions(),
    )
    metrics.save(run_dir / "metrics.json")
    finish_run(run, metrics)

    train_env.close()
    eval_env.close()
    measure_env.close()

    reached = metrics.env_steps_to_threshold
    print(
        f"[ppo] seed {seed}: final return {final_mean:.1f} +/- {final_std:.1f} | "
        f"train {train_wall_clock_s:.1f}s (eval overhead {timer.paused_s:.1f}s) | "
        f"threshold {threshold:g} at "
        + (f"{reached:,} steps" if reached is not None else "NOT REACHED")
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/ppo.yaml"))
    parser.add_argument(
        "--timesteps", type=int, help="override train.total_timesteps (for smoke tests)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="override benchmark.seeds")
    parser.add_argument("--device", help="override train.device")
    parser.add_argument("--out", type=Path, help="override output.dir")
    parser.add_argument(
        "--video-freq", type=int, help="override video.every_env_steps (0 disables)"
    )
    parser.add_argument("--no-video", action="store_true", help="disable video capture")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="open a live window and render the evaluation episodes as training goes",
    )
    parser.add_argument(
        "--eval-freq", type=int, help="override eval.freq_env_steps"
    )
    parser.add_argument(
        "--eval-episodes", type=int, help="override eval.curve_episodes"
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.timesteps is not None:
        config["train"]["total_timesteps"] = args.timesteps
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.out is not None:
        config["output"]["dir"] = str(args.out)
    if args.video_freq is not None:
        config["video"]["every_env_steps"] = args.video_freq
    if args.no_video:
        config["video"]["enabled"] = False
    if args.watch:
        config["eval"]["render"] = True
    if args.eval_freq is not None:
        config["eval"]["freq_env_steps"] = args.eval_freq
    if args.eval_episodes is not None:
        config["eval"]["curve_episodes"] = args.eval_episodes
    if args.no_wandb:
        config["wandb"]["enabled"] = False

    for seed in args.seeds or config["benchmark"]["seeds"]:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()
