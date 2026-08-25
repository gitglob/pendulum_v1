"""Model-free PPO baseline on Pendulum-v1 (stable-baselines3).

This is the comparison baseline for the model-based (PETS) agent, not the
subject of the report. Every setting comes from `config/ppo.yaml`.

Usage:
    .venv/bin/python -m src.ppo_train
    .venv/bin/python -m src.ppo_train --config config/ppo.yaml --no-wandb
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder

from ..common.cli import build_parser, load_and_override
from ..common.config import resolve_device
from ..common.evaluate import benchmark_inference, evaluate, make_env
from ..common.metrics import PausableTimer, RunMetrics, collect_versions, steps_to_threshold
from ..common.obs import apply_pixel_wrappers, is_pixels, render_mode_for
from ..common.wandb_utils import finish_run, init_run, log_eval_point


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
    timesteps = config["train"]["total_env_steps"]
    device = resolve_device(config["train"]["device"])
    threshold = config["benchmark"]["threshold"]

    obs_type = config["obs"]["type"]
    print(f"[ppo/{obs_type}] seed {seed}: training for {timesteps:,} env steps on {device}")
    set_random_seed(seed)
    run = init_run(config, seed)

    run_dir = Path(config["output"]["dir"]) / obs_type / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    n_envs = config["train"]["n_envs"]
    record_video = config["video"]["enabled"] and config["video"]["every_env_steps"]
    # Pixel runs must render to build their observations; video needs the same.
    render_mode = render_mode_for(config, "rgb_array" if record_video else None)
    train_env = make_vec_env(
        config["env"]["id"],
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"render_mode": render_mode} if render_mode else None,
        wrapper_class=(
            (lambda env: apply_pixel_wrappers(env, config)) if is_pixels(config) else None
        ),
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
        obs_type=obs_type,
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
        hyperparams={**config["hyperparams"], **config["train"], **config["obs"]},
        versions=collect_versions(),
    )
    metrics.save(run_dir / "metrics.json")
    finish_run(run, metrics)

    train_env.close()
    eval_env.close()
    measure_env.close()

    reached = metrics.env_steps_to_threshold
    print(
        f"[ppo/{obs_type}] seed {seed}: final return {final_mean:.1f} +/- {final_std:.1f} | "
        f"train {train_wall_clock_s:.1f}s (eval overhead {timer.paused_s:.1f}s) | "
        f"threshold {threshold:g} at "
        + (f"{reached:,} steps" if reached is not None else "NOT REACHED")
    )
    return metrics


def main() -> None:
    args = build_parser(__doc__, "config/ppo.yaml").parse_args()
    config = load_and_override(args)
    for seed in args.seeds or config["benchmark"]["seeds"]:
        train_one_seed(config, seed)


if __name__ == "__main__":
    main()
