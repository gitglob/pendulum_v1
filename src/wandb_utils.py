"""Weights & Biases logging, shared by the model-free and model-based runs.

Every helper degrades to a no-op when wandb is disabled in the config or not
usable (not logged in, no network), so a training run never fails because of
logging.
"""

from __future__ import annotations

from typing import Any

from .metrics import RunMetrics


def init_run(config: dict[str, Any], seed: int) -> Any | None:
    """Start a wandb run, or return None if logging is off/unavailable."""
    wandb_config = config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    try:
        import wandb

        return wandb.init(
            project=wandb_config.get("project"),
            entity=wandb_config.get("entity"),
            mode=wandb_config.get("mode", "online"),
            # picks up the mp4s VecVideoRecorder writes
            monitor_gym=config.get("video", {}).get("enabled", False),
            group=config["algo"],
            name=f"{config['algo']}-seed{seed}",
            tags=[config["algo"], config["env"]["id"]],
            config={**config, "seed": seed},
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        print(f"[wandb] disabled ({type(exc).__name__}: {exc})")
        return None


def log_eval_point(run: Any | None, env_steps: int, mean: float, std: float) -> None:
    """Log one learning-curve point, x-axis = real environment steps.

    Using env steps (not gradient updates or episodes) as the x-axis is what
    makes the PPO and PETS curves directly comparable on sample efficiency.
    """
    if run is None:
        return
    run.log(
        {"eval/return_mean": mean, "eval/return_std": std, "env_steps": env_steps},
        step=env_steps,
    )


def finish_run(run: Any | None, metrics: RunMetrics) -> None:
    """Push the headline metrics into the run summary and close the run."""
    if run is None:
        return
    run.summary.update(
        {
            "final_return_mean": metrics.final_return_mean,
            "final_return_std": metrics.final_return_std,
            "env_steps_to_threshold": metrics.env_steps_to_threshold,
            "train_wall_clock_s": metrics.train_wall_clock_s,
            "eval_wall_clock_s": metrics.eval_wall_clock_s,
            "inference_ms_per_action": metrics.inference_ms_per_action["mean"],
            "total_env_steps": metrics.total_env_steps,
        }
    )
    run.finish()
