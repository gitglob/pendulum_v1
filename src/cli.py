"""Shared command-line interface for the trainers.

Both `src/ppo_train.py` and `src/pets_train.py` take the same flags, and each
flag is nothing more than an override of a `config/*.yaml` key -- the YAML
stays the single source of truth for every setting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import deep_merge, load_config
from .obs import PIXELS, STATE


def build_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=Path(default_config))
    parser.add_argument(
        "--obs-type",
        choices=[STATE, PIXELS],
        help="observation the agent learns from (default: obs.type in the config)",
    )
    parser.add_argument(
        "--timesteps", type=int, help="override train.total_env_steps (for smoke tests)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", help="override benchmark.seeds")
    parser.add_argument("--device", help="override train.device")
    parser.add_argument("--out", type=Path, help="override output.dir")
    parser.add_argument(
        "--video-freq", type=int, help="override the video interval (0 disables)"
    )
    parser.add_argument("--no-video", action="store_true", help="disable video capture")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="open a live window and render the evaluation episodes as training goes",
    )
    parser.add_argument("--eval-freq", type=int, help="override eval.freq_env_steps")
    parser.add_argument(
        "--eval-episodes", type=int, help="override eval.curve_episodes"
    )
    parser.add_argument(
        "--final-episodes", type=int, help="override eval.final_episodes"
    )
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    return parser


def load_and_override(args: argparse.Namespace) -> dict[str, Any]:
    """Load the config named by `--config` and apply the flag overrides."""
    config = load_config(args.config)

    if args.obs_type is not None:
        config["obs"]["type"] = args.obs_type
    # The algo config carries its pixel settings in a `pixels:` block; fold it in
    # before the remaining flags, so an explicit flag still wins over it.
    pixel_overrides = config.pop(PIXELS, {})
    if config["obs"]["type"] == PIXELS:
        config = deep_merge(config, pixel_overrides)

    if args.timesteps is not None:
        config["train"]["total_env_steps"] = args.timesteps
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.out is not None:
        config["output"]["dir"] = str(args.out)
    if args.video_freq is not None:
        # The two recorders trigger on different units: SB3's VecVideoRecorder
        # on env steps, gymnasium's RecordVideo on episodes.
        key = "every_env_steps" if "every_env_steps" in config["video"] else "every_episodes"
        config["video"][key] = args.video_freq
    if args.no_video:
        config["video"]["enabled"] = False
    if args.watch:
        config["eval"]["render"] = True
    if args.eval_freq is not None:
        config["eval"]["freq_env_steps"] = args.eval_freq
    if args.eval_episodes is not None:
        config["eval"]["curve_episodes"] = args.eval_episodes
    if args.final_episodes is not None:
        config["eval"]["final_episodes"] = args.final_episodes
    if args.no_wandb:
        config["wandb"]["enabled"] = False

    return config
