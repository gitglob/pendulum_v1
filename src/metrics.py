"""Shared metrics schema and timing helpers.

Both the model-free (PPO) and model-based (PETS) runs write the exact same
JSON schema so that `compare.py` can build the report table without knowing
anything about how either algorithm was trained.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMetrics:
    """One training run of one algorithm at one seed."""

    algo: str
    env_id: str
    seed: int
    device: str
    total_env_steps: int
    threshold: float
    env_steps_to_threshold: int | None
    train_wall_clock_s: float
    eval_wall_clock_s: float
    final_return_mean: float
    final_return_std: float
    final_return_episodes: int
    inference_ms_per_action: dict[str, float]
    learning_curve: list[dict[str, float]]
    hyperparams: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunMetrics":
        return cls(**json.loads(Path(path).read_text()))


def steps_to_threshold(
    learning_curve: list[dict[str, float]], threshold: float
) -> int | None:
    """Env steps at the first eval point whose mean return reaches `threshold`.

    Returns None when the threshold was never reached, so that "not reached"
    is never silently reported as 0.
    """
    for point in learning_curve:
        if point["return_mean"] >= threshold:
            return int(point["env_steps"])
    return None


class PausableTimer:
    """Wall-clock timer that can be paused for evaluation.

    Training time and evaluation time are reported separately: the eval
    schedule is a measurement choice, not part of the cost of training, and
    including it would make wall-clock depend on how often we happened to
    evaluate.
    """

    def __init__(self) -> None:
        self._elapsed = 0.0
        self._started_at: float | None = None
        self._pause_started = 0.0
        self._was_running = False
        self.paused_s = 0.0

    def start(self) -> "PausableTimer":
        self._started_at = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._started_at is not None:
            self._elapsed += time.perf_counter() - self._started_at
            self._started_at = None
        return self._elapsed

    @property
    def elapsed(self) -> float:
        running = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
        return self._elapsed + running

    def pause(self) -> "PausableTimer":
        """Use as `with timer.pause():` — time inside is excluded from `elapsed`."""
        return self

    def __enter__(self) -> "PausableTimer":
        self._was_running = self._started_at is not None
        self.stop()
        self._pause_started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.paused_s += time.perf_counter() - self._pause_started
        if self._was_running:
            self._started_at = time.perf_counter()


def collect_versions() -> dict[str, str]:
    """Record library versions so results stay interpretable later."""
    versions = {"python": platform.python_version()}
    for module_name, key in [
        ("torch", "torch"),
        ("gymnasium", "gymnasium"),
        ("stable_baselines3", "stable_baselines3"),
        ("numpy", "numpy"),
    ]:
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[key] = "not installed"
    return versions
