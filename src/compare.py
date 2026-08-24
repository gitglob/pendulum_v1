"""Build the model-based vs model-free comparison table from saved runs.

Reads every `results/<algo>/seed*/metrics.json` and prints the five report
metrics as a markdown table. It never retrains anything -- the table is a pure
function of what is on disk -- so the README can be regenerated at any time.

Usage:
    .venv/bin/python -m src.compare
    .venv/bin/python -m src.compare --plot results/learning_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import load_config
from .metrics import RunMetrics

# Algorithms in report order: the model-based agent is the subject, PPO is the
# baseline it is measured against.
ALGO_LABELS = {"pets": "PETS (model-based)", "ppo": "PPO (model-free)"}


def load_runs(results_root: Path) -> dict[str, list[RunMetrics]]:
    """Group every metrics.json under `results_root` by algorithm."""
    runs: dict[str, list[RunMetrics]] = {}
    for path in sorted(results_root.glob("*/seed*/metrics.json")):
        metrics = RunMetrics.load(path)
        runs.setdefault(metrics.algo, []).append(metrics)
    return runs


def _agg(values: list[float], fmt: str) -> str:
    """Format a metric across seeds: bare value for one seed, mean+/-std beyond."""
    array = np.asarray(values, dtype=float)
    if array.size == 1:
        return format(array[0], fmt)
    return f"{format(array.mean(), fmt)} +/- {format(array.std(), fmt)}"


def _steps_to_threshold(runs: list[RunMetrics]) -> str:
    reached = [r.env_steps_to_threshold for r in runs if r.env_steps_to_threshold is not None]
    if not reached:
        return "not reached"
    suffix = "" if len(reached) == len(runs) else f" ({len(reached)}/{len(runs)} seeds)"
    return _agg([float(s) for s in reached], ",.0f") + suffix


def build_table(runs: dict[str, list[RunMetrics]]) -> str:
    algos = [a for a in ALGO_LABELS if a in runs] + [a for a in runs if a not in ALGO_LABELS]
    if not algos:
        return "No runs found."

    threshold = runs[algos[0]][0].threshold
    rows = [
        ("Final return (mean +/- std)", lambda rs: _agg([r.final_return_mean for r in rs], ".1f")),
        (f"Env steps to return >= {threshold:g}", _steps_to_threshold),
        ("Env steps trained", lambda rs: _agg([float(r.total_env_steps) for r in rs], ",.0f")),
        ("Training wall-clock (s)", lambda rs: _agg([r.train_wall_clock_s for r in rs], ".1f")),
        ("Inference (ms / action)", lambda rs: _agg([r.inference_ms_per_action["mean"] for r in rs], ".3f")),
        ("Device", lambda rs: rs[0].device),
        ("Seeds", lambda rs: str(len(rs))),
    ]

    header = "| Metric | " + " | ".join(ALGO_LABELS.get(a, a) for a in algos) + " |"
    divider = "| --- | " + " | ".join("---" for _ in algos) + " |"
    lines = [header, divider]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(runs[a]) for a in algos) + " |")
    return "\n".join(lines)


def plot_curves(runs: dict[str, list[RunMetrics]], out_path: Path) -> None:
    """Sample-efficiency plot: return vs REAL env steps, the headline claim."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo, algo_runs in runs.items():
        # Seeds may hit eval points at slightly different steps; plot each seed
        # and label only the first so the legend stays one entry per algorithm.
        for i, run in enumerate(algo_runs):
            steps = [p["env_steps"] for p in run.learning_curve]
            means = [p["return_mean"] for p in run.learning_curve]
            stds = [p["return_std"] for p in run.learning_curve]
            ax.plot(steps, means, label=ALGO_LABELS.get(algo, algo) if i == 0 else None)
            ax.fill_between(
                steps,
                np.subtract(means, stds),
                np.add(means, stds),
                alpha=0.15,
            )

    threshold = next(iter(runs.values()))[0].threshold
    ax.axhline(threshold, ls="--", c="gray", lw=1, label=f"threshold ({threshold:g})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episodic return")
    ax.set_title("Sample efficiency on Pendulum-v1")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[compare] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/base.yaml"))
    parser.add_argument("--plot", type=Path, nargs="?", const=Path("results/learning_curves.png"))
    args = parser.parse_args()

    results_root = Path(load_config(args.config)["output"]["root"])
    runs = load_runs(results_root)
    if not runs:
        print(f"No metrics.json found under {results_root}/. Train something first.")
        return

    print(build_table(runs))
    if args.plot is not None:
        plot_curves(runs, args.plot)


if __name__ == "__main__":
    main()
