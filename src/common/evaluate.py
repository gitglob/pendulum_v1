"""Algorithm-agnostic evaluation and inference-latency measurement.

Everything here works on a plain `act_fn(obs) -> action` callable rather than
on a specific agent class, so the SB3 PPO policy and the PETS CEM planner are
measured by identical code on identical initial states. All numbers come from
the caller's config; nothing is hard-coded here.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .obs import apply_pixel_wrappers, is_pixels, render_mode_for

ActFn = Callable[[np.ndarray], np.ndarray]


def reset_agent(act_fn: ActFn) -> None:
    """Tell a stateful agent a fresh episode is starting, if it cares.

    Some agents carry state between steps -- Dreamer's recurrent latent, PETS'
    warm-started plan -- and leaking it across an episode boundary is a
    correctness bug, not a rounding error. Agents that need the signal expose a
    `reset()`; a plain policy lambda does not, and is left alone. Keeping the
    hook optional is what lets every agent share this one measurement path.
    """
    reset = getattr(act_fn, "reset", None)
    if callable(reset):
        reset()


def make_env(config: dict[str, Any], render_mode: str | None = None) -> gym.Env:
    """Build the env with the observation the config asks for.

    Under `obs.type: pixels` the observation is a stack of rendered frames, so
    `render_mode` is forced to "rgb_array" and a request for a live window
    cannot be honoured.
    """
    env = gym.make(config["env"]["id"], render_mode=render_mode_for(config, render_mode))
    return apply_pixel_wrappers(env, config) if is_pixels(config) else env


def evaluate(
    act_fn: ActFn,
    env: gym.Env,
    config: dict[str, Any],
    n_episodes: int,
    seed: int,
) -> tuple[float, float, list[float]]:
    """Run `n_episodes` deterministic episodes; return (mean, std, returns).

    Episode i always starts from seed `eval.seed_offset + seed + i`, so every
    algorithm and every eval point faces the same set of initial states and
    differences in return come from the policy, not from luck of the reset.

    Steps taken here are evaluation overhead and are deliberately NOT counted
    towards an algorithm's environment-sample budget.
    """
    seed_offset = config["eval"]["seed_offset"]
    returns: list[float] = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + seed + i)
        reset_agent(act_fn)
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(act_fn(obs))
            total += float(reward)
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns)), returns


def benchmark_inference(
    act_fn: ActFn,
    env: gym.Env,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """Per-action decision latency in milliseconds, measured on real states.

    Observations come from an actual rollout so the timing reflects the states
    the agent really visits. Only the `act_fn` call is timed -- env stepping is
    excluded. This is the metric where a one-shot policy forward pass and a
    CEM planning loop differ by orders of magnitude.
    """
    n_actions = config["inference"]["n_actions"]
    warmup = config["inference"]["warmup"]

    obs, _ = env.reset(seed=config["eval"]["seed_offset"] + seed)
    reset_agent(act_fn)
    for _ in range(warmup):
        obs, _, terminated, truncated, _ = env.step(act_fn(obs))
        if terminated or truncated:
            obs, _ = env.reset()
            reset_agent(act_fn)

    latencies_ms: list[float] = []
    for _ in range(n_actions):
        start = time.perf_counter()
        action = act_fn(obs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            reset_agent(act_fn)

    latencies = np.asarray(latencies_ms)
    return {
        "mean": float(latencies.mean()),
        "median": float(np.median(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "n_actions": int(n_actions),
    }
