"""Observation pipeline: the 3-d state vector, or a stack of rendered frames.

The pixel pipeline is built from gymnasium's own wrappers and is shared by both
trainers and by evaluation, so every agent sees byte-identical observations.

A stack of 3 frames is what makes the task solvable from vision at all: one
frame fixes the angle, two are needed for angular velocity, three for angular
acceleration.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium.wrappers import (
    AddRenderObservation,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)

PIXELS = "pixels"
STATE = "state"


def is_pixels(config: dict[str, Any]) -> bool:
    return config["obs"]["type"] == PIXELS


def apply_pixel_wrappers(env: gym.Env, config: dict[str, Any]) -> gym.Env:
    """Wrap a `render_mode="rgb_array"` env so observations are stacked frames.

    Args:
        env: the raw env, which MUST have been created with
            `render_mode="rgb_array"` -- the render is what becomes the
            observation.
        config: full config; the `obs` block supplies `image_size`,
            `grayscale` and `frame_stack`.

    Returns:
        The wrapped env. With the defaults (64px, grayscale, 3 frames) its
        observations are `(3, 64, 64)` uint8, which is channel-first and is
        recognised by SB3 as an image space.
    """
    obs_config = config["obs"]
    size = obs_config["image_size"]

    env = AddRenderObservation(env, render_only=True)
    env = ResizeObservation(env, (size, size))
    if obs_config["grayscale"]:
        env = GrayscaleObservation(env, keep_dim=False)
    return FrameStackObservation(env, obs_config["frame_stack"])


def render_mode_for(config: dict[str, Any], requested: str | None) -> str | None:
    """Resolve the env's render_mode, given what the caller wants.

    Pixel runs must render to `rgb_array` because that render *is* the
    observation, so a request for a live "human" window cannot be honoured.
    """
    if is_pixels(config):
        return "rgb_array"
    return requested
