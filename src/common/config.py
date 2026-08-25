"""YAML config loading.

All settings -- env id, hyperparameters, eval protocol, thresholds -- live in
`config/*.yaml`, never in the Python modules. An algorithm config declares
`inherits: base.yaml` and is deep-merged on top of it, so the shared
evaluation protocol is defined exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a config file, resolving its `inherits` chain relative to itself."""
    path = Path(path)
    config = yaml.safe_load(path.read_text()) or {}

    parent_name = config.pop("inherits", None)
    if parent_name is None:
        return config
    return deep_merge(load_config(path.parent / parent_name), config)


def resolve_device(name: str) -> str:
    """Return the requested device, falling back to cpu if cuda is missing.

    The resolved value is what gets recorded in metrics.json, so the reported
    wall-clock time always names the hardware it was actually measured on.
    """
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[config] {name} requested but cuda is unavailable; using cpu")
        return "cpu"
    return name
