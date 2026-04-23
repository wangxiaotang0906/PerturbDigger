from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} must be a mapping.")

    base_path = raw.pop("base", None)
    if base_path is None:
        return raw

    base_cfg = load_config(path.parent / base_path)
    return _merge_dicts(base_cfg, raw)


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    output_root = Path(config["experiment"]["output_dir"])
    paths = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "explanations": output_root / "explanations",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
