from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def resolve_device(device_preference: str | None = None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to resolve the runtime device.") from exc

    if device_preference is None or device_preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = torch.device(device_preference)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def get_device_summary(device) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"device": str(device), "torch_available": False}

    summary: Dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        summary["cuda_version"] = torch.version.cuda
        summary["device_count"] = torch.cuda.device_count()
        summary["device_name"] = torch.cuda.get_device_name(device.index or 0)
    return summary


def dump_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def dump_jsonl(rows: list[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
