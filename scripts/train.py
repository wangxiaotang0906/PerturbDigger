from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perturbdigger.config import load_config
from perturbdigger.data import load_dataset_bundle
from perturbdigger.training import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PerturbDigger on a perturbation dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_dataset_bundle(config["data"]["root"])
    runner = ExperimentRunner(config)
    print(f"Using device: {runner.device_summary}")
    summary = runner.run(bundle)
    print("Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
