from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perturbdigger.data.demo import DemoDataConfig, generate_demo_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the demo dataset for PerturbDigger.")
    parser.add_argument("--output", type=str, default="data/demo", help="Output directory.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = generate_demo_dataset(
        DemoDataConfig(
            output_dir=PROJECT_ROOT / args.output,
            seed=args.seed,
        )
    )
    print("Demo dataset generated.")
    print(stats)


if __name__ == "__main__":
    main()
