from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from perturbdigger.config import ensure_output_dirs, load_config
from perturbdigger.data import load_dataset_bundle
from perturbdigger.explain import aggregate_explanations, build_sample_explanations
from perturbdigger.graph import build_graph_specification
from perturbdigger.model import PerturbationResponseModel
from perturbdigger.training.trainer import _build_loader
from perturbdigger.utils import dump_json, dump_jsonl, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mechanistic subgraphs from a trained checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--split", type=str, default=None, help="Override explanation split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.split is not None:
        config["explanations"]["split"] = args.split
    output_dirs = ensure_output_dirs(config)

    device = resolve_device(str(config["training"].get("device", "auto")))
    bundle = load_dataset_bundle(config["data"]["root"])
    graph = build_graph_specification(bundle, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = PerturbationResponseModel(graph, config["model"], checkpoint["graph_weights"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    indices = bundle.perturbation_indices(config["explanations"]["split"])
    max_samples = config["explanations"].get("max_samples")
    if max_samples is not None and max_samples > 0 and len(indices) > int(max_samples):
        rng = np.random.default_rng(int(config["training"]["seed"]) + 3)
        indices = rng.choice(indices, size=int(max_samples), replace=False)
        indices.sort()
    loader = _build_loader(
        bundle,
        indices,
        batch_size=int(config["explanations"]["batch_size"]),
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    sample_records = []
    with torch.no_grad():
        for batch in loader:
            sample_indices = batch["index"].to(device)
            x = batch["x"].to(device)
            z = batch["z"].to(device)
            output = model(x, z, return_subgraph=True)
            sample_records.extend(
                build_sample_explanations(
                    bundle=bundle,
                    graph=graph,
                    sample_indices=sample_indices.cpu(),
                    tg_selected=output["tg_selected"].cpu(),
                    gp_selected=output["gp_selected"].cpu(),
                    topn=int(config["explanations"]["top_edges_per_sample"]),
                )
            )

    aggregated = aggregate_explanations(
        sample_records,
        topn=int(config["explanations"]["top_edges_per_condition"]),
    )
    dump_jsonl(sample_records, output_dirs["explanations"] / "sample_level.jsonl")
    dump_json({"conditions": aggregated}, output_dirs["explanations"] / "perturbation_level.json")
    print(f"Exported {len(sample_records)} sample explanations.")


if __name__ == "__main__":
    main()
