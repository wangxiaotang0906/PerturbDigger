from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DemoDataConfig:
    output_dir: Path
    seed: int = 7
    num_genes: int = 48
    num_tfs: int = 10
    num_pathways: int = 12
    num_control: int = 128
    num_perturb: int = 256
    contexts: int = 4


def _topk_similarity_edges(gene_repr: np.ndarray, topk: int, threshold: float) -> pd.DataFrame:
    gene_repr = gene_repr / (np.linalg.norm(gene_repr, axis=1, keepdims=True) + 1e-8)
    sim = gene_repr @ gene_repr.T
    np.fill_diagonal(sim, -1.0)
    rows = []
    for dst in range(sim.shape[0]):
        neighbor_idx = np.argsort(sim[dst])[::-1][:topk]
        for src in neighbor_idx:
            if sim[dst, src] >= threshold:
                rows.append({"src": src, "dst": dst, "score": float(sim[dst, src])})
    return pd.DataFrame(rows)


def _sample_without_self(rng: np.random.Generator, upper: int, avoid: int, size: int) -> np.ndarray:
    choices = [idx for idx in range(upper) if idx != avoid]
    return rng.choice(choices, size=size, replace=False)


def generate_demo_dataset(config: DemoDataConfig) -> Dict[str, int]:
    rng = np.random.default_rng(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gene_names = [f"G{idx:03d}" for idx in range(config.num_genes)]
    tf_names = set(gene_names[: config.num_tfs])
    pathway_names = [f"P{idx:02d}" for idx in range(config.num_pathways)]

    gene_repr = rng.normal(size=(config.num_genes, 16)).astype(np.float32)
    gene_metadata = rng.normal(size=(config.num_genes, 8)).astype(np.float32)
    context_programs = rng.normal(scale=0.8, size=(config.contexts, config.num_genes)).astype(np.float32)
    perturb_embeddings = rng.normal(scale=0.5, size=(config.num_genes, config.num_genes)).astype(np.float32)

    genes_df = pd.DataFrame(
        {
            "gene": gene_names,
            "is_tf": [int(gene in tf_names) for gene in gene_names],
        }
    )

    gg_df = _topk_similarity_edges(gene_repr, topk=5, threshold=0.15)
    gg_df["src"] = gg_df["src"].map(lambda idx: gene_names[int(idx)])
    gg_df["dst"] = gg_df["dst"].map(lambda idx: gene_names[int(idx)])

    tg_rows: List[Dict[str, str]] = []
    tg_weight_map: Dict[Tuple[int, int], float] = {}
    for dst in range(config.num_genes):
        tf_count = int(rng.integers(1, min(4, config.num_tfs) + 1))
        tf_indices = rng.choice(np.arange(config.num_tfs), size=tf_count, replace=False)
        for src in tf_indices:
            tg_rows.append({"src": gene_names[int(src)], "dst": gene_names[dst]})
            tg_weight_map[(int(src), dst)] = float(rng.uniform(0.35, 0.95))
    tg_df = pd.DataFrame(tg_rows).drop_duplicates(ignore_index=True)

    gp_rows: List[Dict[str, str]] = []
    gene_to_pathways: Dict[int, List[int]] = {}
    for gene_idx in range(config.num_genes):
        membership = rng.choice(
            np.arange(config.num_pathways),
            size=int(rng.integers(1, 4)),
            replace=False,
        )
        gene_to_pathways[gene_idx] = membership.tolist()
        for pathway_idx in membership:
            gp_rows.append({"src": gene_names[gene_idx], "dst": pathway_names[int(pathway_idx)]})
    gp_df = pd.DataFrame(gp_rows).drop_duplicates(ignore_index=True)

    pp_rows: List[Dict[str, str]] = []
    for child in range(1, config.num_pathways):
        parent = int(rng.integers(0, child))
        pp_rows.append({"src": pathway_names[parent], "dst": pathway_names[child]})
    pp_df = pd.DataFrame(pp_rows)

    pathway_effect = rng.normal(scale=0.35, size=(config.num_pathways, config.num_genes)).astype(np.float32)
    pp_weight = rng.uniform(0.2, 0.7, size=len(pp_df)).astype(np.float32)

    def simulate_control_expression() -> np.ndarray:
        context_id = int(rng.integers(0, config.contexts))
        base = context_programs[context_id].copy()
        base += rng.normal(scale=0.2, size=config.num_genes)

        tf_influence = np.zeros(config.num_genes, dtype=np.float32)
        for _, row in tg_df.iterrows():
            src_idx = gene_names.index(row["src"])
            dst_idx = gene_names.index(row["dst"])
            tf_influence[dst_idx] += tg_weight_map[(src_idx, dst_idx)] * np.tanh(base[src_idx])
        base += 0.15 * tf_influence
        return base.astype(np.float32)

    def simulate_response(x_vec: np.ndarray, perturbed: List[int]) -> np.ndarray:
        z_vec = np.zeros(config.num_genes, dtype=np.float32)
        z_vec[perturbed] = 1.0
        seed = perturb_embeddings[perturbed].mean(axis=0)

        gg_signal = np.zeros(config.num_genes, dtype=np.float32)
        for _, row in gg_df.iterrows():
            src_idx = gene_names.index(row["src"])
            dst_idx = gene_names.index(row["dst"])
            gg_signal[dst_idx] += float(row["score"]) * seed[src_idx]

        tg_signal = np.zeros(config.num_genes, dtype=np.float32)
        for _, row in tg_df.iterrows():
            src_idx = gene_names.index(row["src"])
            dst_idx = gene_names.index(row["dst"])
            tg_signal[dst_idx] += tg_weight_map[(src_idx, dst_idx)] * np.tanh(seed[src_idx] + 0.2 * x_vec[src_idx])

        pathway_state = np.zeros(config.num_pathways, dtype=np.float32)
        for gene_idx in range(config.num_genes):
            for pathway_idx in gene_to_pathways[gene_idx]:
                pathway_state[pathway_idx] += 0.2 * tg_signal[gene_idx] + 0.15 * gg_signal[gene_idx]

        for edge_idx, row in pp_df.iterrows():
            src_idx = pathway_names.index(row["src"])
            dst_idx = pathway_names.index(row["dst"])
            pathway_state[dst_idx] += pp_weight[edge_idx] * np.tanh(pathway_state[src_idx])

        delta = 0.35 * gg_signal + 0.45 * tg_signal + pathway_state @ pathway_effect
        delta += 0.15 * z_vec
        delta += rng.normal(scale=0.05, size=config.num_genes)
        return (x_vec + delta).astype(np.float32)

    sample_rows: List[Dict[str, str | int]] = []
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []

    for idx in range(config.num_control):
        x_vec = simulate_control_expression()
        sample_rows.append(
            {
                "sample_id": f"ctrl_{idx:04d}",
                "split": "train" if idx < int(config.num_control * 0.85) else "val",
                "is_control": 1,
                "condition": "control",
                "perturbed_genes": "",
            }
        )
        x_rows.append(x_vec)
        y_rows.append(x_vec.copy())

    split_cutoffs = {
        "train": int(config.num_perturb * 0.7),
        "val": int(config.num_perturb * 0.15),
    }
    for idx in range(config.num_perturb):
        x_vec = simulate_control_expression()
        num_targets = int(rng.integers(1, 3))
        perturbed = rng.choice(np.arange(config.num_genes), size=num_targets, replace=False).tolist()
        y_vec = simulate_response(x_vec, perturbed)

        if idx < split_cutoffs["train"]:
            split = "train"
        elif idx < split_cutoffs["train"] + split_cutoffs["val"]:
            split = "val"
        else:
            split = "test"

        sample_rows.append(
            {
                "sample_id": f"pert_{idx:04d}",
                "split": split,
                "is_control": 0,
                "condition": "+".join(gene_names[g] for g in sorted(perturbed)),
                "perturbed_genes": ";".join(gene_names[g] for g in sorted(perturbed)),
            }
        )
        x_rows.append(x_vec)
        y_rows.append(y_vec)

    genes_df.to_csv(output_dir / "genes.csv", index=False)
    pd.DataFrame({"pathway": pathway_names}).to_csv(output_dir / "pathways.csv", index=False)
    gg_df.to_csv(output_dir / "edges_gg.csv", index=False)
    tg_df.to_csv(output_dir / "edges_tg.csv", index=False)
    gp_df.to_csv(output_dir / "edges_gp.csv", index=False)
    pp_df.to_csv(output_dir / "edges_pp.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output_dir / "samples.csv", index=False)
    np.save(output_dir / "x.npy", np.stack(x_rows, axis=0))
    np.save(output_dir / "y.npy", np.stack(y_rows, axis=0))
    np.save(output_dir / "gene_metadata.npy", gene_metadata)

    return {
        "num_genes": config.num_genes,
        "num_pathways": config.num_pathways,
        "num_samples": len(sample_rows),
        "num_control": config.num_control,
        "num_perturb": config.num_perturb,
    }
