from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    root: Path
    gene_names: List[str]
    pathway_names: List[str]
    gene_metadata: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sample_ids: List[str]
    split: np.ndarray
    is_control: np.ndarray
    conditions: List[str]
    perturbed_gene_lists: List[List[str]]
    perturbed_gene_indices: List[np.ndarray]
    edge_indices: Dict[str, np.ndarray]
    edge_frames: Dict[str, pd.DataFrame]

    @property
    def num_genes(self) -> int:
        return len(self.gene_names)

    @property
    def num_pathways(self) -> int:
        return len(self.pathway_names)

    def split_indices(self, split_name: str, perturbed_only: bool = False) -> np.ndarray:
        mask = self.split == split_name
        if perturbed_only:
            mask = mask & (~self.is_control.astype(bool))
        return np.nonzero(mask)[0]

    def control_indices(self) -> np.ndarray:
        return np.nonzero(self.is_control.astype(bool))[0]

    def perturbation_indices(self, split_name: str | None = None) -> np.ndarray:
        mask = ~self.is_control.astype(bool)
        if split_name is not None:
            mask = mask & (self.split == split_name)
        return np.nonzero(mask)[0]

    def x_rows(self, indices: Sequence[int]) -> np.ndarray:
        return np.asarray(self.x[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


class PerturbationDataset:
    def __init__(self, bundle: DatasetBundle, indices: Sequence[int]):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        idx = int(self.indices[item])
        z = np.zeros((self.bundle.num_genes,), dtype=np.float32)
        perturbed_idx = self.bundle.perturbed_gene_indices[idx]
        if perturbed_idx.size > 0:
            z[perturbed_idx] = 1.0
        return {
            "index": np.array(idx, dtype=np.int64),
            "x": self.bundle.x[idx].astype(np.float32),
            "y": self.bundle.y[idx].astype(np.float32),
            "z": z,
            "is_control": np.array(self.bundle.is_control[idx], dtype=np.float32),
        }


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _parse_perturbed_gene_lists(raw_values: Iterable[str]) -> List[List[str]]:
    gene_lists: List[List[str]] = []
    for value in raw_values:
        if pd.isna(value) or str(value).strip() == "":
            gene_lists.append([])
            continue
        genes = [token.strip() for token in str(value).split(";") if token.strip()]
        gene_lists.append(genes)
    return gene_lists


def _edge_to_index(
    frame: pd.DataFrame,
    source_col: str,
    target_col: str,
    source_map: Dict[str, int],
    target_map: Dict[str, int],
) -> np.ndarray:
    missing_sources = sorted(set(frame[source_col]) - set(source_map))
    missing_targets = sorted(set(frame[target_col]) - set(target_map))
    if missing_sources or missing_targets:
        raise ValueError(
            "Edge table contains unknown nodes. "
            f"Missing sources={missing_sources[:5]}, missing targets={missing_targets[:5]}"
        )
    src = frame[source_col].map(source_map).to_numpy(dtype=np.int64)
    dst = frame[target_col].map(target_map).to_numpy(dtype=np.int64)
    return np.stack([src, dst], axis=0)


def load_dataset_bundle(root: str | Path) -> DatasetBundle:
    root = Path(root)
    genes_df = _load_csv(root / "genes.csv")
    pathways_df = _load_csv(root / "pathways.csv")
    samples_df = _load_csv(root / "samples.csv")

    x = np.load(root / "x.npy", mmap_mode="r")
    y = np.load(root / "y.npy", mmap_mode="r")

    gene_names = genes_df["gene"].astype(str).tolist()
    pathway_names = pathways_df["pathway"].astype(str).tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    pathway_to_idx = {pathway: idx for idx, pathway in enumerate(pathway_names)}

    if x.shape != y.shape:
        raise ValueError(f"x and y shapes must match, got {x.shape} and {y.shape}")
    if x.shape[0] != len(samples_df):
        raise ValueError("samples.csv row count must match x/y first dimension.")
    if x.shape[1] != len(gene_names):
        raise ValueError("x/y second dimension must match number of genes.")

    if (root / "gene_metadata.npy").exists():
        gene_metadata = np.load(root / "gene_metadata.npy").astype(np.float32)
    else:
        meta_cols = [col for col in genes_df.columns if col.startswith("meta_")]
        gene_metadata = (
            genes_df[meta_cols].to_numpy(dtype=np.float32)
            if meta_cols
            else np.zeros((len(gene_names), 0), dtype=np.float32)
        )

    if gene_metadata.shape[0] != len(gene_names):
        raise ValueError("gene_metadata first dimension must match number of genes.")

    perturbed_gene_lists = _parse_perturbed_gene_lists(samples_df["perturbed_genes"].fillna(""))
    perturbed_gene_indices: List[np.ndarray] = []
    for genes in perturbed_gene_lists:
        indices = []
        for gene in genes:
            if gene not in gene_to_idx:
                raise ValueError(f"Unknown perturbed gene '{gene}' in samples.csv")
            indices.append(gene_to_idx[gene])
        perturbed_gene_indices.append(np.asarray(indices, dtype=np.int64))

    edge_frames = {
        "gg": _load_csv(root / "edges_gg.csv"),
        "tg": _load_csv(root / "edges_tg.csv"),
        "gp": _load_csv(root / "edges_gp.csv"),
        "pp": _load_csv(root / "edges_pp.csv"),
    }
    edge_indices = {
        "gg": _edge_to_index(edge_frames["gg"], "src", "dst", gene_to_idx, gene_to_idx),
        "tg": _edge_to_index(edge_frames["tg"], "src", "dst", gene_to_idx, gene_to_idx),
        "gp": _edge_to_index(edge_frames["gp"], "src", "dst", gene_to_idx, pathway_to_idx),
        "pp": _edge_to_index(edge_frames["pp"], "src", "dst", pathway_to_idx, pathway_to_idx),
    }

    return DatasetBundle(
        root=root,
        gene_names=gene_names,
        pathway_names=pathway_names,
        gene_metadata=gene_metadata,
        x=x,
        y=y,
        sample_ids=samples_df["sample_id"].astype(str).tolist(),
        split=samples_df["split"].astype(str).to_numpy(),
        is_control=samples_df["is_control"].astype(np.int64).to_numpy(),
        conditions=samples_df["condition"].astype(str).tolist(),
        perturbed_gene_lists=perturbed_gene_lists,
        perturbed_gene_indices=perturbed_gene_indices,
        edge_indices=edge_indices,
        edge_frames=edge_frames,
    )
