from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from perturbdigger.utils import dump_json


def _decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return str(value)


def _decode_dataset(dataset: h5py.Dataset) -> List[str]:
    return [_decode_value(item) for item in dataset[:]]


def _decode_categorical_obs(obs_group: h5py.Group, key: str) -> List[str]:
    categories = _decode_dataset(obs_group["__categories"][key])
    codes = obs_group[key][:]
    return [categories[int(code)] for code in codes]


def _load_expression_csr(h5ad_path: str | Path) -> tuple[sparse.csr_matrix, List[str], Dict[str, np.ndarray]]:
    with h5py.File(h5ad_path, "r") as handle:
        x_group = handle["X"]
        matrix = sparse.csr_matrix(
            (x_group["data"][:], x_group["indices"][:], x_group["indptr"][:]),
            shape=(len(handle["obs"]["control"]), len(handle["var"]["gene_name"])),
            dtype=np.float32,
        )
        genes = _decode_dataset(handle["var"]["gene_name"])
        obs_group = handle["obs"]
        obs = {
            "cell_barcode": np.asarray(_decode_dataset(obs_group["cell_barcode"]), dtype=object),
            "condition": np.asarray(_decode_categorical_obs(obs_group, "condition"), dtype=object),
            "condition_name": np.asarray(_decode_categorical_obs(obs_group, "condition_name"), dtype=object),
            "cell_type": np.asarray(_decode_categorical_obs(obs_group, "cell_type"), dtype=object),
            "dose_val": np.asarray(_decode_categorical_obs(obs_group, "dose_val"), dtype=object),
            "control": obs_group["control"][:].astype(np.int64),
        }
    return matrix, genes, obs


def _select_topk_edges(
    edge_frame: pd.DataFrame,
    topk: int,
    threshold: float,
    source_whitelist: set[str] | None = None,
    drop_self: bool = True,
) -> pd.DataFrame:
    frame = edge_frame.copy()
    if source_whitelist is not None:
        frame = frame[frame["src"].isin(source_whitelist)]
    if drop_self:
        frame = frame[frame["src"] != frame["dst"]]
    frame = frame[frame["importance"] >= threshold]
    frame = frame.sort_values(["dst", "importance", "src"], ascending=[True, False, True])
    frame = frame.groupby("dst", sort=False).head(topk).reset_index(drop=True)
    return frame


def _parse_reactome_pathways(
    gmt_path: str | Path,
    pathways_path: str | Path,
    relation_path: str | Path,
    selected_genes: Sequence[str],
    min_pathway_genes: int,
    max_pathway_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_gene_set = set(selected_genes)
    pathway_info = pd.read_csv(
        pathways_path,
        sep="\t",
        header=None,
        names=["pathway_id", "pathway_name", "species"],
    )
    human_info = pathway_info[pathway_info["species"] == "Homo sapiens"].copy()
    human_name_map = dict(zip(human_info["pathway_id"], human_info["pathway_name"]))
    human_ids = set(human_name_map)

    pathway_rows: List[Dict[str, Any]] = []
    gp_rows: List[Dict[str, Any]] = []
    selected_pathway_nodes: Dict[str, str] = {}

    with Path(gmt_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pathway_id = parts[1]
            if pathway_id not in human_ids:
                continue
            members = sorted(selected_gene_set.intersection(parts[2:]))
            if not (min_pathway_genes <= len(members) <= max_pathway_genes):
                continue
            pathway_node = f"{pathway_id}|{human_name_map[pathway_id]}"
            selected_pathway_nodes[pathway_id] = pathway_node
            pathway_rows.append(
                {
                    "pathway": pathway_node,
                    "pathway_id": pathway_id,
                    "pathway_name": human_name_map[pathway_id],
                    "num_member_genes": len(members),
                }
            )
            for gene in members:
                gp_rows.append({"src": gene, "dst": pathway_node})

    relations = pd.read_csv(
        relation_path,
        sep="\t",
        header=None,
        names=["src_id", "dst_id"],
    )
    relations = relations[
        relations["src_id"].isin(selected_pathway_nodes)
        & relations["dst_id"].isin(selected_pathway_nodes)
    ].copy()
    relations["src"] = relations["src_id"].map(selected_pathway_nodes)
    relations["dst"] = relations["dst_id"].map(selected_pathway_nodes)
    pp_rows = relations[["src", "dst"]].drop_duplicates(ignore_index=True)

    pathways_df = pd.DataFrame(pathway_rows).drop_duplicates(subset=["pathway"]).reset_index(drop=True)
    gp_df = pd.DataFrame(gp_rows).drop_duplicates(ignore_index=True)
    return pathways_df, gp_df, pp_rows


def _assign_condition_splits(
    conditions: Sequence[str],
    control_mask: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> List[str]:
    perturb_conditions = sorted({condition for condition in conditions if condition != "ctrl"})
    rng = np.random.default_rng(seed)
    shuffled = perturb_conditions.copy()
    rng.shuffle(shuffled)

    num_conditions = len(shuffled)
    num_test = max(1, int(round(num_conditions * test_fraction)))
    num_val = max(1, int(round(num_conditions * val_fraction)))
    num_test = min(num_test, num_conditions - 2)
    num_val = min(num_val, num_conditions - num_test - 1)

    test_conditions = set(shuffled[:num_test])
    val_conditions = set(shuffled[num_test : num_test + num_val])

    splits = []
    for condition, is_control in zip(conditions, control_mask.tolist()):
        if is_control:
            splits.append("train")
        elif condition in test_conditions:
            splits.append("test")
        elif condition in val_conditions:
            splits.append("val")
        else:
            splits.append("train")
    return splits


def prepare_adamson_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = config["data"]
    preprocess_cfg = config["preprocess"]

    output_root = Path(data_cfg["root"])
    output_root.mkdir(parents=True, exist_ok=True)

    matrix, all_genes, obs = _load_expression_csr(data_cfg["raw_h5ad"])
    go_df = pd.read_csv(data_cfg["go_edges"]).rename(columns={"source": "src", "target": "dst"})
    go_genes = set(go_df["src"]).union(set(go_df["dst"]))

    selected_genes = [gene for gene in all_genes if gene in go_genes]
    if not selected_genes:
        raise RuntimeError("No overlapping genes found between Adamson expression data and GO graph.")

    perturb_genes = sorted({condition.replace("+ctrl", "") for condition in obs["condition"] if condition != "ctrl"})
    selected_gene_set = set(selected_genes)
    missing_perturb_genes = sorted(set(perturb_genes) - selected_gene_set)
    if missing_perturb_genes:
        raise RuntimeError(f"Missing perturbation target genes after graph intersection: {missing_perturb_genes[:10]}")

    expr_gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    selected_col_idx = np.asarray([expr_gene_to_idx[gene] for gene in selected_genes], dtype=np.int64)

    filtered_go = go_df[go_df["src"].isin(selected_gene_set) & go_df["dst"].isin(selected_gene_set)].copy()
    out_degree = filtered_go.groupby("src").size()
    regulator_threshold = max(
        int(out_degree.quantile(float(preprocess_cfg["regulator_outdegree_quantile"]))),
        int(preprocess_cfg["regulator_min_outdegree"]),
    )
    regulator_genes = set(out_degree[out_degree >= regulator_threshold].index).union(set(perturb_genes))

    gg_df = _select_topk_edges(
        filtered_go,
        topk=int(preprocess_cfg["gg_topk"]),
        threshold=float(preprocess_cfg["gg_min_importance"]),
    )
    tg_df = _select_topk_edges(
        filtered_go,
        topk=int(preprocess_cfg["tg_topk"]),
        threshold=float(preprocess_cfg["tg_min_importance"]),
        source_whitelist=regulator_genes,
    )
    if tg_df.empty:
        raise RuntimeError("TF-like directed prior is empty after preprocessing; relax tg_topk or tg_min_importance.")

    pathways_df, gp_df, pp_df = _parse_reactome_pathways(
        gmt_path=data_cfg["reactome_gmt"],
        pathways_path=data_cfg["reactome_pathways"],
        relation_path=data_cfg["reactome_relations"],
        selected_genes=selected_genes,
        min_pathway_genes=int(preprocess_cfg["min_pathway_genes"]),
        max_pathway_genes=int(preprocess_cfg["max_pathway_genes"]),
    )

    pathway_member_counts = Counter(gp_df["src"].tolist())
    go_in_degree = filtered_go.groupby("dst").size().reindex(selected_genes, fill_value=0).to_numpy(dtype=np.float32)
    go_out_degree = filtered_go.groupby("src").size().reindex(selected_genes, fill_value=0).to_numpy(dtype=np.float32)
    go_in_strength = filtered_go.groupby("dst")["importance"].sum().reindex(selected_genes, fill_value=0.0).to_numpy(dtype=np.float32)
    go_out_strength = filtered_go.groupby("src")["importance"].sum().reindex(selected_genes, fill_value=0.0).to_numpy(dtype=np.float32)
    pathway_count = np.asarray([pathway_member_counts.get(gene, 0) for gene in selected_genes], dtype=np.float32)
    is_perturbable = np.asarray([1.0 if gene in perturb_genes else 0.0 for gene in selected_genes], dtype=np.float32)

    continuous_features = np.stack(
        [
            np.log1p(go_in_degree),
            np.log1p(go_out_degree),
            np.log1p(go_in_strength),
            np.log1p(go_out_strength),
            np.log1p(pathway_count),
        ],
        axis=1,
    ).astype(np.float32)
    feature_mean = continuous_features.mean(axis=0, keepdims=True)
    feature_std = continuous_features.std(axis=0, keepdims=True) + 1e-6
    continuous_features = (continuous_features - feature_mean) / feature_std
    gene_metadata = np.concatenate([continuous_features, is_perturbable[:, None]], axis=1).astype(np.float32)

    genes_df = pd.DataFrame(
        {
            "gene": selected_genes,
            "meta_go_in_degree": continuous_features[:, 0],
            "meta_go_out_degree": continuous_features[:, 1],
            "meta_go_in_strength": continuous_features[:, 2],
            "meta_go_out_strength": continuous_features[:, 3],
            "meta_pathway_count": continuous_features[:, 4],
            "meta_is_perturbable": is_perturbable,
        }
    )

    control_mask = obs["control"].astype(bool)
    control_centroid = np.asarray(matrix[control_mask][:, selected_col_idx].mean(axis=0)).ravel().astype(np.float32)

    num_obs = matrix.shape[0]
    num_genes = len(selected_genes)
    chunk_size = int(preprocess_cfg["chunk_size"])
    x_mmap = np.lib.format.open_memmap(output_root / "x.npy", mode="w+", dtype=np.float32, shape=(num_obs, num_genes))
    y_mmap = np.lib.format.open_memmap(output_root / "y.npy", mode="w+", dtype=np.float32, shape=(num_obs, num_genes))

    chunk_iterator = range(0, num_obs, chunk_size)
    for start in tqdm(chunk_iterator, desc="Dense matrix export", unit="chunk"):
        end = min(start + chunk_size, num_obs)
        dense_chunk = matrix[start:end][:, selected_col_idx].toarray().astype(np.float32)
        y_mmap[start:end] = dense_chunk
        x_chunk = np.repeat(control_centroid[None, :], end - start, axis=0)
        local_control_mask = control_mask[start:end]
        x_chunk[local_control_mask] = dense_chunk[local_control_mask]
        x_mmap[start:end] = x_chunk

    sample_ids = [barcode if barcode else f"cell_{idx:06d}" for idx, barcode in enumerate(obs["cell_barcode"])]
    conditions = obs["condition"].tolist()
    perturbed_gene_lists = ["" if condition == "ctrl" else condition.replace("+ctrl", "") for condition in conditions]
    splits = _assign_condition_splits(
        conditions=conditions,
        control_mask=control_mask.astype(np.int64),
        val_fraction=float(preprocess_cfg["val_fraction"]),
        test_fraction=float(preprocess_cfg["test_fraction"]),
        seed=int(preprocess_cfg["seed"]),
    )

    samples_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "split": splits,
            "is_control": control_mask.astype(np.int64),
            "condition": conditions,
            "perturbed_genes": perturbed_gene_lists,
            "cell_type": obs["cell_type"],
            "dose_val": obs["dose_val"],
        }
    )

    genes_df.to_csv(output_root / "genes.csv", index=False)
    pathways_df.to_csv(output_root / "pathways.csv", index=False)
    gg_df.to_csv(output_root / "edges_gg.csv", index=False)
    tg_df.to_csv(output_root / "edges_tg.csv", index=False)
    gp_df.to_csv(output_root / "edges_gp.csv", index=False)
    pp_df.to_csv(output_root / "edges_pp.csv", index=False)
    samples_df.to_csv(output_root / "samples.csv", index=False)
    np.save(output_root / "gene_metadata.npy", gene_metadata)

    summary = {
        "num_samples": int(num_obs),
        "num_control_samples": int(control_mask.sum()),
        "num_perturb_samples": int((~control_mask).sum()),
        "num_selected_genes": int(len(selected_genes)),
        "num_perturb_genes": int(len(perturb_genes)),
        "num_pathways": int(len(pathways_df)),
        "num_edges_gg": int(len(gg_df)),
        "num_edges_tg": int(len(tg_df)),
        "num_edges_gp": int(len(gp_df)),
        "num_edges_pp": int(len(pp_df)),
        "regulator_threshold_outdegree": int(regulator_threshold),
        "missing_perturb_genes": missing_perturb_genes,
    }
    dump_json(summary, output_root / "preprocess_summary.json")
    return summary
