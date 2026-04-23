from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch

from perturbdigger.data.dataset import DatasetBundle
from perturbdigger.graph.hetero_graph import GraphSpecification


def _top_edges(
    values: torch.Tensor,
    edge_index: torch.Tensor,
    source_names: List[str],
    target_names: List[str],
    topn: int,
) -> List[Dict[str, float | str]]:
    if values.numel() == 0:
        return []
    topn = min(topn, values.shape[0])
    top_values, top_idx = torch.topk(values, k=topn)
    edges = []
    for score, idx in zip(top_values.tolist(), top_idx.tolist()):
        if score <= 0:
            continue
        src = source_names[int(edge_index[0, idx])]
        dst = target_names[int(edge_index[1, idx])]
        edges.append({"src": src, "dst": dst, "score": float(score)})
    return edges


def build_sample_explanations(
    bundle: DatasetBundle,
    graph: GraphSpecification,
    sample_indices: torch.Tensor,
    tg_selected: torch.Tensor,
    gp_selected: torch.Tensor,
    topn: int,
) -> List[Dict[str, object]]:
    tg_edges = graph.relations["tg"].index.detach().cpu()
    gp_edges = graph.relations["gp"].index.detach().cpu()
    sample_records = []
    for row_idx, sample_idx in enumerate(sample_indices.tolist()):
        sample_records.append(
            {
                "sample_id": bundle.sample_ids[sample_idx],
                "condition": bundle.conditions[sample_idx],
                "perturbed_genes": bundle.perturbed_gene_lists[sample_idx],
                "tf_gene_edges": _top_edges(
                    tg_selected[row_idx].detach().cpu(),
                    tg_edges,
                    bundle.gene_names,
                    bundle.gene_names,
                    topn=topn,
                ),
                "gene_pathway_edges": _top_edges(
                    gp_selected[row_idx].detach().cpu(),
                    gp_edges,
                    bundle.gene_names,
                    bundle.pathway_names,
                    topn=topn,
                ),
            }
        )
    return sample_records


def aggregate_explanations(sample_records: List[Dict[str, object]], topn: int) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, int] = defaultdict(int)

    for record in sample_records:
        condition = str(record["condition"])
        counts[condition] += 1
        for edge in record["tf_gene_edges"]:
            edge_key = f"tg::{edge['src']}->{edge['dst']}"
            grouped[condition][edge_key] += float(edge["score"])
        for edge in record["gene_pathway_edges"]:
            edge_key = f"gp::{edge['src']}->{edge['dst']}"
            grouped[condition][edge_key] += float(edge["score"])

    aggregated = []
    for condition, edge_scores in grouped.items():
        normalized = []
        for edge_key, total_score in edge_scores.items():
            relation, edge = edge_key.split("::", maxsplit=1)
            src, dst = edge.split("->", maxsplit=1)
            normalized.append(
                {
                    "relation": relation,
                    "src": src,
                    "dst": dst,
                    "score": total_score / max(counts[condition], 1),
                }
            )
        normalized.sort(key=lambda item: item["score"], reverse=True)
        aggregated.append(
            {
                "condition": condition,
                "num_samples": counts[condition],
                "edges": normalized[:topn],
            }
        )
    aggregated.sort(key=lambda item: item["condition"])
    return aggregated
