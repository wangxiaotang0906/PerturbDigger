from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from perturbdigger.data.dataset import DatasetBundle


@dataclass
class RelationEdges:
    name: str
    source_type: str
    target_type: str
    index: torch.LongTensor

    @property
    def num_edges(self) -> int:
        return int(self.index.shape[1])


@dataclass
class GraphSpecification:
    gene_names: List[str]
    pathway_names: List[str]
    gene_metadata: torch.FloatTensor
    relations: Dict[str, RelationEdges]

    @property
    def num_genes(self) -> int:
        return len(self.gene_names)

    @property
    def num_pathways(self) -> int:
        return len(self.pathway_names)

    @property
    def gene_meta_dim(self) -> int:
        return int(self.gene_metadata.shape[1])


def build_graph_specification(bundle: DatasetBundle, device: torch.device | str) -> GraphSpecification:
    relations = {
        name: RelationEdges(
            name=name,
            source_type="gene" if name in {"gg", "tg", "gp"} else "pathway",
            target_type="gene" if name in {"gg", "tg"} else "pathway",
            index=torch.as_tensor(edge_index, dtype=torch.long, device=device),
        )
        for name, edge_index in bundle.edge_indices.items()
    }
    return GraphSpecification(
        gene_names=bundle.gene_names,
        pathway_names=bundle.pathway_names,
        gene_metadata=torch.as_tensor(bundle.gene_metadata, dtype=torch.float32, device=device),
        relations=relations,
    )


class LearnableEdgeWeights(nn.Module):
    def __init__(self, graph: GraphSpecification, init_logit: float = 4.0):
        super().__init__()
        self.logits = nn.ParameterDict()
        for name, relation in graph.relations.items():
            init = torch.full((relation.num_edges,), init_logit, dtype=torch.float32)
            self.logits[name] = nn.Parameter(init)

    def forward(self, relation_name: str) -> torch.Tensor:
        return torch.sigmoid(self.logits[relation_name])

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {name: torch.sigmoid(param) for name, param in self.logits.items()}

    def detached(self) -> Dict[str, torch.Tensor]:
        return {name: weight.detach().clone() for name, weight in self.as_dict().items()}

    def regularization_loss(self, q_tg: torch.Tensor | None = None) -> torch.Tensor:
        losses = []
        if "tg" in self.logits and q_tg is not None:
            losses.append(((self("tg") - q_tg) ** 2).mean())
        if "gp" in self.logits:
            losses.append(((1.0 - self("gp")) ** 2).mean())
        if "pp" in self.logits:
            losses.append(((1.0 - self("pp")) ** 2).mean())
        if "gg" in self.logits:
            losses.append(self("gg").abs().mean())
        return torch.stack(losses).sum() if losses else torch.tensor(0.0)


def relation_table_to_numpy(bundle: DatasetBundle, name: str) -> np.ndarray:
    return bundle.edge_indices[name]
