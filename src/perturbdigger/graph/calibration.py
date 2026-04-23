from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import Lasso
from torch import nn
from tqdm.auto import tqdm

from perturbdigger.graph.hetero_graph import GraphSpecification, LearnableEdgeWeights
from perturbdigger.model.modules import build_mlp
from perturbdigger.model.perturbdigger import MechanisticBackbone


def compute_tf_relevance_prior(
    control_x: np.ndarray,
    tg_edge_index: np.ndarray,
    num_genes: int,
    alpha: float = 0.01,
    show_progress: bool = False,
) -> np.ndarray:
    src_tf = tg_edge_index[0]
    dst_gene = tg_edge_index[1]
    relevance = np.full(src_tf.shape[0], 0.5, dtype=np.float32)

    iterator = range(num_genes)
    if show_progress:
        iterator = tqdm(iterator, desc="TF prior", leave=False, unit="gene")

    for gene_idx in iterator:
        edge_mask = dst_gene == gene_idx
        if not np.any(edge_mask):
            continue
        tf_neighbors = src_tf[edge_mask]
        x_target = control_x[:, gene_idx]
        x_reg = control_x[:, tf_neighbors]
        if x_reg.ndim == 1:
            x_reg = x_reg[:, None]
        if x_reg.shape[1] == 0:
            continue
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
        model.fit(x_reg, x_target)
        beta = np.abs(model.coef_)
        relevance[edge_mask] = 1.0 / (1.0 + np.exp(-beta))

    return relevance.astype(np.float32)


class GraphCalibrationModel(nn.Module):
    def __init__(
        self,
        graph: GraphSpecification,
        edge_weights: LearnableEdgeWeights,
        q_tg_prior: torch.Tensor,
        config: Dict[str, float | int],
    ):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        id_dim = int(config["id_dim"])
        meta_hidden_dim = int(config["meta_hidden_dim"])
        dropout = float(config.get("dropout", 0.0))

        self.graph = graph
        self.edge_weights = edge_weights
        self.register_buffer("q_tg_prior", q_tg_prior)

        self.gene_id_embedding = nn.Embedding(graph.num_genes, id_dim)
        self.pathway_id_embedding = nn.Embedding(graph.num_pathways, id_dim)
        gene_meta_input_dim = max(graph.gene_meta_dim, 1)
        self.gene_meta_proj = nn.Linear(gene_meta_input_dim, meta_hidden_dim)
        self.gene_input_proj = nn.Linear(1 + id_dim + meta_hidden_dim, hidden_dim)
        self.pathway_input_proj = nn.Linear(id_dim, hidden_dim)

        self.backbone = MechanisticBackbone(
            hidden_dim=hidden_dim,
            tau_tg=float(config["tau_tg"]),
            topk_tg=int(config["topk_tg"]),
            tau_gp=float(config["tau_gp"]),
            topk_gp=int(config["topk_gp"]),
            dropout=dropout,
        )
        self.decoder = build_mlp(hidden_dim, [hidden_dim], 1, dropout)

    def _gene_static(self) -> torch.Tensor:
        gene_ids = torch.arange(self.graph.num_genes, device=self.graph.gene_metadata.device)
        gene_id_emb = self.gene_id_embedding(gene_ids)
        if self.graph.gene_meta_dim == 0:
            gene_meta = self.graph.gene_metadata.new_zeros((self.graph.num_genes, 1))
        else:
            gene_meta = self.graph.gene_metadata
        meta_repr = self.gene_meta_proj(gene_meta)
        return torch.cat([gene_id_emb, meta_repr], dim=-1)

    def _pathway_static(self) -> torch.Tensor:
        pathway_ids = torch.arange(self.graph.num_pathways, device=self.graph.gene_metadata.device)
        pathway_id_emb = self.pathway_id_embedding(pathway_ids)
        return self.pathway_input_proj(pathway_id_emb)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        if mask is None:
            mask = torch.zeros_like(x, dtype=torch.bool)

        gene_static = self._gene_static().unsqueeze(0).expand(batch_size, -1, -1)
        masked_x = x.masked_fill(mask, 0.0).unsqueeze(-1)
        gene_input = self.gene_input_proj(torch.cat([masked_x, gene_static], dim=-1))
        pathway_input = self._pathway_static().unsqueeze(0).expand(batch_size, -1, -1)

        backbone_output = self.backbone(
            graph=self.graph,
            edge_weights=self.edge_weights,
            gene_states=gene_input,
            pathway_states=pathway_input,
            return_subgraph=False,
        )
        x_hat = self.decoder(backbone_output["gene_states"]).squeeze(-1)
        return {"x_hat": x_hat}

    def graph_regularization(self) -> torch.Tensor:
        return self.edge_weights.regularization_loss(q_tg=self.q_tg_prior)
