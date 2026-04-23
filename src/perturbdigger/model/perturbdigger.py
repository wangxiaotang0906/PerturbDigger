from __future__ import annotations

from typing import Dict, Optional

import math

import torch
from torch import nn

from perturbdigger.graph.hetero_graph import GraphSpecification
from perturbdigger.model.modules import build_mlp, prune_attention_per_target, scatter_sum


class FixedEdgeWeights(nn.Module):
    def __init__(self, weights: Dict[str, torch.Tensor]):
        super().__init__()
        for name, weight in weights.items():
            self.register_buffer(name, weight.detach().clone())

    def forward(self, relation_name: str) -> torch.Tensor:
        return getattr(self, relation_name)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            name: buffer
            for name, buffer in self.named_buffers()
            if name in {"gg", "tg", "gp", "pp"}
        }


class MechanisticBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        tau_tg: float,
        topk_tg: int,
        tau_gp: float,
        topk_gp: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau_tg = tau_tg
        self.topk_tg = topk_tg
        self.tau_gp = tau_gp
        self.topk_gp = topk_gp

        self.mlp_gg = build_mlp(hidden_dim, [hidden_dim], hidden_dim, dropout)
        self.mlp_gp = build_mlp(hidden_dim, [hidden_dim], hidden_dim, dropout)
        self.mlp_pp = build_mlp(hidden_dim, [hidden_dim], hidden_dim, dropout)
        self.mlp_pg = build_mlp(hidden_dim, [hidden_dim], hidden_dim, dropout)

        self.w_gg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_pp = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_pg = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.tg_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tg_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tg_value_tf = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tg_value_gene = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.gp_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gp_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gp_value = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def _apply_edge_block(self, weights: torch.Tensor, block_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if block_mask is None:
            return weights
        return weights.unsqueeze(0) * (1.0 - block_mask)

    def forward(
        self,
        graph: GraphSpecification,
        edge_weights: FixedEdgeWeights | nn.Module,
        gene_states: torch.Tensor,
        pathway_states: torch.Tensor,
        blocked_mediator_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_subgraph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        blocked_mediator_masks = blocked_mediator_masks or {}
        batch_size = gene_states.shape[0]
        num_genes = graph.num_genes
        num_pathways = graph.num_pathways

        gg_edges = graph.relations["gg"].index
        tg_edges = graph.relations["tg"].index
        gp_edges = graph.relations["gp"].index
        pp_edges = graph.relations["pp"].index

        gg_src, gg_dst = gg_edges[0], gg_edges[1]
        gg_w = edge_weights("gg").view(1, -1, 1)
        gg_messages = self.w_gg(gene_states[:, gg_src, :]) * gg_w
        h1 = self.mlp_gg(gene_states) + scatter_sum(gg_messages, gg_dst, num_genes)

        tg_src, tg_dst = tg_edges[0], tg_edges[1]
        q_gene = self.tg_query(h1)
        k_tf = self.tg_key(h1)
        v_tf = self.tg_value_tf(h1)
        v_gene = self.tg_value_gene(h1)
        raw_tg = (q_gene[:, tg_dst, :] * k_tf[:, tg_src, :]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        tg_w = self._apply_edge_block(edge_weights("tg"), blocked_mediator_masks.get("tg"))
        raw_tg = raw_tg * tg_w
        tg_scores = torch.sigmoid(raw_tg)
        tg_selected = prune_attention_per_target(tg_scores, tg_dst, num_genes, self.tau_tg, self.topk_tg)
        tg_messages = tg_selected.unsqueeze(-1) * v_tf[:, tg_src, :]
        h2 = v_gene + scatter_sum(tg_messages, tg_dst, num_genes)

        gp_src, gp_dst = gp_edges[0], gp_edges[1]
        q_pathway = self.gp_query(pathway_states)
        k_gene = self.gp_key(h2)
        v_gene_gp = self.gp_value(h2)
        raw_gp = (q_pathway[:, gp_dst, :] * k_gene[:, gp_src, :]).sum(dim=-1) / math.sqrt(self.hidden_dim)
        gp_w = self._apply_edge_block(edge_weights("gp"), blocked_mediator_masks.get("gp"))
        raw_gp = raw_gp * gp_w
        gp_scores = torch.sigmoid(raw_gp)
        gp_selected = prune_attention_per_target(gp_scores, gp_dst, num_pathways, self.tau_gp, self.topk_gp)
        gp_messages = gp_selected.unsqueeze(-1) * v_gene_gp[:, gp_src, :]
        bar_h3 = self.mlp_gp(pathway_states) + scatter_sum(gp_messages, gp_dst, num_pathways)

        pp_src, pp_dst = pp_edges[0], pp_edges[1]
        pp_w = edge_weights("pp").view(1, -1, 1)
        pp_messages = self.w_pp(bar_h3[:, pp_src, :]) * pp_w
        h3 = self.mlp_pp(bar_h3) + scatter_sum(pp_messages, pp_dst, num_pathways)

        pg_src = gp_dst
        pg_dst = gp_src
        pg_w = self._apply_edge_block(edge_weights("gp"), blocked_mediator_masks.get("gp")).unsqueeze(-1)
        pg_messages = self.w_pg(h3[:, pg_src, :]) * pg_w
        final_gene = self.mlp_pg(h2) + scatter_sum(pg_messages, pg_dst, num_genes)

        result = {"gene_states": final_gene, "pathway_states": h3}
        if return_subgraph:
            result["tg_selected"] = tg_selected
            result["gp_selected"] = gp_selected
            result["h1"] = h1
            result["h2"] = h2
        return result


class PerturbationResponseModel(nn.Module):
    def __init__(self, graph: GraphSpecification, config: Dict[str, float | int], edge_weights: Dict[str, torch.Tensor]):
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        id_dim = int(config["id_dim"])
        context_dim = int(config["context_dim"])
        meta_hidden_dim = int(config["meta_hidden_dim"])
        dropout = float(config.get("dropout", 0.0))

        self.graph = graph
        self.hidden_dim = hidden_dim
        self.edge_weights = FixedEdgeWeights(edge_weights)

        self.gene_id_embedding = nn.Embedding(graph.num_genes, id_dim)
        self.pathway_id_embedding = nn.Embedding(graph.num_pathways, id_dim)

        gene_meta_input_dim = max(graph.gene_meta_dim, 1)
        self.gene_meta_proj = nn.Linear(gene_meta_input_dim, meta_hidden_dim)
        self.context_encoder = build_mlp(graph.num_genes, [context_dim], context_dim, dropout)

        self.gene_seed_proj = nn.Linear(id_dim + meta_hidden_dim + context_dim, hidden_dim)
        self.pathway_seed_proj = nn.Linear(id_dim, hidden_dim)

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
        return self.pathway_seed_proj(pathway_id_emb)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        blocked_mediator_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_subgraph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.shape[0]
        gene_static = self._gene_static().unsqueeze(0).expand(batch_size, -1, -1)
        pathway_static = self._pathway_static().unsqueeze(0).expand(batch_size, -1, -1)
        context_repr = self.context_encoder(x).unsqueeze(1).expand(-1, self.graph.num_genes, -1)

        seed_features = torch.cat([gene_static, context_repr], dim=-1)
        gene_input = self.gene_seed_proj(seed_features) * z.unsqueeze(-1)

        backbone_output = self.backbone(
            graph=self.graph,
            edge_weights=self.edge_weights,
            gene_states=gene_input,
            pathway_states=pathway_static,
            blocked_mediator_masks=blocked_mediator_masks,
            return_subgraph=return_subgraph,
        )
        delta_hat = self.decoder(backbone_output["gene_states"]).squeeze(-1)
        backbone_output["delta_hat"] = delta_hat
        return backbone_output
