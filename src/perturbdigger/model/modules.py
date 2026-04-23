from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def scatter_sum(messages: torch.Tensor, dst_index: torch.Tensor, output_size: int) -> torch.Tensor:
    batch_size, _, hidden_dim = messages.shape
    output = messages.new_zeros((batch_size, output_size, hidden_dim))
    expanded_index = dst_index.view(1, -1, 1).expand(batch_size, -1, hidden_dim)
    output.scatter_add_(1, expanded_index, messages)
    return output


def prune_attention_per_target(
    scores: torch.Tensor,
    dst_index: torch.Tensor,
    num_targets: int,
    threshold: float,
    topk: int,
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores

    pruned = torch.zeros_like(scores)
    if topk <= 0:
        return pruned

    for target_idx in range(num_targets):
        mask = dst_index == target_idx
        if not bool(mask.any()):
            continue
        local_scores = torch.where(mask.unsqueeze(0), scores, torch.zeros_like(scores))
        local_scores = torch.where(local_scores >= threshold, local_scores, torch.zeros_like(local_scores))
        available = int(mask.sum().item())
        k = min(topk, available)
        if k == 0:
            continue
        top_vals, top_idx = torch.topk(local_scores, k=k, dim=1)
        top_vals = torch.where(top_vals > 0, top_vals, torch.zeros_like(top_vals))
        local_pruned = torch.zeros_like(scores)
        local_pruned.scatter_(1, top_idx, top_vals)
        pruned = torch.maximum(pruned, local_pruned)
    return pruned
