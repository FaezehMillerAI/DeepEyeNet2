from __future__ import annotations

import torch
import torch.nn as nn


class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_weights = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * 0.02)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor, rel_adj: torch.Tensor) -> torch.Tensor:
        out = self.self_loop(h)
        for r in range(rel_adj.shape[0]):
            msg = rel_adj[r] @ h
            out = out + msg @ self.rel_weights[r]
        return self.act(out)


class RetinalDiseaseKGEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        rel_adj: torch.Tensor,
        hidden_dim: int = 256,
        num_layers: int = 2,
        pad_id: int | None = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pad_id = pad_id if pad_id is not None else num_nodes

        self.node_embed = nn.Embedding(num_nodes + 1, hidden_dim)
        self.layers = nn.ModuleList([RGCNLayer(hidden_dim, hidden_dim, rel_adj.shape[0]) for _ in range(num_layers)])
        self.register_buffer("rel_adj", rel_adj)

    def encode_graph(self) -> torch.Tensor:
        h = self.node_embed.weight[: self.num_nodes]
        for layer in self.layers:
            h = layer(h, self.rel_adj)
        return h

    def forward(self, keyword_ids: torch.Tensor) -> torch.Tensor:
        full = self.encode_graph()
        pad_vec = torch.zeros(1, full.shape[1], device=full.device, dtype=full.dtype)
        full_plus_pad = torch.cat([full, pad_vec], dim=0)
        out = full_plus_pad[keyword_ids]
        return out
