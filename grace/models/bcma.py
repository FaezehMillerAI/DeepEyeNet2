from __future__ import annotations

import torch
import torch.nn as nn


class BCMA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.v_to_g = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.g_to_v = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor, keyword_tokens: torch.Tensor) -> torch.Tensor:
        z_vg, _ = self.v_to_g(query=visual_tokens, key=keyword_tokens, value=keyword_tokens)
        z_gv, _ = self.g_to_v(query=keyword_tokens, key=visual_tokens, value=visual_tokens)

        g_summary = z_gv.mean(dim=1, keepdim=True).expand(-1, visual_tokens.shape[1], -1)
        lam = self.gate(torch.cat([z_vg, g_summary], dim=-1))
        fused = lam * z_vg + (1.0 - lam) * g_summary
        return self.norm(fused + visual_tokens)
