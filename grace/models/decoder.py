from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ClinicalDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor, causal_mask: torch.Tensor | None = None):
        x = self.embed(tgt_ids)
        x = self.pos(x)
        x = self.dropout(x)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=causal_mask)
        return self.out(x)
