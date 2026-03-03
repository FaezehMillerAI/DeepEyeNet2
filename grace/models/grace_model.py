from __future__ import annotations

import torch
import torch.nn as nn

from .bcma import BCMA
from .decoder import ClinicalDecoder
from .msve_pafp import MSVEPAFP
from .rdkge import RetinalDiseaseKGEncoder


class GRACEModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        rel_adj: torch.Tensor,
        num_graph_nodes: int,
        graph_pad_id: int,
        d_model: int = 256,
        num_heads: int = 8,
        decoder_layers: int = 4,
        dropout: float = 0.2,
        graph_layers: int = 2,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.msve = MSVEPAFP(out_channels=160, d_model=d_model, pretrained=pretrained_backbone)
        self.kge = RetinalDiseaseKGEncoder(
            num_nodes=num_graph_nodes,
            rel_adj=rel_adj,
            hidden_dim=d_model,
            num_layers=graph_layers,
            pad_id=graph_pad_id,
        )
        self.bcma = BCMA(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.decoder = ClinicalDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            layers=decoder_layers,
            dropout=dropout,
        )

    @staticmethod
    def causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, images: torch.Tensor, keyword_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        visual = self.msve(images)
        kw = self.kge(keyword_ids)
        fused = self.bcma(visual, kw)
        cmask = self.causal_mask(tgt_ids.shape[1], tgt_ids.device)
        logits = self.decoder(tgt_ids=tgt_ids, memory=fused, causal_mask=cmask)
        return logits

    def generate(
        self,
        images: torch.Tensor,
        keyword_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ):
        bsz = images.shape[0]
        visual = self.msve(images)
        kw = self.kge(keyword_ids)
        fused = self.bcma(visual, kw)

        ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=images.device)
        for _ in range(max_len - 1):
            cmask = self.causal_mask(ys.shape[1], ys.device)
            logits = self.decoder(ys, fused, cmask)
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_id).all():
                break
        return ys

    def mc_dropout_generate(
        self,
        images: torch.Tensor,
        keyword_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        passes: int,
    ):
        self.train()
        preds = []
        probs = []
        with torch.no_grad():
            for _ in range(passes):
                ys = self.generate(images, keyword_ids, bos_id, eos_id, max_len)
                preds.append(ys)

                visual = self.msve(images)
                kw = self.kge(keyword_ids)
                fused = self.bcma(visual, kw)
                cmask = self.causal_mask(ys.shape[1], ys.device)
                logits = self.decoder(ys, fused, cmask)
                probs.append(torch.softmax(logits, dim=-1))

        pred_stack = torch.stack(preds, dim=0)
        prob_stack = torch.stack(probs, dim=0)
        mean_probs = prob_stack.mean(dim=0)
        token_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum(dim=-1)

        mode_pred = pred_stack.mode(dim=0).values
        self.eval()
        return mode_pred, token_entropy, mean_probs
