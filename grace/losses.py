from __future__ import annotations

from typing import Dict, List, Set

import torch
import torch.nn.functional as F


def weighted_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    clinical_token_ids: Set[int],
    delta: float,
) -> torch.Tensor:
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_target = target_ids.reshape(-1)

    per_token = F.cross_entropy(flat_logits, flat_target, ignore_index=pad_id, reduction="none")
    weights = torch.ones_like(flat_target, dtype=per_token.dtype)

    if clinical_token_ids:
        clinical_mask = torch.zeros_like(flat_target, dtype=torch.bool)
        for tid in clinical_token_ids:
            clinical_mask |= flat_target == tid
        weights = weights + delta * clinical_mask.float()

    valid = flat_target != pad_id
    loss = (per_token * weights * valid.float()).sum() / valid.float().sum().clamp_min(1.0)
    return loss


def semantic_consistency_loss(pred_hidden: torch.Tensor, tgt_hidden: torch.Tensor) -> torch.Tensor:
    pred_vec = F.normalize(pred_hidden.mean(dim=1), dim=-1)
    tgt_vec = F.normalize(tgt_hidden.mean(dim=1), dim=-1)
    cos = (pred_vec * tgt_vec).sum(dim=-1)
    return 1.0 - cos.mean()


def kg_grounding_loss(
    probs: torch.Tensor,
    keyword_ids: torch.Tensor,
    keyword_neighbors: Dict[int, Set[int]],
    node_to_token_id: Dict[int, int],
    pad_keyword_id: int,
) -> torch.Tensor:
    # probs: [B, T, V]
    bsz = probs.shape[0]
    sample_losses: List[torch.Tensor] = []

    for b in range(bsz):
        nbr_tok_ids = set()
        for kid in keyword_ids[b].tolist():
            if kid == pad_keyword_id:
                continue
            for nbr in keyword_neighbors.get(kid, set()):
                if nbr in node_to_token_id:
                    nbr_tok_ids.add(node_to_token_id[nbr])

        if not nbr_tok_ids:
            continue

        selected = probs[b, :, list(nbr_tok_ids)]
        hit_prob = selected.max(dim=-1).values.mean().clamp_min(1e-8)
        sample_losses.append(-torch.log(hit_prob))

    if not sample_losses:
        return torch.tensor(0.0, device=probs.device)
    return torch.stack(sample_losses).mean()


def grace_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    clinical_token_ids: Set[int],
    delta: float,
    keyword_ids: torch.Tensor,
    keyword_neighbors: Dict[int, Set[int]],
    node_to_token_id: Dict[int, int],
    pad_keyword_id: int,
    alpha: float,
    beta: float,
    gamma: float,
    decoder_embed,
) -> tuple[torch.Tensor, dict]:
    ce = weighted_cross_entropy(logits, target_ids, pad_id, clinical_token_ids, delta)

    pred_tokens = logits.argmax(dim=-1)
    pred_hidden = decoder_embed(pred_tokens)
    tgt_hidden = decoder_embed(target_ids)
    sem = semantic_consistency_loss(pred_hidden, tgt_hidden)

    probs = torch.softmax(logits, dim=-1)
    kg = kg_grounding_loss(
        probs=probs,
        keyword_ids=keyword_ids,
        keyword_neighbors=keyword_neighbors,
        node_to_token_id=node_to_token_id,
        pad_keyword_id=pad_keyword_id,
    )

    total = alpha * ce + beta * sem + gamma * kg
    return total, {"ce": ce.item(), "sem": sem.item(), "kg": kg.item(), "total": total.item()}
