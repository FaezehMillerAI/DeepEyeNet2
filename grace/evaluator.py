from __future__ import annotations

from typing import Dict, List, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics


@torch.no_grad()
def run_inference(
    model,
    loader: DataLoader,
    tokenizer,
    device: torch.device,
    max_gen_len: int,
    mc_passes: int,
):
    refs, hyps = [], []
    token_conf_all = []
    token_correct_all = []
    token_entropy_all = []

    raw_rows = []

    for batch in loader:
        images = batch["image"].to(device)
        keyword_ids = batch["keyword_ids"].to(device)
        gt_ids = batch["report_ids"].to(device)

        pred_ids, token_entropy, mean_probs = model.mc_dropout_generate(
            images=images,
            keyword_ids=keyword_ids,
            bos_id=tokenizer.vocab.bos_id,
            eos_id=tokenizer.vocab.eos_id,
            max_len=max_gen_len,
            passes=mc_passes,
        )

        conf = mean_probs.max(dim=-1).values.cpu().numpy()
        pred_tok = pred_ids.cpu().numpy()
        gt_tok = gt_ids[:, : pred_ids.shape[1]].cpu().numpy()
        corr = (pred_tok == gt_tok).astype(np.float32)

        for i in range(pred_ids.shape[0]):
            hyp = tokenizer.decode(pred_tok[i].tolist())
            ref = tokenizer.decode(gt_tok[i].tolist())
            hyps.append(hyp)
            refs.append(ref)

            raw_rows.append(
                {
                    "image_path": batch["image_path"][i],
                    "reference": ref,
                    "hypothesis": hyp,
                    "keywords": batch["keyword_text"][i],
                    "mean_entropy": float(token_entropy[i].mean().cpu().item()),
                }
            )

        token_conf_all.append(conf.reshape(-1))
        token_correct_all.append(corr.reshape(-1))
        token_entropy_all.append(token_entropy.cpu().numpy().reshape(-1))

    return {
        "refs": refs,
        "hyps": hyps,
        "token_conf": np.concatenate(token_conf_all),
        "token_correct": np.concatenate(token_correct_all),
        "token_entropy": np.concatenate(token_entropy_all),
        "rows": raw_rows,
    }


def evaluate_predictions(
    refs: Sequence[str],
    hyps: Sequence[str],
    token_conf: np.ndarray,
    token_correct: np.ndarray,
    concept_lexicon: Sequence[str],
    sample_allowed_concepts: Sequence[Set[str]],
    bins: int = 10,
) -> Dict[str, float]:
    return compute_all_metrics(
        refs=refs,
        hyps=hyps,
        concept_lexicon=concept_lexicon,
        sample_allowed_concepts=sample_allowed_concepts,
        token_conf=token_conf,
        token_correct=token_correct,
        bins=bins,
    )
