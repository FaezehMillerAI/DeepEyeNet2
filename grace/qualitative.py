from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pickle
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from grace.config import GRACEConfig
from grace.data.deepeyenet import DeepEyeNetDataset, load_split_csv
from grace.data.graph import build_graph_from_dataframe
from grace.data.tokenizer import ClinicalTokenizer, Vocab
from grace.evaluate import _device_from_cfg, _reorder_rel_mats
from grace.models.grace_model import GRACEModel
from grace.utils.common import save_json, set_seed


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        tqdm.write("Detected legacy checkpoint format. Reloading with weights_only=False for compatibility.")
        return torch.load(path, map_location="cpu", weights_only=False)


def _build_tokenizer_from_ckpt_or_data(ckpt, train_df: pd.DataFrame) -> ClinicalTokenizer:
    vocab_itos = ckpt.get("tokenizer_vocab")
    if vocab_itos is None:
        tok = ClinicalTokenizer(max_vocab_size=15000, min_freq=1)
        tok.fit(train_df["report_text"].fillna("").tolist())
        return tok

    stoi = {tok: i for i, tok in enumerate(vocab_itos)}
    tok = ClinicalTokenizer()
    tok.vocab = Vocab(stoi=stoi, itos=vocab_itos)
    return tok


def _decode_with_uncertainty(tokenizer: ClinicalTokenizer, token_ids: List[int], entropy: List[float], conf: List[float]):
    special = {"<pad>", "<bos>", "<unk>"}
    items = []
    for tid, ent, c in zip(token_ids, entropy, conf):
        tok = tokenizer.vocab.itos[tid]
        if tok == "<eos>":
            break
        if tok in special:
            continue
        items.append({"token": tok, "entropy": float(ent), "confidence": float(c)})
    return items


def _highlight_uncertain(items: List[dict], entropy_threshold: float) -> str:
    out = []
    for x in items:
        tok = x["token"]
        if x["entropy"] >= entropy_threshold:
            out.append(f"[{tok}]")
        else:
            out.append(tok)
    txt = " ".join(out)
    txt = txt.replace(" ,", ",").replace(" .", ".").replace(" ;", ";").replace(" :", ":")
    return txt.strip()


def _case_panel(image_path: Path, keywords: str, reference: str, prediction: str, out_path: Path):
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Fundus Image")

    body = (
        f"Keywords:\n{textwrap.fill(keywords, width=60)}\n\n"
        f"Reference:\n{textwrap.fill(reference, width=60)}\n\n"
        f"Generated (uncertain tokens in []):\n{textwrap.fill(prediction, width=60)}"
    )
    axes[1].axis("off")
    axes[1].text(0.01, 0.99, body, ha="left", va="top", fontsize=10)
    axes[1].set_title("Qualitative Comparison")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_qualitative(cfg: GRACEConfig, checkpoint_path: str | Path, n_samples: int = 10, seed: int = 42) -> Dict:
    set_seed(seed)
    out_dir = Path(cfg.output_dir)
    q_dir = out_dir / "qualitative"
    q_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".py":
        raise ValueError(f"Checkpoint must be .pt, got: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_df = load_split_csv(cfg.data.dataset_root, cfg.data.train_csv)
    test_df = load_split_csv(cfg.data.dataset_root, cfg.data.test_csv)
    graph = build_graph_from_dataframe(train_df, relation_types=cfg.model.relation_types)

    ckpt = _load_checkpoint(checkpoint_path)
    tokenizer = _build_tokenizer_from_ckpt_or_data(ckpt, train_df)

    ckpt_node2id = ckpt.get("graph_node2id", graph.node2id)
    graph_pad_id = len(ckpt_node2id)

    full_ds = DeepEyeNetDataset(
        test_df,
        dataset_root=cfg.data.dataset_root,
        tokenizer=tokenizer,
        node2id=ckpt_node2id,
        max_report_len=cfg.data.max_report_len,
        max_keywords=cfg.data.max_keywords,
        image_size=cfg.data.image_size,
        graph_pad_id=graph_pad_id,
        augment=False,
    )

    n = min(n_samples, len(full_ds))
    indices = torch.randperm(len(full_ds)).tolist()[:n]
    subset = Subset(full_ds, indices)
    loader = DataLoader(subset, batch_size=min(cfg.train.batch_size, n), shuffle=False, num_workers=cfg.data.num_workers)

    rel_adj = _reorder_rel_mats(graph, cfg.model.relation_types, ckpt_node2id)
    device = _device_from_cfg(cfg.train.device)

    model = GRACEModel(
        vocab_size=tokenizer.vocab_size,
        rel_adj=rel_adj,
        num_graph_nodes=len(ckpt_node2id),
        graph_pad_id=graph_pad_id,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        decoder_layers=cfg.model.decoder_layers,
        dropout=cfg.model.dropout,
        graph_layers=cfg.model.graph_layers,
        pretrained_backbone=cfg.model.pretrained_backbone,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    records = []
    all_token_items = []

    pbar = tqdm(loader, desc=f"Qualitative Inference ({n} samples)")
    offset = 0
    for batch in pbar:
        images = batch["image"].to(device)
        kw_ids = batch["keyword_ids"].to(device)
        gt_ids = batch["report_ids"].to(device)

        pred_ids, token_entropy, mean_probs = model.mc_dropout_generate(
            images=images,
            keyword_ids=kw_ids,
            bos_id=tokenizer.vocab.bos_id,
            eos_id=tokenizer.vocab.eos_id,
            max_len=cfg.evaluation.max_gen_len,
            passes=cfg.evaluation.mc_dropout_passes,
        )

        conf = mean_probs.max(dim=-1).values.cpu()
        pred_np = pred_ids.cpu().numpy()
        gt_np = gt_ids[:, : pred_ids.shape[1]].cpu().numpy()
        ent_np = token_entropy.cpu().numpy()

        for i in range(pred_ids.shape[0]):
            local_idx = offset + i
            ds_idx = indices[local_idx]
            row = test_df.iloc[ds_idx]

            token_items = _decode_with_uncertainty(
                tokenizer,
                pred_np[i].tolist(),
                ent_np[i].tolist(),
                conf[i].tolist(),
            )
            all_token_items.extend(token_items)

            reference = tokenizer.decode(gt_np[i].tolist())
            generated_plain = tokenizer.decode(pred_np[i].tolist())

            records.append(
                {
                    "sample_rank": local_idx + 1,
                    "dataset_index": int(ds_idx),
                    "image_path": str(row["image_path"]),
                    "keywords": str(batch["keyword_text"][i]),
                    "reference": reference,
                    "generated_plain": generated_plain,
                    "mean_entropy": float(token_entropy[i].mean().item()),
                    "token_items": token_items,
                }
            )
        offset += pred_ids.shape[0]

    entropy_threshold = 0.0
    if all_token_items:
        vals = [x["entropy"] for x in all_token_items]
        vals_t = torch.tensor(vals, dtype=torch.float32)
        entropy_threshold = float(torch.quantile(vals_t, 0.80).item())

    # second pass to add highlighted text with a stable threshold
    enriched = []
    for rec in records:
        rec["generated_highlighted"] = _highlight_uncertain(rec["token_items"], entropy_threshold)
        rec["uncertainty_note"] = f"Uncertain tokens are marked with [] at entropy >= {entropy_threshold:.3f}."
        rec["mean_confidence"] = (
            float(sum(x["confidence"] for x in rec["token_items"]) / max(1, len(rec["token_items"])))
            if rec["token_items"]
            else 0.0
        )
        enriched.append(rec)

    # Generate case panels and markdown report
    md_lines = [
        "# Qualitative Results (10 Test Samples)",
        "",
        f"Checkpoint: `{checkpoint_path}`",
        f"Entropy threshold for uncertainty marking (top 20% tokens): `{entropy_threshold:.3f}`",
        "",
    ]

    for rec in enriched:
        img_abs = Path(cfg.data.dataset_root) / rec["image_path"]
        panel_path = q_dir / f"case_{rec['sample_rank']:02d}.png"

        highlighted = rec["generated_highlighted"]

        _case_panel(
            image_path=img_abs,
            keywords=rec["keywords"],
            reference=rec["reference"],
            prediction=highlighted,
            out_path=panel_path,
        )

        rec["panel_path"] = str(panel_path)

        md_lines.extend(
            [
                f"## Case {rec['sample_rank']}",
                "",
                f"- Image: `{rec['image_path']}`",
                f"- Keywords: {rec['keywords']}",
                f"- Mean uncertainty: {rec['mean_entropy']:.4f}",
                f"- Mean confidence: {rec['mean_confidence']:.4f}",
                f"- Reference: {rec['reference']}",
                f"- Generated: {highlighted}",
                f"![case_{rec['sample_rank']:02d}]({panel_path})",
                "",
            ]
        )

    export_rows = []
    for rec in enriched:
        row = dict(rec)
        row["token_items"] = str(row["token_items"])
        export_rows.append(row)
    pd.DataFrame(export_rows).to_csv(q_dir / "qualitative_samples.csv", index=False)
    save_json(export_rows, q_dir / "qualitative_samples.json")
    (q_dir / "qualitative_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "output_dir": str(q_dir),
        "num_samples": len(enriched),
        "report": str(q_dir / "qualitative_report.md"),
        "table": str(q_dir / "qualitative_samples.csv"),
    }
