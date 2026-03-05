from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from grace.config import GRACEConfig
from grace.data.deepeyenet import DeepEyeNetDataset, load_split_csv
from grace.data.graph import build_graph_from_dataframe, parse_keywords_field
from grace.data.tokenizer import ClinicalTokenizer
from grace.evaluator import evaluate_predictions, run_inference
from grace.losses import grace_loss
from grace.models.grace_model import GRACEModel
from grace.utils.common import save_json, set_seed
from grace.utils.viz import (
    save_calibration_curve,
    save_keyword_graph,
    save_metric_bar,
    save_radar,
    save_training_curves,
    save_uncertainty_hist,
)


def _device_from_cfg(cfg_device: str) -> torch.device:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_allowed_concepts(df: pd.DataFrame, graph_artifacts) -> List[Set[str]]:
    out = []
    for kw in df["Keywords"].tolist():
        ids = [graph_artifacts.node2id[k] for k in parse_keywords_field(kw) if k in graph_artifacts.node2id]
        allowed = set()
        for i in ids:
            allowed.add(graph_artifacts.id2node[i])
            for n in graph_artifacts.neighbors.get(i, set()):
                allowed.add(graph_artifacts.id2node[n])
        out.append(allowed)
    return out


def train_grace(cfg: GRACEConfig, resume_path: str | None = None, auto_resume: bool = True) -> Dict:
    set_seed(cfg.train.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split_csv(cfg.data.dataset_root, cfg.data.train_csv)
    valid_df = load_split_csv(cfg.data.dataset_root, cfg.data.valid_csv)
    test_df = load_split_csv(cfg.data.dataset_root, cfg.data.test_csv)

    tokenizer = ClinicalTokenizer(max_vocab_size=15000, min_freq=1)
    tokenizer.fit(train_df["report_text"].fillna("").tolist())

    graph = build_graph_from_dataframe(train_df, relation_types=cfg.model.relation_types)
    graph_pad_id = len(graph.node2id)

    concept_lexicon = sorted(graph.node2id.keys())
    clinical_token_ids = {
        tokenizer.vocab.stoi[t] for t in concept_lexicon if t in tokenizer.vocab.stoi
    }
    node_to_token_id = {
        nid: tokenizer.vocab.stoi[t] for t, nid in graph.node2id.items() if t in tokenizer.vocab.stoi
    }

    train_ds = DeepEyeNetDataset(
        train_df,
        dataset_root=cfg.data.dataset_root,
        tokenizer=tokenizer,
        node2id=graph.node2id,
        max_report_len=cfg.data.max_report_len,
        max_keywords=cfg.data.max_keywords,
        image_size=cfg.data.image_size,
        graph_pad_id=graph_pad_id,
        augment=True,
    )
    valid_ds = DeepEyeNetDataset(
        valid_df,
        dataset_root=cfg.data.dataset_root,
        tokenizer=tokenizer,
        node2id=graph.node2id,
        max_report_len=cfg.data.max_report_len,
        max_keywords=cfg.data.max_keywords,
        image_size=cfg.data.image_size,
        graph_pad_id=graph_pad_id,
        augment=False,
    )
    test_ds = DeepEyeNetDataset(
        test_df,
        dataset_root=cfg.data.dataset_root,
        tokenizer=tokenizer,
        node2id=graph.node2id,
        max_report_len=cfg.data.max_report_len,
        max_keywords=cfg.data.max_keywords,
        image_size=cfg.data.image_size,
        graph_pad_id=graph_pad_id,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    rel_adj = torch.stack(
        [torch.tensor(graph.relation_matrices[r], dtype=torch.float32) for r in cfg.model.relation_types],
        dim=0,
    )

    device = _device_from_cfg(cfg.train.device)
    model = GRACEModel(
        vocab_size=tokenizer.vocab_size,
        rel_adj=rel_adj,
        num_graph_nodes=len(graph.node2id),
        graph_pad_id=graph_pad_id,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        decoder_layers=cfg.model.decoder_layers,
        dropout=cfg.model.dropout,
        graph_layers=cfg.model.graph_layers,
        pretrained_backbone=cfg.model.pretrained_backbone,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    ckpt_best_path = out_dir / "best_grace.pt"
    ckpt_last_path = out_dir / "last_checkpoint.pt"

    best_bleu4 = -1.0
    patience = 0
    history = []
    start_epoch = 1

    valid_allowed = _build_allowed_concepts(valid_df, graph)
    test_allowed = _build_allowed_concepts(test_df, graph)

    chosen_resume = None
    if resume_path:
        chosen_resume = Path(resume_path)
    elif auto_resume and ckpt_last_path.exists():
        chosen_resume = ckpt_last_path

    if chosen_resume is not None and chosen_resume.exists():
        tqdm.write(f"Resuming from checkpoint: {chosen_resume}")
        resume_ckpt = torch.load(chosen_resume, map_location=device, weights_only=True)
        model.load_state_dict(resume_ckpt["model_state"])
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        best_bleu4 = float(resume_ckpt.get("best_bleu4", best_bleu4))
        patience = int(resume_ckpt.get("patience", patience))
        history = list(resume_ckpt.get("history", history))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        if start_epoch > cfg.train.epochs:
            tqdm.write(
                f"Checkpoint epoch ({start_epoch - 1}) already reached/exceeded configured epochs ({cfg.train.epochs}). "
                "Skipping training and proceeding to final evaluation."
            )

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        running = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            keyword_ids = batch["keyword_ids"].to(device)
            report_ids = batch["report_ids"].to(device)

            inp = report_ids[:, :-1]
            tgt = report_ids[:, 1:]

            logits = model(images=images, keyword_ids=keyword_ids, tgt_ids=inp)
            loss, parts = grace_loss(
                logits=logits,
                target_ids=tgt,
                pad_id=tokenizer.vocab.pad_id,
                clinical_token_ids=clinical_token_ids,
                delta=cfg.train.clinical_weight_delta,
                keyword_ids=keyword_ids,
                keyword_neighbors=graph.neighbors,
                node_to_token_id=node_to_token_id,
                pad_keyword_id=graph_pad_id,
                alpha=cfg.train.alpha_ce,
                beta=cfg.train.beta_sem,
                gamma=cfg.train.gamma_kg,
                decoder_embed=model.decoder.embed,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            running.append(parts["total"])
            pbar.set_postfix(loss=f"{parts['total']:.4f}")

        avg_train = float(sum(running) / max(1, len(running)))
        tqdm.write(f"[Epoch {epoch}] Training complete. Running validation inference...")

        val_out = run_inference(
            model=model,
            loader=valid_loader,
            tokenizer=tokenizer,
            device=device,
            max_gen_len=cfg.evaluation.max_gen_len,
            mc_passes=cfg.evaluation.mc_dropout_passes,
            stage_name=f"Validation {epoch}/{cfg.train.epochs}",
        )
        tqdm.write(f"[Epoch {epoch}] Validation inference complete. Computing validation metrics...")
        val_metrics = evaluate_predictions(
            refs=val_out["refs"],
            hyps=val_out["hyps"],
            token_conf=val_out["token_conf"],
            token_correct=val_out["token_correct"],
            concept_lexicon=concept_lexicon,
            sample_allowed_concepts=valid_allowed,
            bins=cfg.evaluation.calibration_bins,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train,
                "val_bleu4": val_metrics["bleu4"],
                "val_c3s": val_metrics["c3s"],
                "val_chr": val_metrics["chr"],
                "val_ucs": val_metrics["ucs"],
            }
        )

        if val_metrics["bleu4"] > best_bleu4:
            best_bleu4 = val_metrics["bleu4"]
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "tokenizer_vocab": tokenizer.vocab.itos,
                    "graph_node2id": graph.node2id,
                    # Store plain dict to keep checkpoint compatible with torch.load(weights_only=True).
                    "config_dict": asdict(cfg),
                },
                ckpt_best_path,
            )
        else:
            patience += 1

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_bleu4": best_bleu4,
                "patience": patience,
                "history": history,
                "tokenizer_vocab": tokenizer.vocab.itos,
                "graph_node2id": graph.node2id,
                "config_dict": asdict(cfg),
            },
            ckpt_last_path,
        )

        if patience >= cfg.train.early_stop_patience:
            tqdm.write("Early stopping triggered.")
            break

    if ckpt_best_path.exists():
        ckpt = torch.load(ckpt_best_path, map_location=device, weights_only=True)
    else:
        tqdm.write("Best checkpoint not found; falling back to last checkpoint for final evaluation.")
        ckpt = torch.load(ckpt_last_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    tqdm.write("Best checkpoint loaded. Running final test evaluation...")

    test_out = run_inference(
        model=model,
        loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_gen_len=cfg.evaluation.max_gen_len,
        mc_passes=cfg.evaluation.mc_dropout_passes,
        stage_name="Test Evaluation",
    )

    tqdm.write("Test inference complete. Computing final metrics...")
    test_metrics = evaluate_predictions(
        refs=test_out["refs"],
        hyps=test_out["hyps"],
        token_conf=test_out["token_conf"],
        token_correct=test_out["token_correct"],
        concept_lexicon=concept_lexicon,
        sample_allowed_concepts=test_allowed,
        bins=cfg.evaluation.calibration_bins,
    )

    tqdm.write("Saving metrics, predictions, and visualization artifacts...")
    save_json(history, out_dir / "training_history.json")
    save_json(test_metrics, out_dir / "test_metrics.json")
    pd.DataFrame(test_out["rows"]).to_csv(out_dir / "test_predictions.csv", index=False)

    save_metric_bar(test_metrics, out_dir / "metric_bar.png")
    save_radar(test_metrics, out_dir / "metric_radar.png")
    save_calibration_curve(
        test_out["token_conf"],
        test_out["token_correct"],
        out_dir / "calibration_curve.png",
        bins=cfg.evaluation.calibration_bins,
    )
    save_uncertainty_hist(test_out["token_entropy"], out_dir / "uncertainty_hist.png")
    save_training_curves(history, out_dir / "training_curves.png")
    save_keyword_graph(graph.node2id, graph.neighbors, out_dir / "knowledge_graph_snapshot.png")

    return {
        "history": history,
        "test_metrics": test_metrics,
        "output_dir": str(out_dir),
    }
