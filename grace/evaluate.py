from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

from grace.config import GRACEConfig
from grace.data.deepeyenet import DeepEyeNetDataset, load_split_csv
from grace.data.graph import build_graph_from_dataframe, parse_keywords_field
from grace.data.tokenizer import ClinicalTokenizer, Vocab
from grace.evaluator import evaluate_predictions, run_inference
from grace.models.grace_model import GRACEModel
from grace.utils.common import save_json, set_seed
from grace.utils.viz import (
    save_calibration_curve,
    save_keyword_graph,
    save_metric_bar,
    save_radar,
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


def _reorder_rel_mats(graph, relation_types: List[str], target_node2id: Dict[str, int]) -> torch.Tensor:
    source_node2id = graph.node2id
    if set(source_node2id.keys()) != set(target_node2id.keys()):
        missing = sorted(set(target_node2id) - set(source_node2id))[:10]
        extra = sorted(set(source_node2id) - set(target_node2id))[:10]
        raise ValueError(
            "Graph vocabulary mismatch between checkpoint and dataset. "
            f"Missing sample={missing}, extra sample={extra}"
        )

    n = len(target_node2id)
    rel_tensors = []
    for r in relation_types:
        src = graph.relation_matrices[r]
        out = torch.zeros((n, n), dtype=torch.float32)
        for term_i, i_new in target_node2id.items():
            i_old = source_node2id[term_i]
            for term_j, j_new in target_node2id.items():
                j_old = source_node2id[term_j]
                out[i_new, j_new] = float(src[i_old, j_old])
        rel_tensors.append(out)
    return torch.stack(rel_tensors, dim=0)


def evaluate_grace(cfg: GRACEConfig, checkpoint_path: str | Path) -> Dict:
    set_seed(cfg.train.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".py":
        raise ValueError(
            f"Checkpoint must be a .pt file, got Python file: {checkpoint_path}. "
            "Use best_grace.pt (or last_checkpoint.pt)."
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    train_df = load_split_csv(cfg.data.dataset_root, cfg.data.train_csv)
    test_df = load_split_csv(cfg.data.dataset_root, cfg.data.test_csv)

    graph = build_graph_from_dataframe(train_df, relation_types=cfg.model.relation_types)
    test_allowed = _build_allowed_concepts(test_df, graph)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Backward compatibility for older checkpoints that stored dataclass objects.
        tqdm.write(
            "Detected legacy checkpoint format. Reloading with weights_only=False for compatibility."
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    vocab_itos = ckpt.get("tokenizer_vocab")
    if vocab_itos is None:
        tqdm.write("Checkpoint missing tokenizer vocab; rebuilding tokenizer from train split.")
        tokenizer = ClinicalTokenizer(max_vocab_size=15000, min_freq=1)
        tokenizer.fit(train_df["report_text"].fillna("").tolist())
    else:
        stoi = {tok: i for i, tok in enumerate(vocab_itos)}
        tokenizer = ClinicalTokenizer()
        tokenizer.vocab = Vocab(stoi=stoi, itos=vocab_itos)

    ckpt_node2id = ckpt.get("graph_node2id", graph.node2id)
    graph_pad_id = len(ckpt_node2id)

    test_ds = DeepEyeNetDataset(
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
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

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
    tqdm.write(f"Loaded checkpoint: {checkpoint_path}")

    test_out = run_inference(
        model=model,
        loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_gen_len=cfg.evaluation.max_gen_len,
        mc_passes=cfg.evaluation.mc_dropout_passes,
        stage_name="Evaluation Only (Test)",
    )

    concept_lexicon = sorted(ckpt_node2id.keys())
    test_metrics = evaluate_predictions(
        refs=test_out["refs"],
        hyps=test_out["hyps"],
        token_conf=test_out["token_conf"],
        token_correct=test_out["token_correct"],
        concept_lexicon=concept_lexicon,
        sample_allowed_concepts=test_allowed,
        bins=cfg.evaluation.calibration_bins,
    )

    save_json(test_metrics, out_dir / "test_metrics_eval_only.json")
    pd.DataFrame(test_out["rows"]).to_csv(out_dir / "test_predictions_eval_only.csv", index=False)

    save_metric_bar(test_metrics, out_dir / "metric_bar_eval_only.png")
    save_radar(test_metrics, out_dir / "metric_radar_eval_only.png")
    save_calibration_curve(
        test_out["token_conf"],
        test_out["token_correct"],
        out_dir / "calibration_curve_eval_only.png",
        bins=cfg.evaluation.calibration_bins,
    )
    save_uncertainty_hist(test_out["token_entropy"], out_dir / "uncertainty_hist_eval_only.png")
    save_keyword_graph(ckpt_node2id, graph.neighbors, out_dir / "knowledge_graph_snapshot_eval_only.png")

    return {
        "test_metrics": test_metrics,
        "output_dir": str(out_dir),
        "checkpoint": str(checkpoint_path),
    }
