from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")


def save_metric_bar(metrics: Dict[str, float], out_path: str | Path) -> None:
    out_path = Path(out_path)
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x=names, y=vals, palette="viridis")
    ax.set_ylim(0, max(1.0, max(vals) * 1.1))
    plt.xticks(rotation=45, ha="right")
    plt.title("GRACE Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_radar(metrics: Dict[str, float], out_path: str | Path) -> None:
    plot_keys = ["bleu4", "rougeL", "cider_lite", "c3s", "ucs", "chr"]
    vals = []
    for k in plot_keys:
        v = metrics.get(k, 0.0)
        vals.append(1.0 - v if k == "chr" else v)

    angles = np.linspace(0, 2 * np.pi, len(plot_keys), endpoint=False).tolist()
    vals = vals + [vals[0]]
    angles = angles + [angles[0]]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), plot_keys)
    ax.set_ylim(0, 1)
    plt.title("GRACE Clinical-Linguistic Radar (higher is better)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_calibration_curve(conf: np.ndarray, correct: np.ndarray, out_path: str | Path, bins: int = 10) -> None:
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(xs, ys, marker="o", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Token Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_uncertainty_hist(entropy: np.ndarray, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.histplot(entropy, bins=30, kde=True, color="#2a9d8f")
    plt.title("Token Uncertainty Distribution (Entropy)")
    plt.xlabel("Entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_training_curves(history: Sequence[dict], out_path: str | Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_bleu4 = [h["val_bleu4"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, color="#e76f51", marker="o", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#e76f51")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_bleu4, color="#264653", marker="s", label="Val BLEU-4")
    ax2.set_ylabel("BLEU-4", color="#264653")

    plt.title("Training Dynamics")
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_keyword_graph(node2id: Dict[str, int], neighbors: Dict[int, set], out_path: str | Path, max_nodes: int = 60) -> None:
    g = nx.Graph()
    items = list(node2id.items())[:max_nodes]
    kept_ids = {idx for _, idx in items}
    id2node = {idx: name for name, idx in items}

    for idx in kept_ids:
        g.add_node(id2node[idx])
    for i in kept_ids:
        for j in neighbors.get(i, set()):
            if j in kept_ids:
                g.add_edge(id2node[i], id2node[j])

    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(g, seed=42, k=0.4)
    nx.draw_networkx_nodes(g, pos, node_size=500, node_color="#a8dadc")
    nx.draw_networkx_edges(g, pos, alpha=0.4, width=1.2)
    nx.draw_networkx_labels(g, pos, font_size=7)
    plt.title("Retinal Concept Graph Snapshot")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
