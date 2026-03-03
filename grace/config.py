from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    dataset_root: str = "./DeepEyeNet"
    train_csv: str = "train.csv"
    valid_csv: str = "valid.csv"
    test_csv: str = "test.csv"
    image_size: int = 224
    max_report_len: int = 160
    max_keywords: int = 12
    num_workers: int = 2


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    decoder_layers: int = 4
    dropout: float = 0.2
    fpn_out_channels: int = 160
    graph_hidden_dim: int = 256
    graph_layers: int = 2
    pretrained_backbone: bool = False
    relation_types: List[str] = field(default_factory=lambda: ["co_occurs", "lexical_related"])


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 8
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    clinical_weight_delta: float = 3.0
    alpha_ce: float = 1.0
    beta_sem: float = 0.2
    gamma_kg: float = 0.2
    early_stop_patience: int = 5


@dataclass
class EvalConfig:
    mc_dropout_passes: int = 20
    calibration_bins: int = 10
    max_gen_len: int = 160


@dataclass
class GRACEConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./outputs"


def load_config(path: str | Path) -> GRACEConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = GRACEConfig()
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "train" in raw:
        cfg.train = TrainConfig(**raw["train"])
    if "evaluation" in raw:
        cfg.evaluation = EvalConfig(**raw["evaluation"])
    if "output_dir" in raw:
        cfg.output_dir = raw["output_dir"]
    return cfg
