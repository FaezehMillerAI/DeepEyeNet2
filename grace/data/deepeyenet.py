from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .graph import parse_keywords_field, keyword_ids_for_sample


@dataclass
class Sample:
    image_path: str
    keywords: list
    clinical_description: str
    report_text: str


def load_split_csv(root: str | Path, csv_name: str) -> pd.DataFrame:
    root = Path(root)
    path = root / csv_name
    df = pd.read_csv(path)
    needed = ["image_path", "Keywords", "clinical-description", "report_text"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


class DeepEyeNetDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_root: str | Path,
        tokenizer,
        node2id: Dict[str, int],
        max_report_len: int,
        max_keywords: int,
        image_size: int,
        graph_pad_id: int,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.node2id = node2id
        self.max_report_len = max_report_len
        self.max_keywords = max_keywords
        self.graph_pad_id = graph_pad_id

        if augment:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size + 16, image_size + 16)),
                    transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / str(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        keywords = parse_keywords_field(row["Keywords"])
        keyword_ids = keyword_ids_for_sample(
            keywords=keywords,
            node2id=self.node2id,
            max_keywords=self.max_keywords,
            pad_id=self.graph_pad_id,
        )

        report = str(row["report_text"])
        clinical_description = str(row["clinical-description"])
        report_ids = self.tokenizer.encode(report, max_len=self.max_report_len)

        return {
            "image": image,
            "keyword_ids": torch.tensor(keyword_ids, dtype=torch.long),
            "report_ids": torch.tensor(report_ids, dtype=torch.long),
            "report_text": report,
            "clinical_description": clinical_description,
            "keyword_text": "; ".join(keywords),
            "image_path": str(row["image_path"]),
        }
