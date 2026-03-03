from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


@dataclass
class GraphArtifacts:
    node2id: Dict[str, int]
    id2node: Dict[int, str]
    relation_matrices: Dict[str, np.ndarray]
    neighbors: Dict[int, Set[int]]


def _norm_kw(s: str) -> str:
    return str(s).strip().lower()


def parse_keywords_field(kw):
    if isinstance(kw, list):
        return [_norm_kw(x) for x in kw if str(x).strip()]
    txt = str(kw).strip()
    if txt.startswith("[") and txt.endswith("]"):
        txt = txt[1:-1]
    parts = [p.strip(" '\"\t\n") for p in txt.split(",") if p.strip(" '\"\t\n")]
    return [_norm_kw(p) for p in parts]


def build_graph_from_dataframe(df: pd.DataFrame, relation_types: List[str]) -> GraphArtifacts:
    keyword_lists = [parse_keywords_field(x) for x in df["Keywords"].fillna("").tolist()]
    vocab = sorted({k for kws in keyword_lists for k in kws if k})
    node2id = {k: i for i, k in enumerate(vocab)}
    id2node = {i: k for k, i in node2id.items()}
    n = len(vocab)

    mats = {r: np.zeros((n, n), dtype=np.float32) for r in relation_types}
    neighbors = defaultdict(set)

    if "co_occurs" in mats:
        for kws in keyword_lists:
            ids = [node2id[k] for k in kws if k in node2id]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    mats["co_occurs"][a, b] += 1
                    mats["co_occurs"][b, a] += 1
                    neighbors[a].add(b)
                    neighbors[b].add(a)

    if "lexical_related" in mats:
        tokenized = {k: set(k.replace("-", " ").split()) for k in vocab}
        for i, a in enumerate(vocab):
            ta = tokenized[a]
            for j in range(i + 1, len(vocab)):
                b = vocab[j]
                tb = tokenized[b]
                if ta and tb and len(ta & tb) > 0:
                    mats["lexical_related"][i, j] = 1.0
                    mats["lexical_related"][j, i] = 1.0
                    neighbors[i].add(j)
                    neighbors[j].add(i)

    for r, mat in mats.items():
        rowsum = mat.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        mats[r] = mat / rowsum

    return GraphArtifacts(node2id=node2id, id2node=id2node, relation_matrices=mats, neighbors=neighbors)


def keyword_ids_for_sample(keywords: List[str], node2id: Dict[str, int], max_keywords: int, pad_id: int) -> List[int]:
    ids = [node2id[k] for k in keywords if k in node2id][:max_keywords]
    if len(ids) < max_keywords:
        ids.extend([pad_id] * (max_keywords - len(ids)))
    return ids
