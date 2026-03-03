from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def _tok(text: str) -> List[str]:
    return str(text).lower().split()


def bleu_scores(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    smooth = SmoothingFunction().method1
    out = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}

    for r, h in zip(refs, hyps):
        rt = _tok(r)
        ht = _tok(h)
        out["bleu1"].append(sentence_bleu([rt], ht, weights=(1, 0, 0, 0), smoothing_function=smooth))
        out["bleu2"].append(sentence_bleu([rt], ht, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth))
        out["bleu3"].append(sentence_bleu([rt], ht, weights=(1 / 3, 1 / 3, 1 / 3, 0), smoothing_function=smooth))
        out["bleu4"].append(sentence_bleu([rt], ht, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth))

    return {k: float(np.mean(v)) for k, v in out.items()}


def rouge_l(refs: Sequence[str], hyps: Sequence[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    return float(np.mean(vals))


def cider_lite(refs: Sequence[str], hyps: Sequence[str]) -> float:
    # Lightweight CIDEr-style consensus based on TF-IDF cosine.
    docs = [_tok(r) for r in refs] + [_tok(h) for h in hyps]
    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1

    n_docs = len(docs)

    def tfidf(tokens):
        tf = Counter(tokens)
        vec = {}
        for t, c in tf.items():
            idf = np.log((1 + n_docs) / (1 + df[t])) + 1
            vec[t] = c * idf
        return vec

    def cosine(a, b):
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        da = np.sqrt(sum(v * v for v in a.values()))
        db = np.sqrt(sum(v * v for v in b.values()))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    vals = []
    for r, h in zip(refs, hyps):
        vals.append(cosine(tfidf(_tok(r)), tfidf(_tok(h))))
    return float(np.mean(vals))


def extract_concepts(text: str, concept_lexicon: Iterable[str]) -> Set[str]:
    t = str(text).lower()
    found = {c for c in concept_lexicon if c and c in t}
    return found


def clinical_concept_coverage(refs: Sequence[str], hyps: Sequence[str], concept_lexicon: Iterable[str]) -> float:
    vals = []
    for r, h in zip(refs, hyps):
        rc = extract_concepts(r, concept_lexicon)
        hc = extract_concepts(h, concept_lexicon)
        if not rc:
            vals.append(1.0)
        else:
            vals.append(len(rc & hc) / max(1, len(rc)))
    return float(np.mean(vals))


def clinical_hallucination_rate(
    refs: Sequence[str],
    hyps: Sequence[str],
    sample_allowed_concepts: Sequence[Set[str]],
    concept_lexicon: Iterable[str],
) -> float:
    vals = []
    for r, h, allowed in zip(refs, hyps, sample_allowed_concepts):
        rc = extract_concepts(r, concept_lexicon)
        hc = extract_concepts(h, concept_lexicon)
        if not hc:
            vals.append(0.0)
            continue
        invalid = hc - (rc | allowed)
        vals.append(len(invalid) / len(hc))
    return float(np.mean(vals))


def uncertainty_calibration_score(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    assert conf.shape == correct.shape
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() == 0:
            continue
        bin_acc = correct[m].mean()
        bin_conf = conf[m].mean()
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)
    return float(1.0 - ece)


def compute_all_metrics(
    refs: Sequence[str],
    hyps: Sequence[str],
    concept_lexicon: Iterable[str],
    sample_allowed_concepts: Sequence[Set[str]],
    token_conf: np.ndarray,
    token_correct: np.ndarray,
    bins: int = 10,
) -> Dict[str, float]:
    out = {}
    out.update(bleu_scores(refs, hyps))
    out["rougeL"] = rouge_l(refs, hyps)
    out["cider_lite"] = cider_lite(refs, hyps)
    out["c3s"] = clinical_concept_coverage(refs, hyps, concept_lexicon)
    out["chr"] = clinical_hallucination_rate(refs, hyps, sample_allowed_concepts, concept_lexicon)
    out["ucs"] = uncertainty_calibration_score(token_conf, token_correct, bins=bins)
    return out
