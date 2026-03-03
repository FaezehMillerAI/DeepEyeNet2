from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable, List


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


@dataclass
class Vocab:
    stoi: dict
    itos: list

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]


class ClinicalTokenizer:
    def __init__(self, max_vocab_size: int = 12000, min_freq: int = 1):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab: Vocab | None = None

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = ClinicalTokenizer.normalize(text)
        return re.findall(r"[a-z0-9\.\-/]+|[,;:()]", text)

    def fit(self, texts: Iterable[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self.tokenize(str(t)))

        tokens = [tok for tok, freq in counter.items() if freq >= self.min_freq]
        tokens = sorted(tokens, key=lambda x: (-counter[x], x))[: self.max_vocab_size - len(SPECIAL_TOKENS)]
        itos = SPECIAL_TOKENS + tokens
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        self.vocab = Vocab(stoi=stoi, itos=itos)

    def encode(self, text: str, max_len: int) -> List[int]:
        assert self.vocab is not None, "Tokenizer is not fitted."
        ids = [self.vocab.bos_id]
        for tok in self.tokenize(text):
            ids.append(self.vocab.stoi.get(tok, self.vocab.unk_id))
            if len(ids) >= max_len - 1:
                break
        ids.append(self.vocab.eos_id)
        if len(ids) < max_len:
            ids.extend([self.vocab.pad_id] * (max_len - len(ids)))
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        assert self.vocab is not None, "Tokenizer is not fitted."
        out = []
        for tid in token_ids:
            tok = self.vocab.itos[tid]
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            if tok == "<eos>":
                break
            out.append(tok)
        text = " ".join(out)
        text = re.sub(r"\s+([,;:()])", r"\1", text)
        return text.strip()

    @property
    def vocab_size(self) -> int:
        assert self.vocab is not None, "Tokenizer is not fitted."
        return len(self.vocab.itos)
