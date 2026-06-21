"""Fingerprint multi-resolução de um documento (calculado 1x no ingest)."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field

import numpy as np

from . import normalize
from .minhash_lsh import MinHasher

# Cabeçalhos de seção típicos de peças/sentenças (para alinhamento estrutural).
_SECTION_RE = re.compile(
    r"\b(relat[óo]rio|fundamenta[çc][ãa]o|dispositivo|dos fatos|do direito|"
    r"do pedido|da fundamenta[çc][ãa]o)\b",
    re.IGNORECASE,
)


def word_shingles(toks: list[str], k: int = 5) -> set[str]:
    """Conjunto de shingles de k-gramas de palavras (eixo lexical/sequencial)."""
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def split_sections(normalized_text: str) -> dict[str, str]:
    """Quebra o texto por cabeçalhos de seção (heurística simples)."""
    matches = list(_SECTION_RE.finditer(normalized_text))
    if not matches:
        return {"_full": normalized_text}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized_text)
        sections[name] = normalized_text[start:end].strip()
    return sections


@dataclass
class Document:
    doc_id: str
    case_id: str
    raw: str
    k: int = 5
    normalized: str = field(init=False)
    toks: list[str] = field(init=False)
    shingles: set[str] = field(init=False)
    sha256: str = field(init=False)
    sections: dict[str, str] = field(init=False)
    signature: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self):
        self.normalized = normalize.normalize(self.raw)
        self.toks = normalize.tokens(self.normalized)
        self.shingles = word_shingles(self.toks, self.k)
        self.sha256 = hashlib.sha256(self.normalized.encode("utf-8")).hexdigest()
        self.sections = split_sections(self.normalized)


def compute_idf(shingle_sets: list[set[str]]) -> dict[str, float]:
    """IDF por shingle sobre o corpus.

    Shingles ubíquos (boilerplate jurídico) aparecem em quase todos os docs ->
    IDF baixo -> peso baixo. Shingles distintivos -> IDF alto -> peso alto.
    É isso que neutraliza a armadilha do vocabulário compartilhado.
    """
    n = len(shingle_sets)
    df: dict[str, int] = {}
    for s in shingle_sets:
        for sh in s:
            df[sh] = df.get(sh, 0) + 1
    return {sh: math.log((1 + n) / (1 + d)) + 1.0 for sh, d in df.items()}


def attach_signatures(docs: list[Document], hasher: MinHasher) -> None:
    for d in docs:
        d.signature = hasher.signature(d.shingles)
