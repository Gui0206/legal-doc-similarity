"""A cascata de comparação e o classificador de 2 eixos.

Tiers:
  0. Hash exato (SHA-256)          -> cópia exata, custo ~0
  1. MinHash/LSH dentro do caso    -> gera candidatos, custo baixo
  2. Jaccard ponderado por IDF +   -> classifica a maioria, sem LLM
     cosseno + edit ratio
  3. (produção) LLM no diff         -> só na faixa de incerteza  [não exercitado aqui]

O classificador é uma regra interpretável sobre (lexical, semantico, edit_ratio),
com limiares CALIBRADOS por busca em grade num split de treino.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import lexical, normalize
from .fingerprint import Document, attach_signatures, compute_idf
from .minhash_lsh import LSHIndex, MinHasher, estimated_jaccard
from .semantic import SemanticModel

LABELS = ("copia", "versao", "diferente")


@dataclass
class Thresholds:
    copy_lex: float = 0.70   # lexical alto -> cópia
    copy_sem: float = 0.80
    ver_lex: float = 0.30    # lexical médio + semântico alto -> versão
    ver_sem: float = 0.60


@dataclass
class PairScore:
    base_id: str
    other_id: str
    sha_equal: bool
    weighted_jaccard: float
    minhash_jaccard: float
    semantic_cosine: float
    edit_ratio: float
    pred: str = ""
    tier: int = -1


class SimilarityIndex:
    """Constrói os fingerprints, o índice LSH, o IDF e o modelo semântico."""

    def __init__(self, num_perm: int = 128, bands: int = 64, k: int = 5,
                 boiler_df: float = 0.15):
        self.k = k
        self.boiler_df = boiler_df  # shingle em > boiler_df do corpus = boilerplate
        self.hasher = MinHasher(num_perm=num_perm)
        self.lsh = LSHIndex(num_perm=num_perm, bands=bands)
        self.docs: dict[str, Document] = {}
        self.idf: dict[str, float] = {}
        self.boilerplate: set[str] = set()
        self.default_idf: float = 1.0
        self.semantic = SemanticModel()

    def build(self, items: list[tuple[str, str, str]]) -> "SimilarityIndex":
        """items = [(doc_id, case_id, raw_text), ...]"""
        docs = [Document(doc_id=i, case_id=c, raw=t, k=self.k) for i, c, t in items]
        attach_signatures(docs, self.hasher)
        self.docs = {d.doc_id: d for d in docs}
        self.idf = compute_idf([d.shingles for d in docs])
        n = len(docs)
        self.default_idf = math.log((1 + n) / 1.0) + 1.0  # peso de um shingle único
        # "stopword corpus" de boilerplate: shingles ubíquos no corpus.
        df: dict[str, int] = {}
        for d in docs:
            for sh in d.shingles:
                df[sh] = df.get(sh, 0) + 1
        self.boilerplate = {sh for sh, c in df.items() if c / n > self.boiler_df}
        for d in docs:
            self.lsh.add(d.doc_id, d.signature)
        self.semantic.fit([d.doc_id for d in docs], [d.normalized for d in docs])
        return self

    def distinctive(self, doc_id: str) -> set[str]:
        """Shingles do doc após remover o boilerplate ubíquo."""
        return self.docs[doc_id].shingles - self.boilerplate

    # ---- candidatos (Tier 1) ----
    def candidates_same_case(self, doc_id: str) -> set[str]:
        cand = self.lsh.candidates(doc_id)
        case = self.docs[doc_id].case_id
        return {c for c in cand if self.docs[c].case_id == case}

    # ---- score de um par (Tier 0 + 2) ----
    def score_pair(self, a: str, b: str) -> PairScore:
        da, db = self.docs[a], self.docs[b]
        sha_equal = da.sha256 == db.sha256
        # eixo lexical sobre shingles DISTINTIVOS (boilerplate removido):
        # docs que só compartilham o template caem para ~0 aqui.
        wj = lexical.weighted_jaccard(self.distinctive(a), self.distinctive(b),
                                      self.idf, self.default_idf)
        mj = estimated_jaccard(da.signature, db.signature)
        cos = self.semantic.cosine(a, b)
        er = lexical.edit_ratio(normalize.sentences(da.normalized), normalize.sentences(db.normalized))
        return PairScore(a, b, sha_equal, wj, mj, cos, er)


def needs_llm(score: PairScore, th: Thresholds, margin: float = 0.05) -> bool:
    """Tier 3: roteia para o LLM só os pares na FAIXA DE INCERTEZA: topicamente
    parecidos (semântica alta) MAS lexicalmente na fronteira versão/diferente —
    o caso clássico de versão muito reescrita vs. documento diferente da mesma
    área. Pares com sinais claros são decididos barato, sem LLM."""
    if score.sha_equal:
        return False
    lexicalmente_ambiguo = abs(score.weighted_jaccard - th.ver_lex) < margin
    topicamente_proximo = score.semantic_cosine >= 0.85
    return lexicalmente_ambiguo and topicamente_proximo


def classify(score: PairScore, th: Thresholds) -> tuple[str, int]:
    """Regra de decisão de 2 eixos. Retorna (label, tier)."""
    if score.sha_equal:
        return "copia", 0
    if score.weighted_jaccard >= th.copy_lex and score.semantic_cosine >= th.copy_sem:
        return "copia", 2
    if score.weighted_jaccard >= th.ver_lex and score.semantic_cosine >= th.ver_sem:
        return "versao", 2
    return "diferente", 2


# ----------------- calibração de limiares (split de treino) -----------------
def _macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    f1s = []
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def calibrate(scores: list[PairScore], y_true: list[str]) -> Thresholds:
    """Busca em grade os limiares que maximizam o macro-F1 no treino."""
    grid = [round(x, 2) for x in np.arange(0.10, 0.96, 0.05)]
    best, best_f1 = Thresholds(), -1.0
    for cl, cs, vl, vs in product(grid, grid, grid, grid):
        if cl < vl or cs < vs:  # cópia exige limiar >= versão
            continue
        th = Thresholds(cl, cs, vl, vs)
        preds = [classify(s, th)[0] for s in scores]
        f1 = _macro_f1(y_true, preds)
        if f1 > best_f1:
            best_f1, best = f1, th
    return best
