"""Avaliação: métricas por classe, matriz de confusão, recall do LSH e a
comparação decisiva 2-eixos vs. só-semântica vs. só-lexical."""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .pipeline import LABELS, PairScore, SimilarityIndex, Thresholds, calibrate, classify


def lsh_candidate_recall(index: SimilarityIndex, pairs: list[tuple[str, str, str]]) -> dict:
    """Dos pares que SÃO cópia/versão, qual fração o LSH recupera como candidato?
    (recall do estágio de geração de candidatos — Tier 1)."""
    relevant = [(b, o) for b, o, lab in pairs if lab in ("copia", "versao")]
    hit = 0
    for b, o in relevant:
        cand = index.candidates_same_case(b) | index.candidates_same_case(o)
        if o in index.candidates_same_case(b) or b in index.candidates_same_case(o):
            hit += 1
    return {"relevant_pairs": len(relevant), "retrieved": hit,
            "recall": hit / len(relevant) if relevant else 0.0}


# -------- baselines de 1 eixo (1 feature, 2 limiares calibrados) --------
def _calibrate_1d(values: list[float], y_true: list[str]) -> tuple[float, float]:
    grid = [round(x, 2) for x in np.arange(0.05, 0.99, 0.03)]
    best, best_f1 = (0.7, 0.4), -1.0
    for hi, lo in product(grid, grid):
        if hi < lo:
            continue
        preds = [_apply_1d(v, hi, lo) for v in values]
        f1 = _macro_f1(y_true, preds)
        if f1 > best_f1:
            best_f1, best = f1, (hi, lo)
    return best


def _apply_1d(v: float, hi: float, lo: float) -> str:
    if v >= hi:
        return "copia"
    if v >= lo:
        return "versao"
    return "diferente"


def _macro_f1(y_true, y_pred) -> float:
    f1s = []
    for lab in LABELS:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def false_merge_rate(y_true, y_pred) -> float:
    """Fração de 'diferente' reais classificados como cópia/versão (falso merge)."""
    diffs = [(t, p) for t, p in zip(y_true, y_pred) if t == "diferente"]
    if not diffs:
        return 0.0
    bad = sum(1 for _, p in diffs if p in ("copia", "versao"))
    return bad / len(diffs)


def evaluate_model(name: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    rep = classification_report(y_true, y_pred, labels=list(LABELS),
                                output_dict=True, zero_division=0)
    return {
        "name": name,
        "macro_f1": rep["macro avg"]["f1-score"],
        "accuracy": rep["accuracy"],
        "false_merge_rate": false_merge_rate(y_true, y_pred),
        "per_class": {lab: {"precision": rep[lab]["precision"],
                            "recall": rep[lab]["recall"],
                            "f1": rep[lab]["f1-score"]} for lab in LABELS},
        "confusion_matrix": cm.tolist(),
    }
