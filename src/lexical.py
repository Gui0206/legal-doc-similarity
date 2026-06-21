"""Eixo LEXICAL: o quanto dois documentos compartilham o mesmo texto literal.

É o sinal que distingue "cópia/versão" de "apenas mesmo vocabulário jurídico".
Baseado em sobreposição de shingles (sequências), não de palavras soltas.
"""
from __future__ import annotations

from difflib import SequenceMatcher


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float], default_idf: float) -> float:
    """Jaccard ponderado por IDF.

    Cada shingle contribui com seu peso IDF. Boilerplate (IDF baixo) quase não
    conta; trechos distintivos compartilhados (IDF alto) dominam. Resultado:
    dois docs que só compartilham o template têm weighted_jaccard ~0.
    """
    union = a | b
    if not union:
        return 1.0
    inter = a & b
    w_inter = sum(idf.get(s, default_idf) for s in inter)
    w_union = sum(idf.get(s, default_idf) for s in union)
    return w_inter / w_union if w_union else 0.0


def edit_ratio(sents_a: list[str], sents_b: list[str]) -> float:
    """Razão de similaridade por alinhamento de sentenças (estilo diff/LCS).

    ~1.0 = praticamente idêntico; valores intermediários = edições localizadas
    (típico de versão); baixo = documentos distintos.
    """
    return SequenceMatcher(None, sents_a, sents_b).ratio()
