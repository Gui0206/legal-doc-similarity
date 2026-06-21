"""Eixo SEMÂNTICO: o quanto dois documentos falam sobre a mesma coisa.

IMPORTANTE (honestidade intelectual):
O eixo semântico aqui é um PROXY offline e determinístico (sem download de
modelos). Usamos cosseno de term-frequency SEM IDF (`use_idf=False`), de
propósito: ele reproduz o comportamento que nos interessa demonstrar dos
EMBEDDINGS densos — eles não fazem down-weighting de termos, então linguagem
jurídica compartilhada (boilerplate + vocabulário da área) infla a
similaridade. É exatamente aí que mora a "armadilha do boilerplate".

(Se usássemos TF-IDF, o próprio IDF do vetorizador já derrubaria o boilerplate
e mascararia a armadilha — o que NÃO acontece com embeddings reais. Por isso
TF puro é o proxy mais fiel ao modo de falha que queremos expor.)

Em produção troca-se este backend por embeddings de sentença (ex.: multilingual
MiniLM). `SemanticModel.cosine` é a única dependência do pipeline -> troca de
uma linha. Embeddings capturam paráfrase melhor; a armadilha do boilerplate se
manifesta igual, pois é dirigida por vocabulário/tópico compartilhado.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticModel:
    def __init__(self, ngram_range: tuple[int, int] = (1, 2)):
        # use_idf=False -> cosseno de term-frequency: similaridade topica/de
        # vocabulario que (como embeddings densos) e enganada por boilerplate.
        self._vec = TfidfVectorizer(ngram_range=ngram_range, use_idf=False, sublinear_tf=True)
        self._index: dict[str, int] = {}
        self._matrix = None

    def fit(self, doc_ids: list[str], texts: list[str]) -> "SemanticModel":
        self._matrix = self._vec.fit_transform(texts)
        self._index = {d: i for i, d in enumerate(doc_ids)}
        return self

    def cosine(self, doc_id_a: str, doc_id_b: str) -> float:
        i, j = self._index[doc_id_a], self._index[doc_id_b]
        return float(cosine_similarity(self._matrix[i], self._matrix[j])[0, 0])

    def cosine_text(self, text_a: str, text_b: str) -> float:
        """Cosseno entre dois textos avulsos (usado em testes)."""
        v = self._vec.transform([text_a, text_b])
        return float(cosine_similarity(v[0], v[1])[0, 0])
