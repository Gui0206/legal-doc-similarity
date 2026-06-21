"""Testes de invariantes do núcleo da cascata."""
import math

import numpy as np
import pytest

from src import lexical, normalize
from src.fingerprint import Document, compute_idf, word_shingles
from src.minhash_lsh import MinHasher, estimated_jaccard
from src.pipeline import SimilarityIndex, Thresholds, classify

BOILER = ("excelentissimo senhor doutor juiz de direito da vara civel da comarca "
          "com fundamento no artigo 319 do codigo de processo civil diante do exposto "
          "requer a procedencia total da demanda termos em que pede deferimento")


def test_normalize_remove_ruido_ocr():
    a = normalize.normalize("Folha 3\nO  autorr  sofreu  dano.\npag. 3/8")
    b = normalize.normalize("o autor sofreu dano")
    # cabeçalho/rodapé somem; sobra conteúdo comparável
    assert "folha" not in a and "pag" not in a


def test_hash_igual_para_copia_exata():
    t = "o autor sofreu negativacao indevida no valor de cinco mil reais"
    d1 = Document("a", "c1", t)
    d2 = Document("b", "c1", t)
    assert d1.sha256 == d2.sha256


def test_minhash_aproxima_jaccard():
    h = MinHasher(num_perm=256, seed=1)
    toks_a = (BOILER + " o autor sofreu dano moral grave e duradouro").split()
    toks_b = (BOILER + " o autor sofreu dano material leve e pontual").split()
    A, B = word_shingles(toks_a, 5), word_shingles(toks_b, 5)
    true_j = lexical.jaccard(A, B)
    est_j = estimated_jaccard(h.signature(A), h.signature(B))
    assert abs(true_j - est_j) < 0.08  # erro de estimativa pequeno


def test_idf_derruba_boilerplate():
    # boilerplate aparece em todos; trecho distintivo (com numero unico) em um so
    base = BOILER.split()
    sets = []
    for i in range(10):
        toks = base + f"fato distintivo numero {i} exclusivo".split()
        sets.append(word_shingles(toks, 5))
    idf = compute_idf(sets)
    boiler_sh = " ".join(base[:5])              # shingle 100% boilerplate (df=10)
    distinct_sh = next(s for s in sets[0] if "exclusivo" in s)  # contem o numero -> df=1
    assert idf[boiler_sh] < idf[distinct_sh]    # boilerplate pesa menos


def test_armadilha_boilerplate_classifica_diferente():
    """Dois docs com MESMO boilerplate mas fatos diferentes -> 'diferente'.

    O IDF e dependente do corpus: precisa de um corpus representativo (onde o
    boilerplate e ubiquo) para conseguir derruba-lo. Em producao o indice tem o
    corpus inteiro; aqui simulamos isso com varios docs que compartilham o template.
    """
    doc_base = BOILER + " o autor teve seu nome negativado indevidamente apos quitar a divida"
    doc_trap = BOILER + " o passageiro teve seu voo cancelado sem qualquer assistencia material"
    # corpus de fundo: muitos docs com o mesmo boilerplate (torna-o ubiquo -> IDF baixo)
    bg = [(f"bg{i}", "c1", BOILER + f" fato corriqueiro distinto numero {i} apenas") for i in range(15)]
    idx = SimilarityIndex().build(
        [("base", "c1", doc_base), ("trap", "c1", doc_trap), ("copy", "c1", doc_base)] + bg)
    trap = idx.score_pair("base", "trap")
    copy = idx.score_pair("base", "copy")
    # invariante central: o trap NAO e cópia/versão, e seu lexical fica bem
    # abaixo do de uma cópia real -> e o eixo lexical que neutraliza a armadilha
    assert classify(trap, Thresholds())[0] == "diferente"
    assert trap.weighted_jaccard < 0.5 < copy.weighted_jaccard


def test_copia_exata_tier0():
    t = BOILER + " conteudo identico nos dois documentos para teste de hash"
    idx = SimilarityIndex().build([("a", "c1", t), ("b", "c1", t)])
    s = idx.score_pair("a", "b")
    pred, tier = classify(s, Thresholds())
    assert s.sha_equal and pred == "copia" and tier == 0


def test_lsh_recupera_copia_como_candidato():
    t = BOILER + " documento sobre cobranca indevida de tarifas bancarias ao consumidor"
    idx = SimilarityIndex().build([("a", "c1", t), ("b", "c1", t),
                                   ("x", "c1", BOILER + " assunto totalmente diverso usucapiao")])
    assert "b" in idx.candidates_same_case("a")
