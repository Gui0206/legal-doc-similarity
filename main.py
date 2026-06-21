"""Executa a validação completa da abordagem de similaridade.

  python main.py

Gera corpus sintético -> constrói índice -> calibra limiares no treino ->
avalia no teste (held-out por caso) -> compara 2-eixos vs só-semântica vs
só-lexical -> mede recall do LSH e o custo da cascata -> roda um teste de
robustez multi-seed -> salva results/metrics.json e results/scatter.png
"""
from __future__ import annotations

import json
import os
import random
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import evaluate as ev
from src.dataset import build_corpus, labeled_pairs
from src.pipeline import SimilarityIndex, calibrate, classify, needs_llm

RESULTS = os.path.join(os.path.dirname(__file__), "results")
LABEL_COLOR = {"copia": "#16a34a", "versao": "#eab308", "diferente": "#dc2626"}


def case_of(doc_id):  # base_id = case_xxx_base
    return "_".join(doc_id.split("_")[:2])


def build_and_score(corpus_seed: int):
    samples, meta = build_corpus(num_cases=40, seed=corpus_seed)
    pairs = labeled_pairs(samples)
    index = SimilarityIndex().build([(s.doc_id, s.case_id, s.text) for s in samples])
    scored = {(b, o): index.score_pair(b, o) for b, o, _ in pairs}
    return index, samples, pairs, scored, meta


def split(samples, pairs, split_seed: int):
    cases = sorted({s.case_id for s in samples})
    rnd = random.Random(split_seed)
    rnd.shuffle(cases)
    train_cases = set(cases[: len(cases) // 2])
    train = [(b, o, lab) for b, o, lab in pairs if case_of(b) in train_cases]
    test = [(b, o, lab) for b, o, lab in pairs if case_of(b) not in train_cases]
    return train, test


def evaluate_split(scored, train, test):
    tr_scores = [scored[(b, o)] for b, o, _ in train]
    tr_y = [lab for _, _, lab in train]
    te_scores = [scored[(b, o)] for b, o, _ in test]
    te_y = [lab for _, _, lab in test]

    th = calibrate(tr_scores, tr_y)
    two_pred = [classify(s, th)[0] for s in te_scores]
    res_two = ev.evaluate_model("2 eixos (lexical + semantico)", te_y, two_pred)

    sem_hi, sem_lo = ev._calibrate_1d([s.semantic_cosine for s in tr_scores], tr_y)
    sem_pred = [ev._apply_1d(s.semantic_cosine, sem_hi, sem_lo) for s in te_scores]
    res_sem = ev.evaluate_model("baseline so-semantica (cosseno)", te_y, sem_pred)

    lex_hi, lex_lo = ev._calibrate_1d([s.weighted_jaccard for s in tr_scores], tr_y)
    lex_pred = [ev._apply_1d(s.weighted_jaccard, lex_hi, lex_lo) for s in te_scores]
    res_lex = ev.evaluate_model("baseline so-lexical (jaccard idf)", te_y, lex_pred)

    routed = [needs_llm(s, th) for s in te_scores]
    auto_idx = [i for i, r in enumerate(routed) if not r]
    cascade = {
        "tier0_hash_exato": int(sum(s.sha_equal for s in te_scores)),
        "auto_resolvido_sem_llm": len(auto_idx),
        "roteado_ao_llm_tier3": int(sum(routed)),
        "pct_roteado_llm": sum(routed) / len(te_scores),
        "acuracia_nos_auto_resolvidos": (sum(two_pred[i] == te_y[i] for i in auto_idx)
                                         / len(auto_idx) if auto_idx else 0.0),
    }
    return dict(th=th, models=[res_two, res_sem, res_lex], cascade=cascade,
                thr_baselines={"semantico": [sem_hi, sem_lo], "lexical": [lex_hi, lex_lo]},
                two_pred=two_pred, sem_pred=sem_pred, te=test)


def main():
    os.makedirs(RESULTS, exist_ok=True)

    # ---- run principal (detalhado + artefatos) ----
    index, samples, pairs, scored, meta = build_and_score(corpus_seed=7)
    train, test = split(samples, pairs, split_seed=123)
    r = evaluate_split(scored, train, test)
    th = r["th"]
    lsh = ev.lsh_candidate_recall(index, pairs)

    trap = []
    for b, o, lab in pairs:
        if o.endswith("_diffarea"):
            s = scored[(b, o)]
            trap.append({"par": f"{b} x {o}", "semantic_cosine": round(s.semantic_cosine, 3),
                         "weighted_jaccard": round(s.weighted_jaccard, 3),
                         "pred_2eixos": classify(s, th)[0],
                         "pred_so_semantica": ev._apply_1d(
                             s.semantic_cosine, *r["thr_baselines"]["semantico"])})

    # ---- robustez multi-seed ----
    robust = {m: {"macro_f1": [], "false_merge": []}
              for m in ("2 eixos (lexical + semantico)", "baseline so-semantica (cosseno)",
                        "baseline so-lexical (jaccard idf)")}
    for cs in range(5):
        idx2, samp2, pr2, sc2, _ = build_and_score(corpus_seed=100 + cs)
        tr2, te2 = split(samp2, pr2, split_seed=200 + cs)
        rr = evaluate_split(sc2, tr2, te2)
        for mod in rr["models"]:
            robust[mod["name"]]["macro_f1"].append(mod["macro_f1"])
            robust[mod["name"]]["false_merge"].append(mod["false_merge_rate"])
    robust_summary = {name: {
        "macro_f1_mean": st.mean(v["macro_f1"]), "macro_f1_std": st.pstdev(v["macro_f1"]),
        "false_merge_mean": st.mean(v["false_merge"]), "false_merge_std": st.pstdev(v["false_merge"]),
    } for name, v in robust.items()}

    out = {
        "meta": meta,
        "thresholds_2eixos": th.__dict__,
        "thresholds_baselines": r["thr_baselines"],
        "split": {"train_pairs": len(train), "test_pairs": len(test)},
        "lsh_candidate_recall": lsh,
        "cascata_custo": r["cascade"],
        "modelos": r["models"],
        "robustez_5_seeds": robust_summary,
        "amostras_armadilha_boilerplate": trap[:8],
    }
    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))

    _plot(scored, pairs, th, os.path.join(RESULTS, "scatter.png"))
    _print_summary(out)


def _plot(scored, pairs, th, path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for b, o, lab in pairs:
        s = scored[(b, o)]
        ax.scatter(s.weighted_jaccard, s.semantic_cosine, c=LABEL_COLOR[lab],
                   s=42, alpha=0.75, edgecolors="white", linewidths=0.5)
    ax.axvline(th.ver_lex, ls="--", c="#475569", lw=1.2, alpha=0.8)
    ax.text(th.ver_lex + 0.01, 0.02, "limiar lexical\n(versão/diferente)", fontsize=8, c="#475569")
    for lab, col in LABEL_COLOR.items():
        ax.scatter([], [], c=col, label=lab, s=60)
    ax.set_xlabel("Eixo LEXICAL — Jaccard ponderado por IDF, sem boilerplate (shingles 5-gramas)")
    ax.set_ylabel("Eixo SEMANTICO — cosseno (proxy de embedding)")
    ax.set_title("Separacao das classes no espaco lexical x semantico\n"
                 "armadilha do boilerplate = vermelhos no ALTO (semantica alta) e a ESQUERDA "
                 "(lexical ~0)")
    ax.legend(title="rotulo verdadeiro", loc="center right")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _print_summary(out):
    print("=" * 78)
    print("VALIDAÇÃO — Similaridade entre documentos jurídicos")
    print("=" * 78)
    m = out["meta"]
    print(f"Corpus: {m['num_cases']} casos, {m['total_docs']} documentos (seed={m['seed']}) | "
          f"split: {out['split']['train_pairs']} treino / {out['split']['test_pairs']} teste")
    r = out["lsh_candidate_recall"]
    print(f"\nTier 1 (MinHash/LSH) recall de candidatos: {r['recall']:.1%} "
          f"({r['retrieved']}/{r['relevant_pairs']} pares cópia/versão recuperados)")
    c = out["cascata_custo"]
    print(f"Cascata (teste): {c['tier0_hash_exato']} por hash (Tier 0) | "
          f"{c['auto_resolvido_sem_llm']} sem LLM (acurácia {c['acuracia_nos_auto_resolvidos']:.0%}) | "
          f"{c['roteado_ao_llm_tier3']} ({c['pct_roteado_llm']:.0%}) ao LLM (Tier 3)")
    print(f"Limiares 2-eixos calibrados: {out['thresholds_2eixos']}")

    print("\n" + "-" * 78)
    print(f"{'MODELO (held-out, 1 split)':<40}{'macro-F1':>10}{'acurácia':>10}{'falso merge':>13}")
    print("-" * 78)
    for mod in out["modelos"]:
        print(f"{mod['name']:<40}{mod['macro_f1']:>10.3f}{mod['accuracy']:>10.3f}"
              f"{mod['false_merge_rate']:>12.1%}")
    print("-" * 78)
    print("'falso merge' = % de documentos DIFERENTES classificados como cópia/versão")

    print("\nRobustez em 5 seeds (média ± desvio):")
    print(f"{'MODELO':<40}{'macro-F1':>16}{'falso merge':>18}")
    for name, v in out["robustez_5_seeds"].items():
        print(f"{name:<40}{v['macro_f1_mean']:>8.3f} ±{v['macro_f1_std']:.3f}"
              f"{v['false_merge_mean']:>10.1%} ±{v['false_merge_std']:.1%}")

    print("\nArmadilha do boilerplate (documentos diferentes, mesma área/vocabulário):")
    print(f"{'par':<30}{'cosseno':>9}{'jaccard':>9}{'2-eixos':>11}{'só-sem.':>11}")
    for t in out["amostras_armadilha_boilerplate"][:6]:
        par = t["par"].replace("case_", "c").replace("_base", "").replace("_diffarea", "→diff")
        print(f"{par:<30}{t['semantic_cosine']:>9.2f}{t['weighted_jaccard']:>9.2f}"
              f"{t['pred_2eixos']:>11}{t['pred_so_semantica']:>11}")
    print("\nArtefatos: results/metrics.json, results/scatter.png")


if __name__ == "__main__":
    main()
