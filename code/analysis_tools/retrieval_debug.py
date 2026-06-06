"""Standalone Retrieval Debugger for SecQA Stage-2 retrieval.

Runs ONLY the 4-way RRF retrieval (dense + sparse + BM25 + ColBERT) in isolation.
No LLM, no LM Studio, no fact extraction — needs only the local BGE-M3 model and
the precomputed vector_db. Reads an EXISTING S1 plan (does NOT regenerate it).

Why this tool exists
--------------------
``pipeline.retrieve_pages()`` computes four per-signal page scores internally and
then throws them away, keeping only the fused RRF score. Here those four scores
AND each signal's *standalone rank* are surfaced side by side, so we can see
empirically which signal actually finds the gold-evidence pages in this finance
domain. That is the decision basis for whether to weight signals differently in
the RRF (Weighted-RRF) and/or give each retriever a tailored query.

Reuse
-----
The scoring helpers (``sparse_sim``, ``colbert_maxsim``, ``rrf_fuse``) and the
``DocumentStore`` are imported from ``pipeline`` so the math is byte-for-byte
identical to production. Only the orchestration loop is mirrored locally
(``retrieve_with_signals`` below) — that is the one place where the production
function discards the per-signal intermediates we want to keep. Keep it in sync
with ``code/pipeline.py:retrieve_pages`` (lines ~286-388).

The call-site loop must ALSO stay in sync: like ``pipeline.process_question``,
``main`` retrieves one document at a time (each target is expanded into one
retrieval unit per resolved doc) so every year/document gets its own top-N
instead of all docs competing in one shared pool.

Usage
-----
    python code/retrieval_debug.py --s1-results data/results/s1_eval_20260603_224526.jsonl --split train
    python code/retrieval_debug.py --s1-results ... --qids openqa_168
    python code/retrieval_debug.py --s1-results ... --n 10 --no-colbert --top-pages 20
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel

# Reused, unchanged, from the production pipeline — this is the "wiederverwendete"
# retrieval logic. The standalone tool only adds visibility on top of it.
from pipeline import (
    DocumentStore,
    sparse_sim,
    colbert_maxsim,
    rrf_fuse,
    SPLIT_CONFIG,
    RRF_K,
    EMBEDDING_MODEL,
    RESULTS_DIR,
)

DEFAULT_TOP_PAGES = 20

# Signal display order (matches the table sketch in the task).
SIGNALS = ["dense", "sparse", "bm25", "colbert"]
SCORE_FMT = {"dense": ".4f", "sparse": ".3f", "bm25": ".2f", "colbert": ".2f"}

# ── Weighted-RRF sweep (TRAIN-ONLY experiment — see CLAUDE.md test-split rule) ─
# Dense and ColBERT find gold best; sparse/bm25 rank it worse and drag the
# unweighted fusion down. Each tuple sweeps the two lexical weights down while
# holding dense=colbert=1. This is pure post-processing of the per-signal ranks
# that retrieve_with_signals already returns for EVERY candidate, so one run
# (one encode per query) scores all weight sets at once — exact, no top-N approx.
SWEEP_WEIGHTS = [
    ("baseline 1/1/1/1",  {"dense": 1.0, "sparse": 1.0,  "bm25": 1.0,  "colbert": 1.0}),
    ("lex.75  1/.75/.75/1", {"dense": 1.0, "sparse": .75, "bm25": .75, "colbert": 1.0}),
    ("lex.5   1/.5/.5/1",  {"dense": 1.0, "sparse": .5,  "bm25": .5,  "colbert": 1.0}),
    ("lex.35  1/.35/.35/1", {"dense": 1.0, "sparse": .35, "bm25": .35, "colbert": 1.0}),
    ("lex.25  1/.25/.25/1", {"dense": 1.0, "sparse": .25, "bm25": .25, "colbert": 1.0}),
    ("lex.1   1/.1/.1/1",  {"dense": 1.0, "sparse": .1,  "bm25": .1,  "colbert": 1.0}),
    ("lex0    1/0/0/1",    {"dense": 1.0, "sparse": 0.0, "bm25": 0.0, "colbert": 1.0}),
]


def weighted_rrf_ranks(records, weights, k):
    """Re-rank ALL candidate records under per-signal weights.

    ``records`` is the full candidate list from ``retrieve_with_signals`` (each
    holds the 1-indexed per-signal rank that RRF consumes). Returns
    ``{(doc, page): new_rank}``. With every weight = 1 this reproduces the stored
    ``rrf_rank`` exactly — the built-in calibration check in ``print_sweep``.
    """
    scored = []
    for r in records:
        s = 0.0
        for sig in SIGNALS:
            rk = r.get(f"{sig}_rank")
            w = weights.get(sig, 0.0)
            if rk and w:
                s += w / (k + rk)
        scored.append((s, (r["doc_name"], r["page"])))
    scored.sort(key=lambda x: x[0], reverse=True)
    return {key: i + 1 for i, (_, key) in enumerate(scored)}


# ── Core: retrieval with all per-signal scores + ranks kept ─────────────────

def retrieve_with_signals(doc_store, embed_model, semantic_query,
                          resolved_docs, use_colbert):
    """Mirror of ``pipeline.retrieve_pages`` scoring, but nothing is discarded.

    Returns ``(records, meta)`` where ``records`` is one dict per candidate
    ``(doc, page)`` sorted by RRF rank, each holding the score AND the rank for
    every signal plus the fused RRF score/rank. ``meta`` has ``n_pages`` and the
    list of docs actually scored.

    FAITHFUL to code/pipeline.py:retrieve_pages — the encode, per-chunk scoring,
    best-chunk-per-page reduction, and RRF fusion are identical; we merely also
    record the per-signal rankings (which is exactly what feeds the RRF).
    """
    # --- Encode query (identical to pipeline) ---
    enc = embed_model.encode(
        [semantic_query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=use_colbert,
    )
    query_dense = np.array(enc["dense_vecs"], dtype=np.float32)
    query_dense = query_dense / (
        np.linalg.norm(query_dense, axis=1, keepdims=True) + 1e-8
    )
    query_sparse = enc["lexical_weights"][0]
    query_colbert = enc["colbert_vecs"][0] if use_colbert else None

    # --- Score chunks across all docs, keyed by (doc_name, page) ---
    page_dense, page_bm25, page_sparse, page_colbert = {}, {}, {}, {}
    used_docs = []

    for doc_name in resolved_docs:
        doc_data = doc_store.get(doc_name)
        if doc_data is None:
            print(f"  WARNING: {doc_name} not found in vector_db, skipping")
            continue
        used_docs.append(doc_name)

        meta = doc_data["metadata"]
        all_dense = doc_data["dense_vecs"]
        all_sparse = doc_data["sparse_vecs"]
        bm25 = doc_data["bm25"]
        all_colbert = doc_data.get("colbert_vecs")

        dense_scores = (query_dense @ all_dense.T)[0]
        bm25_scores = bm25.get_scores(semantic_query.lower().split())
        sparse_scores = np.array([sparse_sim(query_sparse, s) for s in all_sparse])
        if use_colbert and all_colbert is not None:
            colbert_scores = np.array(
                [colbert_maxsim(query_colbert, c) for c in all_colbert]
            )
        else:
            colbert_scores = np.zeros(len(meta))

        # Reduce to best chunk score per (doc, page), per signal.
        for idx in range(len(meta)):
            pg = int(meta[idx].get("page", 0))
            key = (doc_name, pg)
            if key not in page_dense or dense_scores[idx] > page_dense[key]:
                page_dense[key] = float(dense_scores[idx])
            if key not in page_bm25 or bm25_scores[idx] > page_bm25[key]:
                page_bm25[key] = float(bm25_scores[idx])
            if key not in page_sparse or sparse_scores[idx] > page_sparse[key]:
                page_sparse[key] = float(sparse_scores[idx])
            if key not in page_colbert or colbert_scores[idx] > page_colbert[key]:
                page_colbert[key] = float(colbert_scores[idx])

    if not page_dense:
        return [], {"n_pages": 0, "docs": used_docs}

    # --- Per-signal rankings (identical construction to pipeline's RRF input) ---
    dense_ranked = sorted(page_dense, key=page_dense.get, reverse=True)
    bm25_ranked = sorted(page_bm25, key=page_bm25.get, reverse=True)
    sparse_ranked = sorted(page_sparse, key=page_sparse.get, reverse=True)
    rankings = [dense_ranked, bm25_ranked, sparse_ranked]
    if use_colbert:
        colbert_ranked = sorted(page_colbert, key=page_colbert.get, reverse=True)
        rankings.append(colbert_ranked)

    rrf_ranked, rrf_scores = rrf_fuse(rankings, k=RRF_K)

    # 1-indexed rank of each page within each signal — exactly the rank RRF uses.
    dense_rk = {k: i + 1 for i, k in enumerate(dense_ranked)}
    bm25_rk = {k: i + 1 for i, k in enumerate(bm25_ranked)}
    sparse_rk = {k: i + 1 for i, k in enumerate(sparse_ranked)}
    colbert_rk = (
        {k: i + 1 for i, k in enumerate(colbert_ranked)} if use_colbert else {}
    )
    rrf_rk = {k: i + 1 for i, k in enumerate(rrf_ranked)}

    records = []
    for key in rrf_ranked:  # already in RRF-rank order
        dn, pg = key
        records.append({
            "doc_name": dn,
            "page": pg,
            "rrf_rank": rrf_rk[key],
            "rrf_score": rrf_scores[key],
            "dense_rank": dense_rk[key],
            "dense_score": page_dense[key],
            "sparse_rank": sparse_rk[key],
            "sparse_score": page_sparse[key],
            "bm25_rank": bm25_rk[key],
            "bm25_score": page_bm25[key],
            "colbert_rank": colbert_rk.get(key) if use_colbert else None,
            "colbert_score": page_colbert[key] if use_colbert else None,
        })

    return records, {"n_pages": len(rrf_ranked), "docs": used_docs}


# ── Console rendering ───────────────────────────────────────────────────────

_SEP = "─" * 96
_GOLD = "★"


def _cell(score, rank, fmt, width):
    """Format a 'score (rN)' cell, left-justified to ``width``."""
    if score is None or rank is None:
        return "—".ljust(width)
    return f"{score:{fmt}} (r{rank})".ljust(width)


def print_target_table(records, gold_keys, top_pages, use_colbert):
    """Top-N pages by RRF, with all four per-signal scores+ranks side by side."""
    gold_set = set(gold_keys)
    top = records[:top_pages]
    docw = min(max((len(r["doc_name"]) for r in top), default=12), 22)
    sw = 15  # signal cell width
    rw = 17  # rrf cell width

    header = (
        f"  {'RRF#':>4}  {'Doc':<{docw}} {'Page':>5}  "
        f"{'Dense':<{sw}}{'Sparse':<{sw}}{'BM25':<{sw}}"
        f"{('ColBERT' if use_colbert else 'ColBERT(off)'):<{sw}}{'RRF':<{rw}}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for r in top:
        key = (r["doc_name"], r["page"])
        marker = f"  {_GOLD} GOLD" if key in gold_set else ""
        cells = (
            _cell(r["dense_score"], r["dense_rank"], SCORE_FMT["dense"], sw)
            + _cell(r["sparse_score"], r["sparse_rank"], SCORE_FMT["sparse"], sw)
            + _cell(r["bm25_score"], r["bm25_rank"], SCORE_FMT["bm25"], sw)
            + _cell(r["colbert_score"], r["colbert_rank"], SCORE_FMT["colbert"], sw)
            + _cell(r["rrf_score"], r["rrf_rank"], ".6f", rw)
        )
        print(
            f"  {r['rrf_rank']:>4}  {r['doc_name']:<{docw}} {r['page']:>5}  "
            f"{cells}{marker}"
        )


def print_gold_section(gold_rows, top_pages, use_colbert):
    """For EVERY gold page (even outside top-N): RRF rank + per-signal rank/score."""
    if not gold_rows:
        print("  Gold: no gold evidence pages fall in this target's docs.")
        return
    print(f"  Gold evidence pages (all, even outside top {top_pages}):")
    for g in gold_rows:
        tag = f"{_GOLD} {g['doc_name']} p.{g['page']}"
        if not g["in_retrieved_doc"]:
            print(f"    {tag}  → doc NOT retrieved (S1 did not resolve it / not in vector_db)")
            continue
        if not g["reachable"]:
            print(f"    {tag}  → NOT RANKED — no chunk indexed on this page")
            continue
        loc = "in top" if g["rrf_rank"] <= top_pages else "OUTSIDE top"
        parts = [
            f"dense r{g['dense_rank']} ({g['dense_score']:.3f})",
            f"sparse r{g['sparse_rank']} ({g['sparse_score']:.2f})",
            f"bm25 r{g['bm25_rank']} ({g['bm25_score']:.2f})",
        ]
        if use_colbert:
            parts.append(f"colbert r{g['colbert_rank']} ({g['colbert_score']:.2f})")
        print(
            f"    {tag}  → RRF r{g['rrf_rank']} ({g['rrf_score']:.6f}) "
            f"[{loc} {top_pages}, of {g['n_pages']} pages]"
        )
        print(f"        " + "  ".join(parts))


# ── Aggregation across all processed gold pages ─────────────────────────────

def _gold_pref_key(g):
    """Sort key for picking the representative of a gold page seen under several
    targets (e.g. two targets of one question resolving the same doc). Higher is
    better: reachable beats no-chunk beats doc-not-retrieved; among reachable, a
    lower RRF rank wins. Used to dedupe so the aggregate counts each gold page
    once per question, never once per target."""
    rr = g["rrf_rank"] if g["rrf_rank"] is not None else 10 ** 9
    return (g["reachable"], g["in_retrieved_doc"], -rr)


def print_aggregate(agg, use_colbert):
    """Mean/median gold rank and recall@k per signal — which signal finds gold."""
    reachable = [g for g in agg if g["reachable"]]
    unreachable = [g for g in agg if g["in_retrieved_doc"] and not g["reachable"]]
    unretrieved = [g for g in agg if not g["in_retrieved_doc"]]

    print(f"\n{_SEP}")
    print(f"  AGGREGATE — {len(agg)} gold pages "
          f"({len(reachable)} ranked, {len(unreachable)} no-chunk, "
          f"{len(unretrieved)} doc-not-retrieved)")
    print(f"{_SEP}")
    if not reachable:
        print("  No reachable gold pages to aggregate.")
        return

    signals = SIGNALS if use_colbert else [s for s in SIGNALS if s != "colbert"]
    cols = signals + ["rrf"]
    ks = [1, 5, 10, 20]

    head = f"  {'Signal':<10}{'mean':>8}{'median':>8}{'best':>6}{'worst':>7}  " \
           + "".join(f"{'r@' + str(k):>8}" for k in ks)
    print(head)
    print(f"  {'-' * (len(head) - 2)}")
    for sig in cols:
        ranks = [g[f"{sig}_rank"] for g in reachable if g.get(f"{sig}_rank")]
        if not ranks:
            continue
        arr = np.array(ranks, dtype=float)
        recalls = "".join(
            f"{np.mean(arr <= k):>8.2f}" for k in ks
        )
        label = "RRF" if sig == "rrf" else sig.capitalize()
        print(
            f"  {label:<10}{arr.mean():>8.1f}{np.median(arr):>8.1f}"
            f"{int(arr.min()):>6}{int(arr.max()):>7}  {recalls}"
        )
    print("\n  r@k = fraction of the %d ranked gold pages this signal alone "
          "places at rank ≤ k." % len(reachable))
    print("  Lower mean/median rank and higher r@k = better at finding gold.")


def print_sweep(sweep_gold):
    """Gold recall@k under each Weighted-RRF set. Returns a JSON-able summary.

    ``sweep_gold[label]`` maps each distinct ranked gold page → its best (min)
    rank under that weight set. The denominator is identical across sets, so the
    rows are directly comparable. Baseline (all-ones) must match the RRF row of
    the aggregate above — that is the calibration check.
    """
    ks = [1, 3, 5, 10, 20]
    base_label = SWEEP_WEIGHTS[0][0]
    n = len(sweep_gold[base_label])
    print(f"\n{_SEP}")
    print(f"  WEIGHTED-RRF SWEEP — exact gold recall@k over {n} ranked gold pages")
    print(f"  (full candidate set re-ranked per weight; dense=colbert=1 fixed)")
    print(f"{_SEP}")
    head = (f"  {'weights d/s/b/c':<22}"
            + "".join(f"{'r@' + str(k):>8}" for k in ks)
            + f"{'mean':>8}{'median':>8}")
    print(head)
    print(f"  {'-' * (len(head) - 2)}")

    out = {}
    base_rec = None
    for label, _ in SWEEP_WEIGHTS:
        ranks = list(sweep_gold[label].values())
        if not ranks:
            continue
        arr = np.array(ranks, dtype=float)
        rec = {k: round(float(np.mean(arr <= k)), 4) for k in ks}
        if base_rec is None:
            base_rec = rec
        out[label] = {
            "recall": rec,
            "mean_rank": round(float(arr.mean()), 2),
            "median_rank": float(np.median(arr)),
        }
        cells = "".join(f"{rec[k]:>8.3f}" for k in ks)
        # mark the best-so-far improvement over baseline at r@10
        delta = rec[10] - base_rec[10]
        tag = f"  (r@10 {'+' if delta >= 0 else ''}{delta:.3f})" if label != base_label else ""
        print(f"  {label:<22}{cells}{arr.mean():>8.1f}{np.median(arr):>8.1f}{tag}")

    print(f"\n  n_gold={n}. Baseline r@10={base_rec[10]:.3f} must match the RRF "
          f"row of the aggregate (calibration).")
    print("  Read down the lex column: pick the smallest weight before r@5/r@10 "
          "stops improving.")
    return {"n_gold_pages": n, "per_weight": out}


# ── Output files ────────────────────────────────────────────────────────────

_GOLD_FIELDS = [
    "qid", "target_idx", "company", "doc_name", "page",
    "in_retrieved_doc", "reachable", "n_pages",
    "rrf_rank", "rrf_score",
    "dense_rank", "dense_score", "sparse_rank", "sparse_score",
    "bm25_rank", "bm25_score", "colbert_rank", "colbert_score",
]
_PAGE_FIELDS = [
    "qid", "target_idx", "company", "doc_name", "page", "is_gold", "n_pages",
    "rrf_rank", "rrf_score",
    "dense_rank", "dense_score", "sparse_rank", "sparse_score",
    "bm25_rank", "bm25_score", "colbert_rank", "colbert_score",
]


def write_outputs(out_dir, prefix, ts, jsonl_records, page_rows, gold_rows,
                  agg, config, sweep_summary=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{ts}"

    jsonl_path = out_dir / f"{base}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pages_path = out_dir / f"{base}_pages.csv"
    with open(pages_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PAGE_FIELDS)
        w.writeheader()
        for row in page_rows:
            w.writerow({k: row.get(k) for k in _PAGE_FIELDS})

    gold_path = out_dir / f"{base}_gold.csv"
    with open(gold_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_GOLD_FIELDS)
        w.writeheader()
        for row in gold_rows:
            w.writerow({k: row.get(k) for k in _GOLD_FIELDS})

    # Aggregate summary (per-signal gold-rank stats).
    summary = {"config": config, "n_gold_pages": len(agg)}
    reachable = [g for g in agg if g["reachable"]]
    summary["n_ranked"] = len(reachable)
    summary["n_no_chunk"] = sum(
        1 for g in agg if g["in_retrieved_doc"] and not g["reachable"]
    )
    summary["n_doc_not_retrieved"] = sum(1 for g in agg if not g["in_retrieved_doc"])
    signals = SIGNALS if config["use_colbert"] else [s for s in SIGNALS if s != "colbert"]
    per_signal = {}
    for sig in signals + ["rrf"]:
        ranks = [g[f"{sig}_rank"] for g in reachable if g.get(f"{sig}_rank")]
        if not ranks:
            continue
        arr = np.array(ranks, dtype=float)
        per_signal[sig] = {
            "mean_rank": round(float(arr.mean()), 3),
            "median_rank": float(np.median(arr)),
            "min_rank": int(arr.min()),
            "max_rank": int(arr.max()),
            **{f"recall@{k}": round(float(np.mean(arr <= k)), 4) for k in (1, 5, 10, 20)},
        }
    summary["per_signal_gold_rank"] = per_signal
    if sweep_summary is not None:
        summary["weighted_rrf_sweep"] = sweep_summary

    summary_path = out_dir / f"{base}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return jsonl_path, pages_path, gold_path, summary_path


# ── Loading ─────────────────────────────────────────────────────────────────

def load_s1_records(path):
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "plan" in rec:  # skip S1 error records
                records[rec["qid"]] = rec
    return records


def load_qa(path):
    qa = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            # Gold evidences are 0-indexed; vector_db is 1-indexed (see CLAUDE.md).
            for ev in q.get("evidences", []):
                ev["page_num"] += 1
            qa[q["qid"]] = q
    return qa


# ── Query-variant sweep (TRAIN-ONLY — Test 1: same query to all retrievers) ───
# S1 gives three text forms per target; production currently feeds the keyword-
# heavy `semantic_query` to ALL retrievers. These variants measure, per signal,
# which query form finds gold best — the basis for deciding whether tailored
# per-retriever queries (Test 2) would pay off. Unlike the weight sweep this
# RE-ENCODES per variant, so it is the encoding cost times the number of variants.
def build_query_variants(target, question):
    """Ordered {label: query_string} from the S1 target + raw question."""
    sq = (target.get("semantic_query") or "").strip()
    ni = (target.get("needed_info") or "").strip()
    q = (question or "").strip()
    variants = {
        "question": q,                            # raw natural-language question
        "keywords": sq,                           # S1 semantic_query (= prod default)
        "question+keywords": f"{q} {sq}".strip(),
        "needed_info": ni,                        # S1 precise NL spec
    }
    return {k: v for k, v in variants.items() if v}


def run_query_sweep(common, s1_records, qa_data, doc_store, embed_model, use_colbert):
    """Per query variant: encode + retrieve per document, record each signal's
    gold rank. Mirrors the main loop's per-document isolation but loops over
    query variants (re-encoding each). Returns ``(query_gold, variant_order,
    cols)`` with ``query_gold[label][signal][(qid, doc, page)] = best rank``."""
    signals = SIGNALS if use_colbert else [s for s in SIGNALS if s != "colbert"]
    cols = signals + ["rrf"]
    query_gold = defaultdict(lambda: defaultdict(dict))
    variant_order = []

    for i, qid in enumerate(common, 1):
        s1 = s1_records[qid]
        qa = qa_data[qid]
        question = qa["question"]
        plan = s1["plan"]
        reslog = s1.get("resolution_log", [])
        evidences = qa.get("evidences", [])
        print(f"  query-sweep {i}/{len(common)}  {qid}            ", end="\r")

        for t_idx, target in enumerate(plan.get("targets", []), 1):
            res = reslog[t_idx - 1] if t_idx - 1 < len(reslog) else {}
            resolved = res.get("docs", res.get("matched", []))
            avail = [d for d in resolved if d in doc_store.available_docs]
            if not avail:
                continue
            variants = build_query_variants(target, question)
            for label in variants:
                if label not in variant_order:
                    variant_order.append(label)
            # distinct gold pages whose doc is among this target's docs
            gold_keys, seen = [], set()
            for ev in evidences:
                key = (ev["doc_name"], ev["page_num"])
                if ev["doc_name"] in avail and key not in seen:
                    seen.add(key)
                    gold_keys.append(key)
            if not gold_keys:
                continue
            for doc_name in dict.fromkeys(avail):
                doc_gold = [(dn, pg) for (dn, pg) in gold_keys if dn == doc_name]
                if not doc_gold:
                    continue
                for label, qtext in variants.items():
                    records, _ = retrieve_with_signals(
                        doc_store, embed_model, qtext, [doc_name], use_colbert)
                    rec_by_key = {(r["doc_name"], r["page"]): r for r in records}
                    for (dn, pg) in doc_gold:
                        rec = rec_by_key.get((dn, pg))
                        if rec is None:
                            continue
                        gk = (qid, dn, pg)
                        for sig in cols:
                            rk = rec.get(f"{sig}_rank")
                            if rk is None:
                                continue
                            d = query_gold[label][sig]
                            if gk not in d or rk < d[gk]:
                                d[gk] = rk
    print()
    return query_gold, variant_order, cols


def print_query_sweep(query_gold, variant_order, cols):
    """Per-signal gold recall@k for each query variant + best-variant summary.

    The summary answers 'is Test 2 worth it?': if different signals prefer
    different query forms, tailored per-retriever queries could help."""
    ks = [1, 5, 10, 20]
    n = max((len(query_gold[v][cols[0]]) for v in variant_order), default=0)
    print(f"\n{_SEP}")
    print(f"  QUERY-SWEEP — per-signal gold recall@k by query variant (~{n} gold pages)")
    print(f"{_SEP}")

    recall = defaultdict(lambda: defaultdict(dict))
    for label in variant_order:
        for sig in cols:
            ranks = list(query_gold[label][sig].values())
            arr = np.array(ranks, dtype=float) if ranks else np.array([])
            for k in ks:
                recall[label][sig][k] = float(np.mean(arr <= k)) if len(arr) else 0.0

    for label in variant_order:
        print(f"\n  query = \"{label}\"")
        head = f"    {'signal':<9}" + "".join(f"{'r@' + str(k):>8}" for k in ks)
        print(head)
        print(f"    {'-' * (len(head) - 4)}")
        for sig in cols:
            cells = "".join(f"{recall[label][sig][k]:>8.3f}" for k in ks)
            lbl = "RRF" if sig == "rrf" else sig.capitalize()
            print(f"    {lbl:<9}{cells}")

    print(f"\n{_SEP}")
    print("  BEST QUERY VARIANT PER SIGNAL (at r@10):")
    best_by_sig = {}
    for sig in cols:
        best = max(variant_order, key=lambda v: recall[v][sig][10])
        best_by_sig[sig] = best
        spread = (max(recall[v][sig][10] for v in variant_order)
                  - min(recall[v][sig][10] for v in variant_order))
        lbl = "RRF" if sig == "rrf" else sig.capitalize()
        print(f"    {lbl:<9} → {best:<20} (r@10 {recall[best][sig][10]:.3f}, "
              f"spread {spread:.3f})")
    distinct = {best_by_sig[s] for s in cols if s != "rrf"}
    print()
    if len(distinct) > 1:
        print("  → Signals prefer DIFFERENT query forms → tailored per-retriever "
              "queries (Test 2) could help: give each signal its best variant.")
    else:
        print(f"  → All signals prefer the SAME variant ({distinct.pop()}) → no "
              "tailored queries needed; just switch the global query to it.")
    return {"n_gold_pages": n,
            "recall": {v: {s: recall[v][s] for s in cols} for v in variant_order},
            "best_per_signal": best_by_sig}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Standalone 4-way retrieval debugger (no LLM)."
    )
    parser.add_argument("--s1-results", required=True, help="Path to S1 eval JSONL")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=None, help="Number of questions")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--qids", nargs="+", help="Specific question IDs")
    parser.add_argument("--top-pages", type=int, default=DEFAULT_TOP_PAGES,
                        help="How many top-RRF pages to show/store (default 20)")
    parser.add_argument("--no-colbert", action="store_true",
                        help="Drop ColBERT → 3-way RRF (saves memory)")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--prefix", default="retrieval_debug")
    parser.add_argument("--sweep", action="store_true",
                        help="Also run the Weighted-RRF gold-recall sweep "
                             "(train-only; exact, no extra encoding)")
    parser.add_argument("--query-sweep", action="store_true",
                        help="Test 1: compare query variants (question/keywords/"
                             "needed_info) by per-signal gold recall (train-only; "
                             "re-encodes per variant). Skips normal per-question output.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    try:  # Windows consoles: make ★ / ─ printable.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    use_colbert = not args.no_colbert
    split = SPLIT_CONFIG[args.split]

    print(f"Loading S1 plans from {args.s1_results}")
    s1_records = load_s1_records(args.s1_results)
    print(f"  {len(s1_records)} plans loaded")

    print(f"Loading QA from {split['qa_file']}")
    qa_data = load_qa(split["qa_file"])
    print(f"  {len(qa_data)} questions loaded")

    common = sorted(set(s1_records) & set(qa_data))
    if args.qids:
        common = [q for q in common if q in set(args.qids)]
    else:
        common = common[args.start:]
        if args.n is not None:
            common = common[:args.n]
    print(f"Processing {len(common)} questions "
          f"(colbert={'ON' if use_colbert else 'OFF'}, top={args.top_pages})")

    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = BGEM3FlagModel(args.embedding_model, use_fp16=True)

    doc_store = DocumentStore(
        split["vector_db"], split["merged_dir"],
        max_cached=8, use_colbert=use_colbert,
    )
    print(f"  {len(doc_store.available_docs)} docs available in vector_db")

    # Test 1 — query-variant sweep: separate, focused path (skips normal output).
    if args.query_sweep:
        query_gold, variant_order, cols = run_query_sweep(
            common, s1_records, qa_data, doc_store, embed_model, use_colbert)
        qsummary = print_query_sweep(query_gold, variant_order, cols)
        if not args.no_save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path(args.out_dir) / f"{args.prefix}_querysweep_{ts}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"split": args.split, "n_questions": len(common),
                           "use_colbert": use_colbert, **qsummary},
                          f, indent=2, ensure_ascii=False)
            print(f"\n  Query-sweep summary saved: {out_path}")
        return

    jsonl_records = []
    page_rows = []
    gold_rows_csv = []
    agg = []  # one entry per gold evidence page, for the aggregate summary
    # Weighted-RRF sweep: per weight set, best (min) rank per distinct gold page.
    sweep_gold = ({label: {} for label, _ in SWEEP_WEIGHTS}
                  if args.sweep else None)

    for i, qid in enumerate(common, 1):
        s1 = s1_records[qid]
        qa = qa_data[qid]
        question = qa["question"]
        plan = s1["plan"]
        reslog = s1.get("resolution_log", [])
        evidences = qa.get("evidences", [])

        print(f"\n{_SEP}")
        print(f"  {i}/{len(common)}  {qid}")
        print(f"{_SEP}")
        print(f"  Q: {question}")
        if qa.get("answer"):
            print(f"  Gold answer: {str(qa['answer'])[:160]}")

        # Docs retrieved across all targets (for 'doc not retrieved' detection).
        retrieved_docs_all = set()
        # Distinct gold pages for THIS question, deduped across targets so the
        # aggregate never double-counts a page two targets both retrieve.
        q_gold = {}

        targets = plan.get("targets", [])
        # Per-target -> per-document: expand each target into one retrieval unit
        # per resolved doc so every year/document gets its OWN top-N (ranked in
        # isolation), mirroring pipeline.process_question. The body below is
        # unchanged and now scores a single-document `available` list.
        # dict.fromkeys dedups while preserving order.
        retrieval_units = []
        for t_idx, target in enumerate(targets, 1):
            res = reslog[t_idx - 1] if t_idx - 1 < len(reslog) else {}
            resolved = res.get("docs", res.get("matched", []))
            avail = [d for d in resolved if d in doc_store.available_docs]
            company = target.get("company", "?")
            semantic_query = target.get("semantic_query", question)

            if not avail:
                print(f"\n  ── Target {t_idx}/{len(targets)}: {company} "
                      f"──  (no resolved docs in vector_db: {resolved})")
                continue
            retrieved_docs_all.update(avail)

            for doc_name in dict.fromkeys(avail):
                retrieval_units.append((t_idx, target, company, semantic_query, [doc_name]))

        # Each unit is a single document with its own top-N (ranked in isolation).
        for t_idx, target, company, semantic_query, available in retrieval_units:

            records, rmeta = retrieve_with_signals(
                doc_store, embed_model, semantic_query, available, use_colbert
            )
            rec_by_key = {(r["doc_name"], r["page"]): r for r in records}
            n_pages = rmeta["n_pages"]

            print(f"\n  ── Target {t_idx}/{len(targets)}: {company} "
                  f"({len(available)} docs: {', '.join(available)}) ──")
            print(f"     query: \"{semantic_query}\"")
            print(f"     {n_pages} candidate pages")
            if not records:
                print("     (no pages scored)")
                continue

            # Gold pages whose doc is among THIS target's retrieved docs.
            target_gold_keys = []
            seen = set()
            for ev in evidences:
                key = (ev["doc_name"], ev["page_num"])
                if ev["doc_name"] in available and key not in seen:
                    seen.add(key)
                    target_gold_keys.append(key)

            print_target_table(records, target_gold_keys, args.top_pages, use_colbert)

            # Weighted-RRF sweep: re-rank this unit's full candidate set under
            # each weight set and keep the best (min) rank per distinct gold page
            # (a page resolved by two targets is counted once, at its best rank).
            if sweep_gold is not None:
                for label, w in SWEEP_WEIGHTS:
                    wr = weighted_rrf_ranks(records, w, RRF_K)
                    for (dn, pg) in target_gold_keys:
                        rank = wr.get((dn, pg))
                        if rank is None:
                            continue
                        gkey = (qid, dn, pg)
                        prev = sweep_gold[label].get(gkey)
                        if prev is None or rank < prev:
                            sweep_gold[label][gkey] = rank

            # Build gold rows (in-retrieved-doc set here; reachable depends on chunk).
            gold_rows = []
            for (dn, pg) in target_gold_keys:
                rec = rec_by_key.get((dn, pg))
                g = {
                    "qid": qid, "target_idx": t_idx, "company": company,
                    "doc_name": dn, "page": pg,
                    "in_retrieved_doc": True,
                    "reachable": rec is not None,
                    "n_pages": n_pages,
                }
                if rec is not None:
                    for fld in ("rrf_rank", "rrf_score", "dense_rank", "dense_score",
                                "sparse_rank", "sparse_score", "bm25_rank", "bm25_score",
                                "colbert_rank", "colbert_score"):
                        g[fld] = rec[fld]
                else:
                    for fld in ("rrf_rank", "rrf_score", "dense_rank", "dense_score",
                                "sparse_rank", "sparse_score", "bm25_rank", "bm25_score",
                                "colbert_rank", "colbert_score"):
                        g[fld] = None
                gold_rows.append(g)

            print_gold_section(gold_rows, args.top_pages, use_colbert)

            # Feed the question-level dedup map (keep best rank across targets).
            for g in gold_rows:
                key = (g["doc_name"], g["page"])
                if key not in q_gold or _gold_pref_key(g) > _gold_pref_key(q_gold[key]):
                    q_gold[key] = g

            # Structured per-page rows (top-N) for the flat CSV.
            top = records[:args.top_pages]
            gold_set = set(target_gold_keys)
            for r in top:
                row = {
                    "qid": qid, "target_idx": t_idx, "company": company,
                    "is_gold": (r["doc_name"], r["page"]) in gold_set,
                    "n_pages": n_pages, **r,
                }
                page_rows.append(row)

            # Nested JSONL record (full per-target detail).
            jsonl_records.append({
                "qid": qid,
                "question": question,
                "target_idx": t_idx,
                "company": company,
                "semantic_query": semantic_query,
                "docs": available,
                "n_candidate_pages": n_pages,
                "use_colbert": use_colbert,
                "top_pages": [
                    {**r, "is_gold": (r["doc_name"], r["page"]) in gold_set}
                    for r in top
                ],
                "gold_pages": gold_rows,
            })

        # Gold evidences whose doc was retrieved by no target at all.
        for ev in evidences:
            if ev["doc_name"] not in retrieved_docs_all:
                key = (ev["doc_name"], ev["page_num"])
                if key in q_gold:
                    continue
                q_gold[key] = {
                    "qid": qid, "target_idx": None, "company": None,
                    "doc_name": ev["doc_name"], "page": ev["page_num"],
                    "in_retrieved_doc": False, "reachable": False, "n_pages": 0,
                    **{fld: None for fld in (
                        "rrf_rank", "rrf_score", "dense_rank", "dense_score",
                        "sparse_rank", "sparse_score", "bm25_rank", "bm25_score",
                        "colbert_rank", "colbert_score")},
                }
                print(f"  {_GOLD} {ev['doc_name']} p.{ev['page_num']}  "
                      f"→ doc NOT retrieved by any target")

        # One row per distinct gold page → drives the aggregate + gold CSV.
        for g in q_gold.values():
            agg.append(g)
            gold_rows_csv.append(g)

    # Final aggregate over all questions.
    print_aggregate(agg, use_colbert)
    sweep_summary = print_sweep(sweep_gold) if sweep_gold is not None else None

    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = {
            "split": args.split,
            "s1_results": args.s1_results,
            "embedding_model": args.embedding_model,
            "use_colbert": use_colbert,
            "top_pages": args.top_pages,
            "rrf_k": RRF_K,
            "n_questions": len(common),
        }
        jsonl_path, pages_path, gold_path, summary_path = write_outputs(
            args.out_dir, args.prefix, ts,
            jsonl_records, page_rows, gold_rows_csv, agg, config, sweep_summary,
        )
        print(f"\n  JSONL (per target):  {jsonl_path}")
        print(f"  Pages CSV (top-N):   {pages_path}")
        print(f"  Gold CSV (per gold): {gold_path}")
        print(f"  Summary JSON:        {summary_path}")


if __name__ == "__main__":
    main()
