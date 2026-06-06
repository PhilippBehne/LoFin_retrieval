"""Chunk-Level-RRF vs. Page-Level-RRF — standalone retrieval experiment (TRAIN-ONLY).

What this measures
------------------
The production retriever (``pipeline.retrieve_pages``) fuses on the PAGE level:
it first reduces every chunk to the best chunk score per ``(doc, page)`` per
signal, then ranks pages per signal, then RRF-fuses the four page rankings. A
page that shows up in several signals with only-mediocre chunks each can sink in
that reduction.

This tool runs the SAME four per-chunk scores but flips the order: fuse FIRST on
the chunk level (rank ALL chunks per signal, no page reduction), RRF the four
chunk rankings, THEN map the fused chunks back to pages. The open question is the
chunk->page mapping; we measure several variants side by side.

It is a pure analysis/debug tool — it does NOT touch the production pipeline. It
needs only the local BGE-M3 model + the precomputed vector_db (no LM Studio).

Faithfulness / isolation
------------------------
Baseline (page-level) and treatment (chunk-level) are derived from the SAME
single encode and the SAME per-chunk score arrays, so the ONLY thing that varies
between them is page-vs-chunk fusion — the chunk effect is isolated exactly. The
baseline reduction is replicated byte-for-byte from ``pipeline.retrieve_pages``
(same max-per-page reduction, same signal order, same ``rrf_fuse`` / ``RRF_K``);
it is the same math ``retrieval_debug.retrieve_with_signals`` already vets.

Reused, unchanged, from production: ``DocumentStore``, ``sparse_sim``,
``colbert_maxsim``, ``rrf_fuse``, ``SPLIT_CONFIG``, ``RRF_K``, ``EMBEDDING_MODEL``,
``RESULTS_DIR``, ``TOP_K_CHUNKS``. ``load_qa`` / ``load_s1_records`` mirror
``retrieval_debug.py`` (identical 0->1-index +1 on gold evidences).

Chunk->page variants (the open design decision)
-----------------------------------------------
Key fact, which this tool also proves empirically: walking the score-sorted chunk
ranking top-down, every page FIRST appears at its highest-scoring chunk. So in
ORDER these three coincide: (a) first-appearance dedup without a cap == (b) walk
until K unique pages == (c) page = MAX chunk score. The genuinely distinct knobs
are therefore only a chunk CAP and SUM aggregation. We measure:

  firstapp   first-appearance dedup over all chunks   = variant (a) no cap = (b)
  cap25/50   first-appearance over the top-N chunks    = variant (a) with a cap
  maxscore   page = best chunk RRF score               = variant (c) max  [calib]
  sumscore   page = sum of its chunks' RRF scores       = variant (c) sum

Trade-offs: firstapp/max are precise ("one strong hit is enough"); sumscore is
recall-friendlier for pages that appear in several signals each only mediocre
(exactly the case page-reduction loses) but risks favouring long/"full" pages;
the cap variants isolate whether the weak long tail of chunks hurts (noise) or
helps (some gold pages only have weak chunks). ``maxscore`` must (up to ties)
reproduce ``firstapp`` — that is the built-in calibration check.

Metric
------
Gold-PAGE recall@{1,5,10,20} + median/mean gold rank, denominator = REACHABLE
gold pages (>=1 chunk on the page) so baseline and every variant share an
identical population (matches ``retrieval_debug``'s aggregate). Computed per query
form (``semantic_query`` AND ``needed_info``) so the chunk effect can be read as
additive-or-not on top of the better query. Each distinct gold page is counted
once per question (best/min rank across units).

Per-document isolation: like ``pipeline.process_question`` every resolved doc is
its own retrieval unit (own top-N). Because a gold page is ranked only within its
own doc, we retrieve ONLY docs that contain a gold page — under isolation this
yields byte-identical gold ranks while skipping wasted encodes on non-gold docs.

Usage
-----
    python code/chunk_rrf_debug.py --s1-results data/results/s1_eval_20260603_224526.jsonl
    python code/chunk_rrf_debug.py --s1-results ... --n 3                 # smoke test (4-way)
    python code/chunk_rrf_debug.py --s1-results ... --queries needed_info --no-colbert
    python code/chunk_rrf_debug.py --s1-results ... --chunk-caps 25 50 100
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

# Reused, unchanged, from the production pipeline — same scoring math.
from pipeline import (
    DocumentStore,
    sparse_sim,
    colbert_maxsim,
    rrf_fuse,
    SPLIT_CONFIG,
    RRF_K,
    EMBEDDING_MODEL,
    RESULTS_DIR,
    TOP_K_CHUNKS,
)

# Signal order mirrors retrieve_pages' RRF input [dense, bm25, sparse, colbert].
# (RRF is order-independent for the score, but we keep it identical anyway.)
SIGNALS = ["dense", "bm25", "sparse", "colbert"]
RECALL_KS = [1, 5, 10, 20]
DEFAULT_CHUNK_CAPS = [25, 50]
DEFAULT_TOP_PAGES = 20
DEFAULT_QUERIES = ["semantic_query", "needed_info"]
BASELINE = "baseline"

_SEP = "─" * 96


# ── Query forms ─────────────────────────────────────────────────────────────
# Held CONSTANT per comparison section so only chunk-vs-page varies. semantic_query
# falls back to the raw question (matches production: target.get("semantic_query",
# question)); needed_info has no sensible fallback, so a target without it is
# skipped for that form. combined concatenates both into ONE query.

def query_text_for(form, target, question):
    if form == "semantic_query":
        return (target.get("semantic_query") or question or "").strip()
    if form == "needed_info":
        return (target.get("needed_info") or "").strip()
    if form == "combined":
        # needed_info (precise NL spec) + semantic_query (keyword bag) as ONE
        # string — the only production-compatible combination, since retrieve_pages
        # feeds all four signals the SAME query. The idea: lexical signals (BM25/
        # sparse) get the synonyms/keywords, dense/ColBERT get the NL spec + periods.
        ni = (target.get("needed_info") or "").strip()
        sq = (target.get("semantic_query") or "").strip()
        return (ni + " " + sq).strip()
    raise ValueError(f"unknown query form: {form}")


# ── Encode + per-chunk scoring (single encode feeds BOTH fusions) ────────────

def encode_query(embed_model, query, use_colbert):
    """Encode one query exactly like pipeline.retrieve_pages."""
    enc = embed_model.encode(
        [query],
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
    return query_dense, query_sparse, query_colbert


def score_chunks(doc_store, query_dense, query_sparse, query_colbert,
                 query_text, resolved_docs, use_colbert):
    """Per-chunk 4-signal scores across docs. Returns (chunks, used_docs) where
    each chunk is a dict with key (doc, idx), its page, and the four raw scores.
    Identical per-chunk math to retrieve_pages — nothing is reduced here."""
    chunks = []
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
        bm25_scores = bm25.get_scores(query_text.lower().split())
        sparse_scores = np.array([sparse_sim(query_sparse, s) for s in all_sparse])
        if use_colbert and all_colbert is not None:
            colbert_scores = np.array(
                [colbert_maxsim(query_colbert, c) for c in all_colbert]
            )
        else:
            colbert_scores = np.zeros(len(meta))

        for idx in range(len(meta)):
            chunks.append({
                "key": (doc_name, idx),
                "doc": doc_name,
                "idx": idx,
                "page": int(meta[idx].get("page", 0)),
                "dense": float(dense_scores[idx]),
                "bm25": float(bm25_scores[idx]),
                "sparse": float(sparse_scores[idx]),
                "colbert": float(colbert_scores[idx]),
            })
    return chunks, used_docs


# ── Baseline: page-level RRF (byte-for-byte pipeline.retrieve_pages) ─────────

def page_level_rrf(chunks, use_colbert):
    """Reduce chunks to best score per (doc, page) per signal, rank pages per
    signal, RRF-fuse. Returns {(doc, page): rank} (1-indexed). This is the exact
    production baseline."""
    page_dense, page_bm25, page_sparse, page_colbert = {}, {}, {}, {}
    for c in chunks:
        key = (c["doc"], c["page"])
        if key not in page_dense or c["dense"] > page_dense[key]:
            page_dense[key] = c["dense"]
        if key not in page_bm25 or c["bm25"] > page_bm25[key]:
            page_bm25[key] = c["bm25"]
        if key not in page_sparse or c["sparse"] > page_sparse[key]:
            page_sparse[key] = c["sparse"]
        if key not in page_colbert or c["colbert"] > page_colbert[key]:
            page_colbert[key] = c["colbert"]

    if not page_dense:
        return {}

    dense_ranked = sorted(page_dense, key=page_dense.get, reverse=True)
    bm25_ranked = sorted(page_bm25, key=page_bm25.get, reverse=True)
    sparse_ranked = sorted(page_sparse, key=page_sparse.get, reverse=True)
    rankings = [dense_ranked, bm25_ranked, sparse_ranked]
    if use_colbert:
        colbert_ranked = sorted(page_colbert, key=page_colbert.get, reverse=True)
        rankings.append(colbert_ranked)

    rrf_ranked, _ = rrf_fuse(rankings, k=RRF_K)
    return {key: i + 1 for i, key in enumerate(rrf_ranked)}


# ── Treatment: chunk-level RRF ───────────────────────────────────────────────

def chunk_level_rrf(chunks, use_colbert):
    """RRF over ALL chunks (no page reduction). Returns (chunk_ranked,
    chunk_scores): chunk_ranked is a list of chunk keys (doc, idx) by fused RRF
    score desc; chunk_scores maps each key -> its fused score."""
    if not chunks:
        return [], {}
    by_key = {c["key"]: c for c in chunks}
    rankings = [
        sorted(by_key, key=lambda k: by_key[k]["dense"], reverse=True),
        sorted(by_key, key=lambda k: by_key[k]["bm25"], reverse=True),
        sorted(by_key, key=lambda k: by_key[k]["sparse"], reverse=True),
    ]
    if use_colbert:
        rankings.append(sorted(by_key, key=lambda k: by_key[k]["colbert"], reverse=True))
    return rrf_fuse(rankings, k=RRF_K)


# ── Chunk -> page mapping variants ───────────────────────────────────────────

def pages_firstapp(chunk_ranked, page_of, cap=None):
    """Variant (a)/(b): walk the chunk ranking, rank each page at its FIRST (=
    highest-scoring) appearance. cap=N limits to the top-N chunks (variant a):
    pages whose chunks are all below rank N never appear. Returns {(doc,page):
    rank}."""
    seq = chunk_ranked if cap is None else chunk_ranked[:cap]
    page_rank, r = {}, 0
    for ck in seq:
        pg = page_of[ck]
        if pg not in page_rank:
            r += 1
            page_rank[pg] = r
    return page_rank


def pages_by_score(chunk_ranked, chunk_scores, page_of, agg, topm=None):
    """Variant (c): aggregate chunk RRF scores to a page score, rank pages desc.
    agg='max' (best chunk), 'sum' (all chunks), or 'topm_sum' (top-m chunks per
    page). Returns {(doc,page): rank}."""
    if agg == "max":
        page_score = {}
        for ck in chunk_ranked:
            pg, s = page_of[ck], chunk_scores[ck]
            if pg not in page_score or s > page_score[pg]:
                page_score[pg] = s
    elif agg == "sum":
        page_score = defaultdict(float)
        for ck in chunk_ranked:
            page_score[page_of[ck]] += chunk_scores[ck]
    elif agg == "topm_sum":
        # chunk_ranked is score-desc, so a page's first m chunks are its top m.
        seen, page_score = defaultdict(int), defaultdict(float)
        for ck in chunk_ranked:
            pg = page_of[ck]
            if seen[pg] < topm:
                page_score[pg] += chunk_scores[ck]
                seen[pg] += 1
    else:
        raise ValueError(f"unknown agg: {agg}")
    ranked = sorted(page_score, key=page_score.get, reverse=True)
    return {pg: i + 1 for i, pg in enumerate(ranked)}


def chunk_variant_names(chunk_caps, topm_sum=None):
    """Ordered list of chunk-variant method names for the given config."""
    names = ["firstapp"] + [f"cap{c}" for c in chunk_caps] + ["maxscore", "sumscore"]
    if topm_sum:
        names.append(f"top{topm_sum}sum")
    return names


def build_chunk_variants(chunk_ranked, chunk_scores, chunks, chunk_caps, topm_sum=None):
    """Derive every chunk->page variant ranking from one chunk ranking."""
    page_of = {c["key"]: (c["doc"], c["page"]) for c in chunks}
    variants = {"firstapp": pages_firstapp(chunk_ranked, page_of, cap=None)}
    for cap in chunk_caps:
        variants[f"cap{cap}"] = pages_firstapp(chunk_ranked, page_of, cap=cap)
    variants["maxscore"] = pages_by_score(chunk_ranked, chunk_scores, page_of, "max")
    variants["sumscore"] = pages_by_score(chunk_ranked, chunk_scores, page_of, "sum")
    if topm_sum:
        variants[f"top{topm_sum}sum"] = pages_by_score(
            chunk_ranked, chunk_scores, page_of, "topm_sum", topm=topm_sum)
    return variants


# ── One retrieval unit (single doc): baseline + all chunk variants ──────────

def retrieve_unit(doc_store, embed_model, query_text, resolved_docs, use_colbert,
                  chunk_caps, topm_sum):
    """Encode once, score chunks once, return method->{(doc,page): rank} for the
    baseline + every chunk variant, the set of reachable pages, and meta."""
    qd, qs, qc = encode_query(embed_model, query_text, use_colbert)
    chunks, used_docs = score_chunks(
        doc_store, qd, qs, qc, query_text, resolved_docs, use_colbert)
    if not chunks:
        return {}, set(), {"n_pages": 0, "n_chunks": 0, "docs": used_docs}

    methods = {BASELINE: page_level_rrf(chunks, use_colbert)}
    chunk_ranked, chunk_scores = chunk_level_rrf(chunks, use_colbert)
    methods.update(build_chunk_variants(
        chunk_ranked, chunk_scores, chunks, chunk_caps, topm_sum))

    reachable_pages = {(c["doc"], c["page"]) for c in chunks}
    meta = {"n_pages": len(reachable_pages), "n_chunks": len(chunks), "docs": used_docs}
    return methods, reachable_pages, meta


# ── Loading (mirrors retrieval_debug.py) ─────────────────────────────────────

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


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_qform(gold_pages, methods, ks):
    """Per method: recall@k + median/mean rank over REACHABLE gold pages.

    gold_pages: list of per-gold-page dicts (one per distinct (qid,doc,page) for
    this query form). Recall denominator = reachable pages (>=1 chunk), shared by
    every method. A page with no finite rank under a method (cap dropped it)
    counts as a miss at all k but does not change the denominator."""
    reachable = [g for g in gold_pages if g["reachable"]]
    n = len(reachable)
    out = {}
    for m in methods:
        finite = [g["ranks"][m] for g in reachable if g["ranks"].get(m) is not None]
        rec = {k: (sum(1 for r in finite if r <= k) / n if n else 0.0) for k in ks}
        out[m] = {
            "recall": {k: round(rec[k], 4) for k in ks},
            "median_rank": float(np.median(finite)) if finite else None,
            "mean_rank": round(float(np.mean(finite)), 2) if finite else None,
            "n_appeared": len(finite),
            "coverage": round(len(finite) / n, 4) if n else 0.0,
        }
    return out, n, reachable


def calibration_firstapp_vs_max(reachable):
    """Fraction of reachable gold pages where firstapp_rank == maxscore_rank.
    Should be ~1.0 — the proof that first-appearance == max aggregation."""
    both = [g for g in reachable
            if g["ranks"].get("firstapp") is not None
            and g["ranks"].get("maxscore") is not None]
    if not both:
        return None
    same = sum(1 for g in both if g["ranks"]["firstapp"] == g["ranks"]["maxscore"])
    return round(same / len(both), 4)


# ── Console rendering ────────────────────────────────────────────────────────

def print_comparison(qform, agg, n_reach, counts, methods, ks, calib):
    print(f"\n{_SEP}")
    print(f"  QUERY FORM = \"{qform}\"   "
          f"(reachable gold pages = {n_reach}; in-doc {counts['in_doc']}, "
          f"no-chunk {counts['no_chunk']}, doc-not-retrieved {counts['not_retr']})")
    print(f"{_SEP}")
    head = (f"  {'method':<12}" + "".join(f"{'r@' + str(k):>8}" for k in ks)
            + f"{'median':>8}{'mean':>8}{'cov':>7}   vs baseline (r@10)")
    print(head)
    print(f"  {'-' * (len(head) - 2)}")
    base_r10 = agg[BASELINE]["recall"][10]
    for m in methods:
        a = agg[m]
        cells = "".join(f"{a['recall'][k]:>8.3f}" for k in ks)
        med = f"{a['median_rank']:>8.1f}" if a["median_rank"] is not None else f"{'-':>8}"
        mean = f"{a['mean_rank']:>8.1f}" if a["mean_rank"] is not None else f"{'-':>8}"
        cov = f"{a['coverage']:>7.2f}"
        if m == BASELINE:
            tag = ""
        else:
            d = a["recall"][10] - base_r10
            tag = f"   {'+' if d >= 0 else ''}{d:.3f}"
        print(f"  {m:<12}{cells}{med}{mean}{cov}{tag}")
    if calib is not None:
        print(f"\n  calibration: firstapp == maxscore on {calib:.1%} of gold "
              f"pages (expect ~100%: first-appearance IS max aggregation).")


def print_cross_query(per_qform_agg, qforms, methods, ks):
    """Is the chunk effect additive on top of the better query? Compare every
    method's recall@10 across query forms, and the chunk-effect (best chunk
    method - baseline) under each form."""
    if len(qforms) < 2:
        return
    print(f"\n{_SEP}")
    print(f"  CROSS-QUERY SUMMARY  (recall@10 by query form)")
    print(f"{_SEP}")
    head = f"  {'method':<12}" + "".join(f"{q:>18}" for q in qforms)
    print(head)
    print(f"  {'-' * (len(head) - 2)}")
    for m in methods:
        cells = "".join(
            f"{per_qform_agg[q][m]['recall'][10]:>18.3f}" for q in qforms)
        print(f"  {m:<12}{cells}")

    chunk_methods = [m for m in methods if m != BASELINE]
    print(f"\n  chunk-effect = (best chunk variant r@10) - (baseline r@10):")
    effects = {}
    for q in qforms:
        base = per_qform_agg[q][BASELINE]["recall"][10]
        best_m = max(chunk_methods, key=lambda m: per_qform_agg[q][m]["recall"][10])
        best = per_qform_agg[q][best_m]["recall"][10]
        effects[q] = best - base
        print(f"    {q:<16} {'+' if effects[q] >= 0 else ''}{effects[q]:.3f}"
              f"  (best = {best_m})")
    vals = list(effects.values())
    if all(v > 0.0 for v in vals):
        spread = max(vals) - min(vals)
        verdict = ("ADDITIVE — the chunk gain holds under both query forms"
                   if spread <= 0.02 else
                   "PARTLY additive — chunk gain differs by query form")
    elif all(v <= 0.0 for v in vals):
        verdict = "NO chunk gain under either query form"
    else:
        verdict = "MIXED — chunk gain only under some query forms"
    print(f"\n  -> {verdict}.")


# ── Output files ─────────────────────────────────────────────────────────────

def write_outputs(out_dir, prefix, ts, gold_rows, per_qform_agg, qform_counts,
                  methods, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{prefix}_{ts}"

    # Per-gold-page CSV: one row per (query_form, distinct gold page) with the
    # gold page's rank under every method.
    rank_fields = [f"{m}_rank" for m in methods]
    gold_fields = (["query_form", "qid", "target_idx", "company", "doc_name",
                    "page", "in_retrieved_doc", "reachable", "n_pages",
                    "n_chunks"] + rank_fields)
    gold_path = out_dir / f"{base}_gold.csv"
    with open(gold_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gold_fields)
        w.writeheader()
        for row in gold_rows:
            flat = {k: row.get(k) for k in gold_fields if not k.endswith("_rank")}
            for m in methods:
                flat[f"{m}_rank"] = row["ranks"].get(m)
            w.writerow(flat)

    # Flat per-(query_form, method) aggregate CSV.
    method_fields = (["query_form", "method", "n_reachable", "n_appeared",
                      "coverage", "median_rank", "mean_rank"]
                     + [f"recall@{k}" for k in RECALL_KS])
    methods_path = out_dir / f"{base}_methods.csv"
    with open(methods_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=method_fields)
        w.writeheader()
        for qform, agg in per_qform_agg.items():
            n_reach = qform_counts[qform]["reachable"]
            for m in methods:
                a = agg[m]
                w.writerow({
                    "query_form": qform, "method": m, "n_reachable": n_reach,
                    "n_appeared": a["n_appeared"], "coverage": a["coverage"],
                    "median_rank": a["median_rank"], "mean_rank": a["mean_rank"],
                    **{f"recall@{k}": a["recall"][k] for k in RECALL_KS},
                })

    # Summary JSON.
    summary = {"config": config, "methods": methods, "per_query_form": {}}
    for qform, agg in per_qform_agg.items():
        c = qform_counts[qform]
        base_r = agg[BASELINE]["recall"]
        summary["per_query_form"][qform] = {
            "n_reachable": c["reachable"],
            "n_in_retrieved_doc": c["in_doc"],
            "n_no_chunk": c["no_chunk"],
            "n_doc_not_retrieved": c["not_retr"],
            "calibration_firstapp_eq_maxscore": c["calibration"],
            "per_method": agg,
            "delta_recall_vs_baseline": {
                m: {f"recall@{k}": round(agg[m]["recall"][k] - base_r[k], 4)
                    for k in RECALL_KS}
                for m in methods if m != BASELINE
            },
        }
    summary_path = out_dir / f"{base}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return gold_path, methods_path, summary_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chunk-level vs page-level RRF retrieval experiment (no LLM).")
    parser.add_argument("--s1-results", required=True, help="Path to S1 eval JSONL")
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="TRAIN only — test is off-limits (see CLAUDE.md)")
    parser.add_argument("--n", type=int, default=None, help="Number of questions")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--qids", nargs="+", help="Specific question IDs")
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES,
                        choices=["semantic_query", "needed_info", "combined"],
                        help="Query forms to compare (default: both). 'combined' = "
                             "needed_info + semantic_query as one query")
    parser.add_argument("--chunk-caps", nargs="+", type=int, default=DEFAULT_CHUNK_CAPS,
                        help="Top-N chunk caps for the first-appearance variant")
    parser.add_argument("--topm-sum", type=int, default=None,
                        help="Optional extra variant: sum of top-m chunk scores per page")
    parser.add_argument("--no-colbert", action="store_true",
                        help="Drop ColBERT -> 3-way RRF (saves memory)")
    parser.add_argument("--top-pages", type=int, default=DEFAULT_TOP_PAGES,
                        help="Per-unit gold-rank detail depth printed to console")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument("--prefix", default="chunk_rrf_debug")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    try:  # Windows consoles: make the box/Delta glyphs printable.
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.split == "test":
        print("REFUSING: the test split is off-limits (see CLAUDE.md). "
              "Run train only.")
        sys.exit(2)

    use_colbert = not args.no_colbert
    split = SPLIT_CONFIG[args.split]
    methods = [BASELINE] + chunk_variant_names(args.chunk_caps, args.topm_sum)

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
    print(f"Processing {len(common)} questions  (colbert={'ON' if use_colbert else 'OFF'}, "
          f"queries={args.queries}, caps={args.chunk_caps}, methods={methods})")

    print(f"Loading embedding model: {args.embedding_model}")
    embed_model = BGEM3FlagModel(args.embedding_model, use_fp16=True)

    doc_store = DocumentStore(
        split["vector_db"], split["merged_dir"],
        max_cached=8, use_colbert=use_colbert,
    )
    print(f"  {len(doc_store.available_docs)} docs available in vector_db")

    # gold_data[qform][(qid, doc, page)] -> per-gold-page record (deduped, best rank)
    gold_data = {q: {} for q in args.queries}

    for i, qid in enumerate(common, 1):
        s1 = s1_records[qid]
        qa = qa_data[qid]
        question = qa["question"]
        plan = s1["plan"]
        reslog = s1.get("resolution_log", [])
        evidences = qa.get("evidences", [])
        print(f"  unit-scan {i}/{len(common)}  {qid}            ", end="\r")

        gold_docs = {ev["doc_name"] for ev in evidences}
        targets = plan.get("targets", [])

        # Docs S1 resolved (and available) across ALL targets -> in_retrieved_doc.
        retrieved_docs_all = set()
        for t_idx, target in enumerate(targets, 1):
            res = reslog[t_idx - 1] if t_idx - 1 < len(reslog) else {}
            resolved = res.get("docs", res.get("matched", []))
            retrieved_docs_all.update(
                d for d in resolved if d in doc_store.available_docs)

        # One retrieval unit per (target, gold-containing doc). Under per-doc
        # isolation only gold docs affect gold ranks, so non-gold docs are skipped.
        for t_idx, target in enumerate(targets, 1):
            res = reslog[t_idx - 1] if t_idx - 1 < len(reslog) else {}
            resolved = res.get("docs", res.get("matched", []))
            avail = [d for d in resolved if d in doc_store.available_docs]
            company = target.get("company", "?")

            for doc_name in dict.fromkeys(avail):
                if doc_name not in gold_docs:
                    continue
                doc_gold = sorted({ev["page_num"] for ev in evidences
                                   if ev["doc_name"] == doc_name})
                for qform in args.queries:
                    qtext = query_text_for(qform, target, question)
                    if not qtext:
                        continue  # e.g. target without needed_info
                    methods_rank, reachable_pages, meta = retrieve_unit(
                        doc_store, embed_model, qtext, [doc_name],
                        use_colbert, args.chunk_caps, args.topm_sum)
                    for pg in doc_gold:
                        key = (doc_name, pg)
                        reachable = key in reachable_pages
                        gk = (qid, doc_name, pg)
                        store = gold_data[qform]
                        rec = store.get(gk)
                        if rec is None:
                            rec = {
                                "query_form": qform, "qid": qid, "target_idx": t_idx,
                                "company": company, "doc_name": doc_name, "page": pg,
                                "in_retrieved_doc": True, "reachable": reachable,
                                "n_pages": meta["n_pages"], "n_chunks": meta["n_chunks"],
                                "ranks": {},
                            }
                            store[gk] = rec
                        rec["reachable"] = rec["reachable"] or reachable
                        for m in methods:
                            rk = methods_rank.get(m, {}).get(key)
                            if rk is None:
                                continue
                            prev = rec["ranks"].get(m)
                            if prev is None or rk < prev:
                                rec["ranks"][m] = rk

        # Gold pages whose doc S1 never resolved (or not in vector_db).
        for ev in evidences:
            if ev["doc_name"] in retrieved_docs_all:
                continue
            for qform in args.queries:
                gk = (qid, ev["doc_name"], ev["page_num"])
                if gk in gold_data[qform]:
                    continue
                gold_data[qform][gk] = {
                    "query_form": qform, "qid": qid, "target_idx": None,
                    "company": None, "doc_name": ev["doc_name"],
                    "page": ev["page_num"], "in_retrieved_doc": False,
                    "reachable": False, "n_pages": 0, "n_chunks": 0, "ranks": {},
                }
    print()

    # Aggregate + render per query form.
    per_qform_agg = {}
    qform_counts = {}
    gold_rows = []
    for qform in args.queries:
        pages = list(gold_data[qform].values())
        gold_rows.extend(pages)
        agg, n_reach, reachable = aggregate_qform(pages, methods, RECALL_KS)
        calib = calibration_firstapp_vs_max(reachable)
        counts = {
            "reachable": n_reach,
            "in_doc": sum(1 for g in pages if g["in_retrieved_doc"]),
            "no_chunk": sum(1 for g in pages
                            if g["in_retrieved_doc"] and not g["reachable"]),
            "not_retr": sum(1 for g in pages if not g["in_retrieved_doc"]),
            "calibration": calib,
        }
        per_qform_agg[qform] = agg
        qform_counts[qform] = counts
        print_comparison(qform, agg, n_reach, counts, methods, RECALL_KS, calib)

    print_cross_query(per_qform_agg, args.queries, methods, RECALL_KS)

    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = {
            "split": args.split,
            "s1_results": args.s1_results,
            "embedding_model": args.embedding_model,
            "use_colbert": use_colbert,
            "queries": args.queries,
            "chunk_caps": args.chunk_caps,
            "topm_sum": args.topm_sum,
            "rrf_k": RRF_K,
            "recall_ks": RECALL_KS,
            "n_questions": len(common),
        }
        gold_path, methods_path, summary_path = write_outputs(
            args.out_dir, args.prefix, ts, gold_rows,
            per_qform_agg, qform_counts, methods, config)
        print(f"\n  Gold CSV (per gold page): {gold_path}")
        print(f"  Methods CSV (aggregate):  {methods_path}")
        print(f"  Summary JSON:             {summary_path}")


if __name__ == "__main__":
    main()
