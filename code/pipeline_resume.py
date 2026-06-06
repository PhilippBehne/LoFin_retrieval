"""RAG Pipeline Resume — continue an interrupted Stage 2 run.

Resumes a pipeline run that was interrupted mid-way: skips questions already
completed, re-processes questions that previously errored (default), and appends
new results to the SAME output files (`rag_<ts>_*`). The final summary covers the
old + new questions combined.

The run is identified by --resume pointing at any file of that run (e.g. its
`_results.jsonl`), the bare `rag_<ts>` prefix, or just `<ts>`. The run config is
supplied via the same flags as pipeline.py (manual); the run's `_summary.json` is
read only to WARN about config drift (a different --split / --s1-results aborts
unless --force).

Usage:
    python code/pipeline_resume.py \
        --resume data/results/rag_20260604_181012_results.jsonl \
        --split train --s1-results data/results/s1_eval_20260603_224526.jsonl
    # optional: --no-colbert  --model <id>  --n 20  --qids openqa_10  --no-retry-errors
"""

import argparse
import json
import os
import re
from pathlib import Path

from pipeline import (
    RESULTS_DIR,
    SPLIT_CONFIG,
    LLM_MODEL,
    TEMPERATURE,
    TOP_PAGES,
    FACT_BATCH_SIZE,
    EMBEDDING_MODEL,
    build_config_from_args,
    compute_common_qids,
    load_inputs,
    progress_line_to_seed,
    read_jsonl_safe,
    run_pipeline,
)

# File suffixes a single run can produce — stripped to recover the rag_<ts> prefix.
KNOWN_SUFFIXES = [
    "_results.jsonl", "_retrieval.jsonl", "_recheck.jsonl",
    "_progress.jsonl", "_summary.json",
    "_judged.jsonl", "_judge_summary.json",
]

TS_RE = re.compile(r"(\d{8}_\d{6})")


# ── Run resolution ──────────────────────────────────────────

def resolve_run(resume_arg):
    """Resolve --resume to (output_dir, prefix, timestamp).

    Accepts a path to any run file, a `rag_<ts>` prefix, or a bare `<ts>`.
    """
    p = Path(resume_arg)
    name = p.name
    parent = str(p.parent)
    output_dir = Path(parent) if parent not in ("", ".") else RESULTS_DIR

    for suf in KNOWN_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break

    m = TS_RE.search(name)
    if not m:
        raise SystemExit(
            f"Could not parse a run timestamp (YYYYMMDD_HHMMSS) from --resume '{resume_arg}'"
        )
    ts = m.group(1)
    return output_dir, f"rag_{ts}", ts


def check_config_drift(output_dir, prefix, config, force=False):
    """Compare the manually-passed config against the original run's _summary.json.

    split / s1_results mismatch aborts (unless --force) — they would corrupt the
    run. Other keys (model, temperature, …) only warn (mixed-config output)."""
    summary_path = output_dir / f"{prefix}_summary.json"
    if not summary_path.exists():
        print(f"  NOTE: {summary_path.name} not found — cannot check config drift")
        return
    try:
        stored = json.loads(summary_path.read_text(encoding="utf-8")).get("config", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"  NOTE: could not read {summary_path.name} for drift check: {e}")
        return

    # Critical: split must match exactly.
    if "split" in stored and stored["split"] != config.get("split"):
        msg = (f"  CONFIG DRIFT: --split={config.get('split')!r} differs from "
               f"original run ({stored['split']!r})")
        if force:
            print(msg + "  [--force: continuing anyway]")
        else:
            raise SystemExit(msg + "\n  Pass --force to override (NOT recommended).")

    # Critical: s1_results compared by filename (tolerates abs vs rel paths).
    if "s1_results" in stored:
        stored_name = Path(stored["s1_results"]).name
        cur_name = Path(config.get("s1_results", "")).name
        if stored_name != cur_name:
            msg = (f"  CONFIG DRIFT: --s1-results={cur_name!r} differs from "
                   f"original run ({stored_name!r})")
            if force:
                print(msg + "  [--force: continuing anyway]")
            else:
                raise SystemExit(msg + "\n  Pass --force to override (NOT recommended).")

    # Soft keys: warn only.
    for key in ("model", "temperature", "top_pages", "fact_batch_size",
                "use_colbert", "embedding_model"):
        if key in stored and stored[key] != config.get(key):
            flag = "--" + key.replace("_", "-")
            print(f"  WARNING: {flag}={config.get(key)!r} differs from original "
                  f"({stored[key]!r}) — output will mix configs")


# ── Progress / resume state ─────────────────────────────────

def load_progress(output_dir, prefix):
    """Return (success_qids, error_qids, seed_by_qid).

    Primary source: the `_progress.jsonl` sidecar. Fallback (runs saved without a
    sidecar, e.g. by the old one-shot save_results): reconstruct from the three
    JSONL logs joined by qid. seed_by_qid maps qid -> a minimal result dict that
    compute_summary() can consume (used to re-seed prior results)."""
    progress_path = output_dir / f"{prefix}_progress.jsonl"
    success, errored, seed = set(), set(), {}

    if progress_path.exists():
        for p in read_jsonl_safe(progress_path):
            qid = p["qid"]
            seed[qid] = progress_line_to_seed(p)
            (errored if p.get("error") else success).add(qid)
        return success, errored, seed

    return _load_progress_fallback(output_dir, prefix)


def _load_progress_fallback(output_dir, prefix):
    """Reconstruct resume state from the 3 logs (no sidecar present)."""
    print(f"  NOTE: no {prefix}_progress.jsonl — reconstructing resume state from logs")
    results = read_jsonl_safe(output_dir / f"{prefix}_results.jsonl")
    retrieval = {e["qid"]: e for e in read_jsonl_safe(output_dir / f"{prefix}_retrieval.jsonl")}
    recheck = {e["qid"]: e for e in read_jsonl_safe(output_dir / f"{prefix}_recheck.jsonl")}

    success, errored, seed = set(), set(), {}
    for rl in results:
        qid = rl["qid"]
        rt = retrieval.get(qid, {})
        rc = recheck.get(qid, {})
        gold_ranks = []
        for t in rt.get("targets", []):
            gold_ranks.extend(t.get("gold_ranks", []))
        seed[qid] = {
            "qid": qid,
            "error": rl.get("error"),
            "n_targets": rl.get("n_targets", 0),
            "n_retrieval_units": rl.get("n_retrieval_units", 0),
            "n_fact_calls": rl.get("n_fact_calls", 0),
            "retrieved_pages": [None] * len(rl.get("retrieved_pages", [])),
            "latency_s": rl.get("latency_s", 0),
            "total_prompt_tokens": rc.get("total_prompt_tokens", 0),
            "total_completion_tokens": rc.get("total_completion_tokens", 0),
            "evidences": rt.get("evidences", []),
            "retrieval_targets": [{"gold_ranks": gold_ranks}],
        }
        (errored if rl.get("error") else success).add(qid)
    return success, errored, seed


def filter_out_qids(path, drop_qids):
    """Atomically rewrite a JSONL log, dropping lines whose qid is in drop_qids.

    Keeps every qid present at most once: used at resume start to remove the stale
    lines of questions we are about to re-process (previously-errored ones), so the
    appended fresh lines don't create duplicate qids. Tolerates a truncated tail."""
    path = Path(path)
    if not path.exists() or not drop_qids:
        return
    kept = [obj for obj in read_jsonl_safe(path) if obj.get("qid") not in drop_qids]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Resume — continue an interrupted Stage 2 run"
    )
    parser.add_argument("--resume", required=True,
                        help="Any file of the run (e.g. ..._results.jsonl), the "
                             "rag_<ts> prefix, or <ts>")
    parser.add_argument("--s1-results", required=True,
                        help="Path to S1 eval JSONL (must match the original run)")
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--n", type=int, default=None,
                        help="Cap: max ADDITIONAL questions to process this run")
    parser.add_argument("--qids", nargs="+",
                        help="Restrict resume to a subset of the remaining qids")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-pages", type=int, default=TOP_PAGES)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--fact-batch-size", type=int, default=FACT_BATCH_SIZE,
                        help="Max pages per fact extraction LLM call")
    parser.add_argument("--no-colbert", action="store_true",
                        help="Skip ColBERT (saves memory)")
    parser.add_argument("--no-retry-errors", action="store_true",
                        help="Treat previously-errored questions as done "
                             "(default: re-process them)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if --split/--s1-results differ from the "
                             "original run's summary")
    args = parser.parse_args()

    output_dir, prefix, ts = resolve_run(args.resume)

    have_results = (output_dir / f"{prefix}_results.jsonl").exists()
    have_progress = (output_dir / f"{prefix}_progress.jsonl").exists()
    if not have_results and not have_progress:
        raise SystemExit(
            f"No resumable run found for prefix '{prefix}' in {output_dir} "
            f"(neither _results.jsonl nor _progress.jsonl present)"
        )

    config = build_config_from_args(args)
    check_config_drift(output_dir, prefix, config, force=args.force)

    s1_records, qa_data = load_inputs(args.s1_results, SPLIT_CONFIG[args.split])
    base = compute_common_qids(s1_records, qa_data)  # full split universe

    success_qids, error_qids, seed_by_qid = load_progress(output_dir, prefix)
    done_qids = set(success_qids) | set(error_qids)

    # Retry policy: by default errored qids are NOT considered done (re-processed).
    skip_qids = done_qids if args.no_retry_errors else set(success_qids)

    remaining = [q for q in base if q not in skip_qids]
    if args.qids:
        qset = set(args.qids)
        remaining = [q for q in remaining if q in qset]
    if args.n is not None:
        remaining = remaining[:args.n]
    to_process = set(remaining)

    # Keep the logs clean (each qid exactly once): drop the stale lines of any qid
    # we are about to re-process (only previously-errored ones can already be there),
    # then re-seed prior results for everything we are NOT re-processing.
    if to_process:
        for suf in ("_results.jsonl", "_retrieval.jsonl",
                    "_recheck.jsonl", "_progress.jsonl"):
            filter_out_qids(output_dir / f"{prefix}{suf}", to_process)

    prior_results = [seed for q, seed in seed_by_qid.items() if q not in to_process]

    retried = len(error_qids & to_process)
    print(f"\n{'=' * 80}")
    print(f"  RESUMING {prefix}  (output_dir: {output_dir})")
    print(f"  done={len(done_qids)} (ok={len(success_qids)}, err={len(error_qids)})  "
          f"base={len(base)}  to_process={len(remaining)}")
    if retried:
        print(f"  re-processing {retried} previously-errored question(s)")
    print(f"{'=' * 80}")

    run_pipeline(config, s1_records, qa_data, remaining,
                 output_dir=output_dir, timestamp=ts,
                 prior_results=prior_results, no_save=False)


if __name__ == "__main__":
    main()
