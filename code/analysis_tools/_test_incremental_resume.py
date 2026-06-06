"""Offline verification for incremental save + resume (no LM Studio / no model load
beyond the module import). Uses synthetic result dicts in a temp dir — touches no
real data and no test split. Run: .venv\\Scripts\\python.exe code\\_test_incremental_resume.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `code/` importable
import pipeline as P
import pipeline_resume as R

CONFIG = {
    "model": "m", "temperature": 0.1, "top_pages": 10, "fact_batch_size": 5,
    "split": "train", "use_colbert": True,
    "s1_results": "data/results/s1.jsonl", "embedding_model": "BAAI/bge-m3",
}


def make_result(qid, ok=True, with_ev=True):
    ev = [{"doc_name": "DOC_2020_10K", "page_num": 5}] if with_ev else []
    if not ok:
        return {
            "qid": qid, "question": f"Q {qid}", "predicted_answer": "",
            "gold_answer": "G", "gold_answer2": None, "evidences": ev,
            "s1_plan": {"targets": [{}]}, "n_targets": 0, "n_retrieval_units": 0,
            "n_fact_calls": 0, "retrieval_targets": [], "error": "Boom: x",
            "latency_s": 1.5, "timestamp": "2026-01-01T00:00:00",
        }
    rt = [{
        "company": "Co", "docs": ["DOC_2020_10K"],
        "pages": [{"rank": 1, "doc_name": "DOC_2020_10K", "page": 5, "score": 0.1, "is_gold": True}],
        "gold_hits": {"hit@1": True, "hit@5": True, "hit@10": True},
        "gold_ranks": [{"doc_name": "DOC_2020_10K", "page": 5, "rank": 1, "score": 0.1}],
    }] if with_ev else [{"company": "Co", "docs": ["DOC_2020_10K"], "pages": [],
                         "gold_hits": {}, "gold_ranks": []}]
    return {
        "qid": qid, "question": f"Q {qid}", "predicted_answer": "42",
        "gold_answer": "42", "gold_answer2": None, "evidences": ev,
        "s1_plan": {"targets": [{}]}, "n_targets": 1, "n_retrieval_units": 1, "n_fact_calls": 2,
        "retrieved_pages": [{"doc_name": "DOC_2020_10K", "page": 5},
                            {"doc_name": "DOC_2020_10K", "page": 6}],
        "targets_detail": [{"target": {}, "resolved_docs": ["DOC_2020_10K"],
                            "retrieved_pages": [], "fact_prompt_system": "sys",
                            "extracted_facts": "f", "fact_batches": []}],
        "retrieval_targets": rt,
        "answer_prompt_system": "asys", "answer_prompt_user": "auser",
        "answer_think": None, "answer_finish_reason": "stop",
        "total_prompt_tokens": 100, "total_completion_tokens": 50,
        "latency_s": 2.0, "timestamp": "2026-01-01T00:00:00",
    }


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pipresume_"))
    results = [make_result("openqa_1"),
               make_result("openqa_2", ok=False),
               make_result("openqa_3", with_ev=False)]
    ts = "20260101_000000"

    # T1 — incremental writer output == one-shot save_results output (byte identical)
    w = P.IncrementalWriter(tmp, ts, CONFIG)
    for r in results:
        w.write_result(r)
    w.finalize()
    bts = "20260101_111111"
    P.save_results(results, tmp, bts, CONFIG)
    for kind in ("results", "retrieval", "recheck"):
        a = (tmp / f"rag_{ts}_{kind}.jsonl").read_text(encoding="utf-8")
        b = (tmp / f"rag_{bts}_{kind}.jsonl").read_text(encoding="utf-8")
        assert a == b, f"T1 {kind} byte mismatch"
    sa = (tmp / f"rag_{ts}_summary.json").read_text(encoding="utf-8")
    sb = (tmp / f"rag_{bts}_summary.json").read_text(encoding="utf-8")
    assert sa == sb, "T1 summary byte mismatch"
    print("T1 byte-identity (incremental == batch) OK")

    # T2 — progress shim reproduces the summary exactly
    prog = P.read_jsonl_safe(tmp / f"rag_{ts}_progress.jsonl")
    assert len(prog) == 3
    shims = [P.progress_line_to_seed(p) for p in prog]
    assert P.compute_summary(shims, CONFIG) == P.compute_summary(results, CONFIG), \
        "T2 shim summary != full"
    print("T2 progress-shim reproduces summary OK")

    # T3 — load_progress classifies done vs errored
    succ, err, seed = R.load_progress(tmp, f"rag_{ts}")
    assert succ == {"openqa_1", "openqa_3"}, succ
    assert err == {"openqa_2"}, err
    print("T3 load_progress classification OK")

    # T4 — retry filter drops the to-be-reprocessed qid from all logs
    to_process = {"openqa_2"}
    for suf in ("_results.jsonl", "_retrieval.jsonl", "_recheck.jsonl", "_progress.jsonl"):
        R.filter_out_qids(tmp / f"rag_{ts}{suf}", to_process)
    after = [r["qid"] for r in P.read_jsonl_safe(tmp / f"rag_{ts}_results.jsonl")]
    assert after == ["openqa_1", "openqa_3"], after
    assert [p["qid"] for p in P.read_jsonl_safe(tmp / f"rag_{ts}_progress.jsonl")] == \
        ["openqa_1", "openqa_3"]
    print("T4 retry filter_out_qids OK")

    # T5 — resume append: re-process openqa_2 (now ok), each qid exactly once
    prior = [s for q, s in seed.items() if q not in to_process]
    w2 = P.IncrementalWriter(tmp, ts, CONFIG, prior_results=prior)
    w2.write_result(make_result("openqa_2"))  # succeeds this time
    w2.finalize()
    final = sorted(r["qid"] for r in P.read_jsonl_safe(tmp / f"rag_{ts}_results.jsonl"))
    assert final == ["openqa_1", "openqa_2", "openqa_3"], final
    assert len(final) == len(set(final)), "duplicate qid after resume"
    s = json.loads((tmp / f"rag_{ts}_summary.json").read_text(encoding="utf-8"))
    assert s["n_questions"] == 3 and s["n_completed"] == 3 and s["n_errors"] == 0, s
    print("T5 resume append, no dup, combined summary OK")

    # T6 — resolve_run accepts file path / prefix / bare ts
    for arg in (str(tmp / f"rag_{ts}_results.jsonl"), str(tmp / f"rag_{ts}_summary.json")):
        od, pref, got = R.resolve_run(arg)
        assert got == ts and pref == f"rag_{ts}" and od == tmp, (arg, od, pref, got)
    od, pref, got = R.resolve_run(f"rag_{ts}")
    assert got == ts and pref == f"rag_{ts}" and od == R.RESULTS_DIR
    od, pref, got = R.resolve_run(ts)
    assert got == ts and pref == f"rag_{ts}"
    print("T6 resolve_run variants OK")

    # T7 — read_jsonl_safe tolerates a truncated final line
    p = tmp / "trunc.jsonl"
    p.write_text('{"qid":"a"}\n{"qid":"b"}\n{"qid":"c"', encoding="utf-8")
    assert [g["qid"] for g in P.read_jsonl_safe(p)] == ["a", "b"]
    print("T7 truncated-tail tolerance OK")

    # T8 — fallback join (no sidecar present, e.g. batch-saved run)
    ts3 = "20260101_222222"
    P.save_results(results, tmp, ts3, CONFIG)  # writes no _progress.jsonl
    assert not (tmp / f"rag_{ts3}_progress.jsonl").exists()
    succ3, err3, seed3 = R.load_progress(tmp, f"rag_{ts3}")
    assert succ3 == {"openqa_1", "openqa_3"} and err3 == {"openqa_2"}, (succ3, err3)
    s_fb = P.compute_summary(list(seed3.values()), CONFIG)
    s_full = P.compute_summary(results, CONFIG)
    for key in ("n_questions", "n_completed", "n_errors", "total_prompt_tokens",
                "total_completion_tokens", "retrieval_hits"):
        assert s_fb[key] == s_full[key], (key, s_fb[key], s_full[key])
    print("T8 fallback-join reconstructs summary OK")

    # T9 — config drift: match passes, split mismatch aborts, soft key warns
    R.check_config_drift(tmp, f"rag_{ts}", CONFIG)  # no raise
    bad = dict(CONFIG, split="test")
    try:
        R.check_config_drift(tmp, f"rag_{ts}", bad)
        raise AssertionError("T9 expected SystemExit on split drift")
    except SystemExit:
        pass
    R.check_config_drift(tmp, f"rag_{ts}", dict(CONFIG, model="other"))  # warn only, no raise
    print("T9 config drift OK")

    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
