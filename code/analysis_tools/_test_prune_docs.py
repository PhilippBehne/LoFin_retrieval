"""Offline self-test for --prune-docs (prune_target_docs in pipeline.py).
No LM Studio / no model load beyond importing pipeline. The replay part reads only
the local TRAIN S1 baseline (skipped if absent); never the test split.
Run: .venv\\Scripts\\python.exe code\\analysis_tools\\_test_prune_docs.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `code/` importable
from pipeline import prune_target_docs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
S1_BASELINE = PROJECT_ROOT / "data" / "results" / "s1_eval_20260603_224526.jsonl"


def _t(years, periods):
    return {"company": "X", "years": years, "periods": periods}


def unit_cases():
    # the one shape that prunes: FY-only, exactly two consecutive years -> newest 10-K
    assert prune_target_docs(_t([2023, 2024], ["FY"]),
                             ["X_2023_10K", "X_2024_10K"]) == ["X_2024_10K"]
    assert prune_target_docs(_t([2024, 2023], ["FY"]),
                             ["X_2023_10K", "X_2024_10K"]) == ["X_2024_10K"]  # year order

    # everything else stays untouched
    gap = ["X_2022_10K", "X_2024_10K"]
    assert prune_target_docs(_t([2022, 2024], ["FY"]), gap) == gap            # gap year (span 2)
    three = ["X_2022_10K", "X_2023_10K", "X_2024_10K"]
    assert prune_target_docs(_t([2022, 2023, 2024], ["FY"]), three) == three  # 3-year span
    mixed = ["X_2023Q1_10Q", "X_2023_10K", "X_2024Q1_10Q", "X_2024_10K"]
    assert prune_target_docs(_t([2023, 2024], ["FY", "Q1"]), mixed) == mixed  # mixed periods
    q_only = ["X_2023Q3_10Q", "X_2024Q3_10Q"]
    assert prune_target_docs(_t([2023, 2024], ["Q3"]), q_only) == q_only      # quarters only
    one = ["X_2024_10K"]
    assert prune_target_docs(_t([2024], ["FY"]), one) == one                  # single year
    assert prune_target_docs(_t([2024, 2024], ["FY"]), one) == one            # duplicate year
    assert prune_target_docs(_t([], ["FY"]), []) == []                        # empty target
    # safety net: newest-year 10-K missing from docs -> do NOT prune to nothing
    only_old = ["X_2023_10K"]
    assert prune_target_docs(_t([2023, 2024], ["FY"]), only_old) == only_old
    print("unit cases OK")


def replay_baseline():
    """Replay the prune over the stored TRAIN S1 plans and pin the measured effect:
    docs 441->420, avg recall 97.8%->97.1%, avg precision 84.3%->87.4%, and the
    only two questions whose S1 doc recall drops are openqa_206 / openqa_209
    (their gold cites both yearly 10-Ks; the values also sit in the newest one's
    comparative columns)."""
    if not S1_BASELINE.exists():
        print(f"SKIP replay: {S1_BASELINE} not found (local-only, gitignored)")
        return
    rows = [json.loads(l) for l in open(S1_BASELINE, encoding="utf-8")]
    rows = [o for o in rows if "plan" in o and not o.get("error")]
    assert len(rows) == 143, len(rows)

    def metrics(res, gt):
        if not gt:
            return 1.0, 1.0
        h = res & gt
        return len(h) / len(gt), (len(h) / len(res)) if res else 0.0

    stats = {}
    for prune_on in (False, True):
        sum_r = sum_p = 0.0
        n_docs = 0
        recall_by_qid = {}
        for o in rows:
            res = set()
            for target, entry in zip(o["plan"]["targets"], o["resolution_log"]):
                docs = entry.get("docs", [])
                if prune_on:
                    docs = prune_target_docs(target, docs)
                res.update(docs)
            r, p = metrics(res, set(o["gt_docs"]))
            recall_by_qid[o["qid"]] = r
            sum_r += r
            sum_p += p
            n_docs += len(res)
        stats[prune_on] = (sum_r / len(rows), sum_p / len(rows), n_docs, recall_by_qid)

    base_r, base_p, base_docs, base_rec = stats[False]
    pr_r, pr_p, pr_docs, pr_rec = stats[True]

    # OFF must reproduce the stored baseline S1 metrics exactly
    assert base_docs == 441, base_docs
    assert abs(base_r - 0.978) < 0.002 and abs(base_p - 0.843) < 0.002, (base_r, base_p)
    # ON: the measured train effect (2026-06-11, see CHANGELOG)
    assert pr_docs == 420, pr_docs
    assert abs(pr_r - 0.971) < 0.002, pr_r
    assert abs(pr_p - 0.874) < 0.002, pr_p
    losers = sorted(q for q in base_rec if pr_rec[q] < base_rec[q])
    assert losers == ["openqa_206", "openqa_209"], losers
    print(f"replay OK: docs {base_docs}->{pr_docs}, recall {base_r:.1%}->{pr_r:.1%}, "
          f"precision {base_p:.1%}->{pr_p:.1%}, recall losses only: {losers}")


def main():
    unit_cases()
    replay_baseline()
    print("prune_docs OK")


if __name__ == "__main__":
    main()
