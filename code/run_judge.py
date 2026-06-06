"""LLM Judge for SecQA — Stage 3.

Evaluates RAG pipeline results by comparing predicted answers against gold answers
using an LLM judge. Always runs on all results since answers are typically sentences.

Usage:
    python code/run_judge.py --results data/results/rag_20260604_181012_results.jsonl
    python run_judge.py --results ... --model qwen/qwq-32b
    python run_judge.py --results ... --concurrency 2
    python run_judge.py --results ... --n 10
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "qwen/qwq-32b"
CONCURRENCY = 1


# ── Judge Prompt ────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """\
You are a strict equivalence evaluator for financial QA over SEC filings.

You will receive a question, one or more gold (reference) answers, and the system's predicted answer.

Determine whether the predicted answer is equivalent to the gold answer(s).

## Equivalence rules (check in order):
1. **Exact string** — after stripping whitespace and normalizing case, the answers match.
2. **Numeric equivalence** — both contain numbers that parse to the same value within ≤1% \
relative error.
3. **Percent scale** — one answer uses decimal form (e.g. 0.1681), the other uses percent \
form (e.g. 16.81%). These are equivalent.
4. **Unit scale** — one uses thousands/millions/billions while the other does not \
(e.g. 19500000 vs 19.5 million). Equivalent if magnitude matches within 1%.
5. **Semantic equivalence** — the answers convey the same facts and numbers but use \
different wording, order, or formatting. The key information must match.
6. **Partial correctness** — the predicted answer contains the main fact or key number \
correctly but is missing details, has minor extras, or differs in non-essential wording.

## Verdict rules:
- **"correct"**: All key facts, numbers, and entities match the gold answer. \
Different wording or formatting is fine.
- **"partially_correct"**: The main answer or primary number is correct, but there are \
minor issues: missing secondary details, slight format differences in multi-part answers, \
or small extra information not in the gold answer.
- **"incorrect"**: Wrong numbers, wrong entity, fundamentally different answer, or \
critical information is missing.

## Output format — respond with ONLY a JSON object, no markdown fences, no extra text:
{"verdict": "correct", "reason": "one sentence explanation"}
or
{"verdict": "partially_correct", "reason": "one sentence explanation"}
or
{"verdict": "incorrect", "reason": "one sentence explanation"}\
"""

JUDGE_USER_TEMPLATE = """\
**Question:** {question}

**Gold answer(s):** {gold_answers}

**Predicted answer:** {predicted_answer}\
"""


# ── LLM Judge Call ──────────────────────────────────────────

def call_judge(question, gold_answers, predicted_answer, url, model,
               max_retries=3, timeout=120):
    """Call the LLM judge. Returns {"verdict": str, "reason": str}."""
    gold_str = " | ".join(str(a) for a in gold_answers if a)
    user_msg = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answers=gold_str,
        predicted_answer=predicted_answer,
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = ""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = (resp.json()["choices"][0]["message"]["content"] or "").strip()

            # Strip think tags
            raw = re.sub(r"<think>[\s\S]*?</think>\s*", "", raw).strip()
            raw = re.sub(r"<think>[\s\S]*$", "", raw).strip()

            # Strip markdown JSON fences
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
            cleaned = re.sub(r"\s*```$", "", cleaned)

            verdict = json.loads(cleaned)
            if verdict.get("verdict") in ("correct", "partially_correct", "incorrect"):
                return verdict

            # Invalid verdict — retry
            if attempt < max_retries - 1:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Invalid format. Respond with ONLY: "
                        '{"verdict": "correct"|"partially_correct"|"incorrect", '
                        '"reason": "..."}'
                    )},
                ]
                continue

        except (json.JSONDecodeError, KeyError):
            if attempt < max_retries - 1:
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Your response was not valid JSON. Respond with ONLY: "
                        '{"verdict": "correct"|"partially_correct"|"incorrect", '
                        '"reason": "..."}'
                    )},
                ]
                continue

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"verdict": "error", "reason": f"API error: {e}"}

    return {"verdict": "error", "reason": f"Max retries exceeded. Last: {raw[:200]}"}


# ── Row Processing ──────────────────────────────────────────

def judge_one(record, url, model, verbose=False, max_retries=3, timeout=120):
    """Judge a single result record."""
    question = record["question"]
    predicted = record.get("predicted_answer", "")
    gold = record.get("gold_answer", "")
    gold2 = record.get("gold_answer2")

    gold_answers = [gold]
    if gold2:
        gold_answers.append(gold2)

    verdict = call_judge(
        question=question,
        gold_answers=gold_answers,
        predicted_answer=predicted,
        url=url,
        model=model,
        max_retries=max_retries,
        timeout=timeout,
    )

    eval_correct = verdict["verdict"] == "correct"
    partially = verdict["verdict"] == "partially_correct"

    if verbose:
        tag = "OK" if eval_correct else ("~" if partially else "FAIL")
        print(f"  [{tag}] {record['qid']}: {verdict['verdict']} "
              f"— {verdict.get('reason', '')}")

    return {
        "qid": record["qid"],
        "question": question,
        "predicted_answer": predicted,
        "gold_answer": gold,
        "gold_answer2": gold2,
        "judge_verdict": verdict["verdict"],
        "judge_reason": verdict.get("reason", ""),
        "eval_correct": eval_correct,
        "eval_partially_correct": partially,
    }


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM Judge — Stage 3")
    parser.add_argument("--results", required=True, help="Path to rag_xxx_results.jsonl")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--url", default=LM_STUDIO_URL)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N records")
    parser.add_argument("--qids", nargs="+", help="Judge specific question IDs only")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    # Load results
    results_path = Path(args.results)
    print(f"Loading results from {results_path}")
    records = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter
    if args.qids:
        qid_set = set(args.qids)
        records = [r for r in records if r["qid"] in qid_set]
    if args.n is not None:
        records = records[:args.n]

    # Skip records with errors (no predicted answer)
    judgeable = [r for r in records if r.get("predicted_answer")
                 and r["predicted_answer"] != "INSUFFICIENT DATA"
                 and not r.get("error")]
    skipped = len(records) - len(judgeable)
    print(f"  {len(judgeable)} to judge, {skipped} skipped (errors/no answer)")

    # Judge
    judged_results = []
    error_count = 0

    if args.concurrency <= 1:
        for i, record in enumerate(judgeable, 1):
            print(f"  [{i}/{len(judgeable)}] {record['qid']}")
            result = judge_one(record, args.url, args.model, verbose)
            judged_results.append(result)
            if result["judge_verdict"] == "error":
                error_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(judge_one, r, args.url, args.model, verbose): r["qid"]
                for r in judgeable
            }
            for future in futures:
                result = future.result()
                judged_results.append(result)
                if result["judge_verdict"] == "error":
                    error_count += 1

    # Retry errors
    error_results = [r for r in judged_results if r["judge_verdict"] == "error"]
    if error_results:
        print(f"\n  Retrying {len(error_results)} errors (max_retries=5, timeout=180s)...")
        error_records = {r["qid"]: r for r in records}
        for er in error_results:
            orig = error_records.get(er["qid"])
            if orig:
                retry = judge_one(orig, args.url, args.model, verbose,
                                  max_retries=5, timeout=180)
                # Replace in results
                for i, jr in enumerate(judged_results):
                    if jr["qid"] == retry["qid"]:
                        judged_results[i] = retry
                        break
                if retry["judge_verdict"] != "error":
                    error_count -= 1

    # Compute metrics
    n_judged = len(judged_results)
    n_correct = sum(1 for r in judged_results if r["judge_verdict"] == "correct")
    n_partial = sum(1 for r in judged_results if r["judge_verdict"] == "partially_correct")
    n_incorrect = sum(1 for r in judged_results if r["judge_verdict"] == "incorrect")
    n_errors = sum(1 for r in judged_results if r["judge_verdict"] == "error")

    def rate(count):
        return round(count / n_judged, 4) if n_judged else 0

    summary = {
        "config": {
            "model": args.model,
            "url": args.url,
            "concurrency": args.concurrency,
            "results_file": str(results_path),
            "timestamp": datetime.now().isoformat(),
        },
        "n_total_records": len(records),
        "n_skipped": skipped,
        "n_judged": n_judged,
        "correct": {"count": n_correct, "rate": rate(n_correct)},
        "partially_correct": {"count": n_partial, "rate": rate(n_partial)},
        "incorrect": {"count": n_incorrect, "rate": rate(n_incorrect)},
        "judge_errors": {"count": n_errors, "rate": rate(n_errors)},
        "accuracy": rate(n_correct),
        "accuracy_with_partial": rate(n_correct + n_partial),
    }

    # Save
    # Derive output paths from results path
    stem = results_path.stem
    if stem.endswith("_results"):
        base = stem[:-len("_results")]
    else:
        base = stem

    judged_path = results_path.with_name(f"{base}_judged.jsonl")
    summary_path = results_path.with_name(f"{base}_judge_summary.json")

    with open(judged_path, "w", encoding="utf-8") as f:
        for r in judged_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"  JUDGE RESULTS  |  {n_judged} judged  |  model={args.model}")
    print(f"{'=' * 60}")
    print(f"  Correct:           {n_correct:>4}/{n_judged}  ({rate(n_correct)*100:5.1f}%)")
    print(f"  Partially correct: {n_partial:>4}/{n_judged}  ({rate(n_partial)*100:5.1f}%)")
    print(f"  Incorrect:         {n_incorrect:>4}/{n_judged}  ({rate(n_incorrect)*100:5.1f}%)")
    print(f"  Judge errors:      {n_errors:>4}/{n_judged}  ({rate(n_errors)*100:5.1f}%)")
    print(f"{'─' * 60}")
    print(f"  Accuracy:              {rate(n_correct)*100:5.1f}%")
    print(f"  Accuracy (w/ partial): {rate(n_correct + n_partial)*100:5.1f}%")
    print(f"{'=' * 60}")
    print(f"\n  Judged:  {judged_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
