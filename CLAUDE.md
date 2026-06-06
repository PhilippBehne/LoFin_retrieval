# CLAUDE.md

Guidance for working in this repo. Financial QA over SEC filings (SecQA): a retrieval-augmented
generation (RAG) pipeline that answers questions about 10-K / 10-Q filings.

## ⛔ THE TEST SPLIT IS OFF-LIMITS — never touch it

**Claude must never use the test split for anything** — not reading it, running on it, sampling
examples from it, inspecting its contents, or tuning/choosing any parameter against it. **All**
development, analysis, debugging, and weight/parameter tuning happens on the **train split only**.
The test split is reserved for a single, final evaluation that the **user** runs deliberately;
any earlier contact leaks information and silently invalidates it (overfitting to test). If a task
appears to require test data, **stop and ask** — do not proceed.

Off-limits concretely:
- `data/qa/secqa_test_test.jsonl` — test questions + gold answers
- `data/src/test/` — test `vector_db/` + `_merged/` filings
- `--split test` on any script (`pipeline.py`, `eval_s1.py`, `retrieval_debug.py`, …)

**Naming trap:** in `secqa_test_*.jsonl` the *suffix* is the split. `secqa_test_train.jsonl` is the
**TRAIN** split (allowed); `secqa_test_test.jsonl` is the **TEST** split (forbidden). The leading
`test_` is part of the dataset name, not the split — do not let it fool you into reading the wrong file.

## Architecture — 3 stages, run in order

All three core scripts live in `code/` and talk to a **local LM Studio server** at
`http://localhost:1234/v1/chat/completions` (OpenAI-compatible). LM Studio must be running with the
right model loaded before any script will work.

1. **S1 — Query Planner** (`code/eval_s1.py`)
   LLM turns each SecQA question into a `plan` with one or more `targets`
   (`company`, `years`, `periods`, `semantic_query`, `needed_info`). Resolves tickers → doc names
   (`TICKER_YYYY[_Qn]_10K|10Q`) and scores the plan against ground truth.
   → writes `data/results/s1_eval_<ts>.jsonl`.

2. **S2 — RAG Pipeline** (`code/pipeline.py` + `code/pipeline_resume.py`)
   Consumes the S1 JSONL. For each target:
   - **Retrieve** top pages via **4-way RRF** (`retrieve_pages`): dense + BM25 + sparse/lexical +
     ColBERT, all from BGE-M3 except BM25 (rank_bm25). Scores are computed per chunk, then reduced to
     the best chunk score per `(doc, page)`, then fused with RRF (`RRF_K = 60`). `--no-colbert`
     drops it to 3-way (saves memory).
   - **Extract facts** (`extract_facts`, `think=False`): pages are sent in batches of
     `FACT_BATCH_SIZE` (5) → with `TOP_PAGES = 10` that's **2 fact calls per target**.
   - **Generate answer** (`generate_answer`, `think=True`): all extracted facts combined into one call.
   → writes `rag_<ts>_results.jsonl` (compact), `_retrieval.jsonl`, `_recheck.jsonl`
     (full prompts + per-target detail), `_summary.json`.
   
   **Resume capability**: `code/pipeline_resume.py` picks up interrupted runs by loading the last checkpoint
   from a prior `rag_<ts>_results.jsonl` and continuing from the next target.

3. **S3 — LLM Judge** (`code/run_judge.py`)
   Compares predicted vs. gold answers with a **separate** judge model.
   → writes `rag_<ts>_judged.jsonl`.

## Running the pipeline

```bash
# S1: plan questions (train split shown)
python code/eval_s1.py --n 20

# S2: retrieve + extract + answer (point --s1-results at the S1 output)
python code/pipeline.py --s1-results data/results/s1_eval_<ts>.jsonl --split train
#   useful flags: --n 10 | --qids openqa_183 openqa_124 | --no-colbert
#                 --model <id> | --top-pages 10 | --fact-batch-size 5

# S3: judge the answers
python code/run_judge.py --results data/results/rag_<ts>_results.jsonl --concurrency 2
```

Use the project venv: `.venv\Scripts\python.exe` (Windows).

## Code structure

**Core pipeline** (in `code/`):
- `eval_s1.py` — S1 query planner
- `pipeline.py` — S2 RAG (retrieve + extract + answer)
- `pipeline_resume.py` — resume interrupted S2 runs
- `run_judge.py` — S3 LLM judge

**Analysis & debugging tools** (in `code/analysis_tools/`):
- `_analyze_weighting.py` — explore retrieval method weights
- `_verify_analysis.py` — verify analysis results
- `_sample_gold.py` — sample gold answers for manual review
- `_test_incremental_resume.py` — test resume logic
- `retrieval_debug.py` — detailed retrieval analysis (recall, precision per method)
- `chunk_rrf_debug.py` — debug RRF fusion at chunk vs. page level

## LM Studio — the thing that bites

- **Models differ per stage.** S1 and S2 use Gemma; the judge uses Qwen. They are configured
  independently:
  - `code/pipeline.py:48`  → `LLM_MODEL = "google/gemma-4-27b-it"`
  - `code/eval_s1.py:31`   → `LLM_MODEL = "google/gemma-4-31b"`  ← currently **inconsistent** with S2
  - `code/run_judge.py:26` → `LLM_MODEL = "qwen/qwq-32b"`  ← **do not** change to Gemma; the judge is
    meant to be an independent model.
  The `model` string in the request must exactly match the identifier of the model loaded in LM
  Studio, or the call fails. `pipeline.py` and `eval_s1.py` also accept `--model` to override.
- **Context length (`n_ctx`) must be ~16k**, not the tighter values you might set to save VRAM.
  Measured on real runs (Gemma tokenizer): a 5-page fact call is ~7.3k input (p90 9.2k, dense pages
  up to 13k); the answer call's think block is ~4.7k output (p90 8k, max ~17k). Worst single call
  needs ~10.5k (p90) to ~17k. Below ~12k, inputs get truncated and outputs hit
  `finish_reason: "length"`. If `n_ctx` < ~20k, keep `LLM_MAX_TOKENS` (`pipeline.py:50`) around
  10000 so answer-input + max_tokens stays inside the window.
- **Think blocks**: fact extraction runs with `think=False` (tiny output); answer generation runs
  with `think=True`. `strip_think` removes `<think>...</think>` from stored content; `extract_think`
  keeps it separately in the `think` field. Both handle an unclosed/truncated think block.

## Data layout

- `data/qa/secqa_test_{train,test}.jsonl` — questions, gold `answer`(s), and `evidences` (gold pages).
- `data/src/{train,test}/vector_db/` — precomputed BGE-M3 dense/sparse/colbert vecs + BM25 per doc.
- `data/src/{train,test}/<ts>_merged/*.json` — merged parsed filings (parent page text).
- `data/results/` — S1/S2/S3 results:
  - Root: `s1_eval_<ts>.jsonl`, `rag_<ts>_results.jsonl`, `rag_<ts>_judged.jsonl`, `*_summary.json`
  - `debug/` — archived debug outputs: `retrieval_debug_*`, `chunk_rrf_*`, logs, analysis files

Splits are wired in `SPLIT_CONFIG` (`pipeline.py:34`); the `_merged` dir timestamps are hardcoded
there (train `20260603_002740`, test `20260603_043105`).

## Gotchas

- **Page indexing**: gold `evidences` are 0-indexed, the vector_db is 1-indexed. `pipeline.py:939`
  adds `+1` on load. Keep this in mind when comparing pages.
- **DocumentStore** caches up to 8 docs (`max_cached=8`); ColBERT vecs are the memory-heavy part.
- **No test suite** currently in the repo (the earlier `test_think.py` was removed).
- Result files predating the batch change store 10 pages in a single fact call (no `fact_batches`
  field); newer runs store per-batch `prompt_tokens` / `completion_tokens` / `finish_reason` / `think`
  under `targets_detail[].fact_batches`.
