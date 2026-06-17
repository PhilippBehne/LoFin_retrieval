# CLAUDE.md

Financial QA over SEC filings (SecQA): a retrieval-augmented generation (RAG) pipeline answering
questions about 10-K / 10-Q filings. The pipeline is essentially complete; work runs on the **train
split**.

Detail lives outside this file — read these only if a task actually needs them:
`STATUS.md` (baseline, ceiling, open levers), `data/results/debug/analysis_history.md` (full
error/analysis history), `CHANGELOG.md` (code changes), `TODO.md` (open actions).

## Pipeline — 3 stages, run in order

A local LM Studio server at `http://localhost:1234/v1/chat/completions` (OpenAI-compatible) must be
running with the right model loaded. Use the venv: `.venv\Scripts\python.exe` (Windows).

```bash
# S1 — query planner: question -> plan/targets (company, years, periods, semantic_query, needed_info);
#      resolves tickers -> doc names TICKER_YYYY[_Qn]_10K|10Q
python code/eval_s1.py --n 20                         # -> data/results/s1_eval_<ts>.jsonl

# S2 — RAG: retrieve (4-way RRF) + extract facts + PoT (compute_values -> safe_eval, AST-only, never
#      Python eval) + answer
python code/pipeline.py --s1-results data/results/s1_eval_<ts>.jsonl --split train
#   flags: --n | --qids | --no-colbert | --model | --top-pages 10 | --fact-batch-size 5
#          --no-pot (PoT on by default) | --prune-docs | --qids-file <txt>
#   -> rag_<ts>_results.jsonl (+ _retrieval / _recheck / _summary). Resume: code/pipeline_resume.py

# S3 — LLM judge (separate model): predicted vs gold
python code/run_judge.py --results data/results/rag_<ts>_results.jsonl --concurrency 2
```

Core scripts: `code/eval_s1.py`, `pipeline.py`, `pipeline_resume.py`, `run_judge.py`. Offline tests
(no model) in `code/analysis_tools/`: `_test_incremental_resume.py` (run with `PYTHONPATH=code`),
`_test_safe_eval.py`, `_test_prune_docs.py`. Compare two runs: `_diff_runs.py` (or the
`accuracy_end_to_end` summary key — the old `accuracy` uses a wandering judged-only denominator).

## LM Studio — the thing that bites

- **Models differ per stage** and the request `model` string must match the loaded model exactly:
  `pipeline.py:54` + `eval_s1.py:31` → `google/gemma-4-31b`; `run_judge.py:26` → `qwen/qwq-32b`.
  **Do not** point the judge at Gemma — it's meant to be an independent model. `--model` overrides S1/S2.
- **`n_ctx` must be ~16k.** Below ~12k, fact inputs truncate and answers hit `finish_reason: "length"`.
  If `n_ctx` < ~20k keep `LLM_MAX_TOKENS` (`pipeline.py:50`) ~10000.
- **Reasoning is per-step** via `reasoning_effort` (`REASONING_FACT/COMPUTE/ANSWER`, `pipeline.py:61`):
  fact extraction `none` (off), compute + answer `high` (on). The old `think` bool was a no-op — LM
  Studio ignored it (all earlier runs really ran "everything thinks"). On gemma-4-31b reasoning is
  binary (`none`=off, anything else=on; `off`/`on` raise HTTP 400). Reasoning text arrives in
  `message.reasoning_content` (not a `<think>` tag) and is logged in the `think`/`answer_think` fields;
  `strip_think`/`extract_think` remain only as a fallback for models that emit a literal `<think>` block.

## Data & gotchas

- `data/qa/secqa_test_train.jsonl` — questions + gold `answer`/`evidences`. `data/src/train/` — BGE-M3
  vecs + BM25 (`vector_db/`) and merged filings (`<ts>_merged/`). `data/results/` is gitignored.
  Splits wired in `SPLIT_CONFIG` (`pipeline.py:34`); `_merged` timestamps are hardcoded there.
- **Page indexing**: gold `evidences` are 0-indexed, the vector_db is 1-indexed — `pipeline.py:939`
  adds `+1` on load.
- `DocumentStore` caches 8 docs (`max_cached=8`); ColBERT vecs are the memory-heavy part.
- Result-file format: newer runs store per-batch `prompt_tokens`/`completion_tokens`/`reasoning_tokens`/
  `finish_reason`/`think` under `targets_detail[].fact_batches`; files predating the batch change put all
  pages in one fact call (no `fact_batches`). `reasoning_tokens` predates only the 2026-06-16 reasoning
  fix (older files lack it; expect 0 for fact batches now that `REASONING_FACT="none"`).
