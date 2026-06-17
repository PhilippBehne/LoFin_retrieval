
# STATUS — current run, ceiling & roadmap

Volatile project state for the SecQA RAG pipeline. CLAUDE.md holds the stable how-to-work guidance;
this file holds the moving parts (current baseline, what the levers are, what the ceiling is). Deep
analysis lives in `data/results/debug/analysis_history.md`; code-change history in `CHANGELOG.md`;
forward action list in `TODO.md`.

_Last updated: 2026-06-14._

## Frozen decisions

- **Retrieval/RRF is frozen** — the 4-way RRF fusion and the `needed_info` query form are settled and
  won't be tuned further. Generation is the confirmed bottleneck (retrieval was fine in 47/50 of the
  chronically-wrong questions).
- `needed_info` stays the **global** query — using `semantic_query` globally regresses currently-found pages.

## Current honest baseline

**`rag_20260613_213203` = 81/143 = 56.6 % strict** (train, distractor corpus, restrictive S1 plan
`s1_eval_20260613_173341.jsonl`, PoT on, TOP_PAGES=10).

This is the first *un-inflated* run. Earlier runs (`020413` 52.4 %, `013732` 60.1 %) were **inflated**:
the S1 planner resolved doc names that weren't in the corpus, so those S1 errors were silently skipped
for free. On 13.06. 44 such false-positive docs were added to the train corpus as real distractors
(172 → 216 docs). **Treat the old numbers and any comparison against them as obsolete.**

The 62 wrong questions split into **34 ceiling** (5 gold errors, 4 judge-noise, 23 metric-ambiguity,
2 transient crashes) + **28 addressable**.

## Roadmap, by net gain / risk

1. **H2 — surgical S1/`needed_info` fix** ← next lever. Turn the compute-phrased retrieval query back
   into a doc-near label (recovers `81`+`110` safely), add a surgical anchor-year rule (`127`) and a
   sum-over-N-years rule (`19`). **Guardrail:** explicitly forbid a broad "keep all years" rule (it
   pushes `272` into an n_ctx crash). Expected 84–86/143 ≈ 59–60 %, low risk, zero crash side effects.

   **Measured 2026-06-14 (run `213203`, train): the `needed_info` lever is RETRIEVAL-ONLY, not a
   double lever.** Wherever the gold page was actually retrieved, the needed operands were **always**
   in the extracted facts (0/75 wrong-but-retrieved cases dropped a value) — the extractor
   over-extracts (~30 facts/page) and ignores `needed_info` phrasing, so rewriting it does **nothing**
   for fact extraction. `81`/`110` are recovered purely by getting the *sibling* page (HON p.14, YUM
   p.72) into top-10; their `INSUFFICIENT` was a retrieval miss on that sibling, not an extraction
   failure. **Caveat:** gold hit@10 *overstates* the lever — many missed gold pages aren't needed
   (`272` answers correctly from comparative columns despite gold at rank 32/54), so the real lever ≈
   the few questions that go `INSUFFICIENT` because a *uniquely-sourced* page is missed.
2. **Risk-free add-ons:** `--fact-batch-size 3` (unblocks the chronic `openqa_28` n_ctx crash) +
   COMPUTED_VALUES_HINT forcing the answer to state the injected total (`267`). Each +0…+1.
3. **Judge-denominator fix** in `run_judge.py` (additive summary keys) — do this *before* the next
   measured run so runs stay comparable. Details in `TODO.md`.
4. **H1 — compute-prompt label-citation** (12 raw questions but ~0 measured conversion, high A/B risk)
   last and A/B-tested separately.

PoT was implemented 2026-06-06 (`--pot`, default ON).

## Re-analysis 2026-06-14 — corrected ceiling

45 wrong = 23 metric-ambiguity + 22 addressable, every one filing-verified
(`reanalysis_213203_SYNTHESIS.md` + `_metric_addressable.md`). The "23 metric-ambiguities" were
**over-counted**, and two findings matter more than any lever:

- **7 are actually GOLD ERRORS — the pipeline was already correct** (`73` Costco used annual not Q1; `133`
  CME "adj-EBITDA 107.9 %" — word "EBITDA" appears on **no** CME page; `156` FIS Merchant 4,651 fabricated
  (real 4,859); `189` LLY Q4-US 10,014 absent, our 9,032 = FY−9M; `199` gold swapped Micron Q1/Q2; `248`
  gold says "YUM reports no non-GAAP EPS" but YUM 10-K p.34/37 reports +6 %; `316` "QoQ" asked, gold gave
  YoY columns). These belong in the **ceiling**, not "addressable".
- **~13 are GENUIN** with our reading the more defensible one (`25` common-only vs incl. preferred; `71`
  cloud-infra narrow vs broad; `121`/`32` D/E = interest-bearing debt (textbook) vs total liabilities;
  `139`/`206`/`190`/`34`). Unwinnable without gold-overfit.
- **Only ~6 are genuinely winnable**, via small `pipeline.py` prompt tweaks: (H-A) **preserve the
  unit/scale** (`160`; value was within 0.66 % — only the missing "thousand" lost it); (H-B) **state the
  injected COMPUTED_VALUES total + don't emit INSUFFICIENT when the facts/computed value are present**
  (`267`,`144`,`126`); (H-C) **state endpoints not just the delta** for change questions (`155`).
  Caveat — borderline over-optimization: each lever nets **only 1–2 questions** (~+3–5 total ≈ within
  judge noise), so the gain is **not cleanly measurable**. H-A/B/C are general robustness rules (not
  gold-specific) so they're defensible as one bundled change; the convention levers (gross-margin-as-$
  `112`/`277`, CAGR-window `11`/`31`) can regress as many as they fix — **skip them**.

**Corrected ceiling: hard ~62–64 %, fair ~67–69 %** (prior "fair ~71 %" was optimistic — most of the
"ambiguity" bucket is gold-error or genuinely-ours, not generation-fixable). Net: the pipeline is
**closer to its realistic ceiling than 56.6 % suggests**; chasing the last points via prompt micro-tweaks
has poor ROI.

## Debug records (in `data/results/debug/`)

- `analysis_history.md` — consolidated analysis write-up (chronology + current potential map for `213203`).
- `fehleranalyse_213203_alle62.md` — all 62 wrong answers with disposition + per-ID mechanism.
- `reanalysis_213203_SYNTHESIS.md` (+ `_metric_addressable.md` per ID) — the filing-verified re-analysis
  of the 45 addressable/metric questions. `wrong_ids_213203.txt` is the raw 62-wrong dump.
