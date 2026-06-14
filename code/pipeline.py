"""RAG Pipeline for SecQA — Stage 2.

Multi-target fact extraction + answer generation pipeline.
Reads S1 query planner results, retrieves pages per target (4-way RRF),
extracts facts via LLM, then generates a final answer from combined facts.

Usage:
    python code/pipeline.py --s1-results data/results/s1_eval_20260613_173341.jsonl --split train
    python pipeline.py --s1-results data/results/s1_eval_xxx.jsonl --split test --n 10
    python pipeline.py --s1-results ... --qids openqa_183 openqa_124
    python pipeline.py --s1-results ... --no-colbert

    # PoT (Program-of-Thought) now runs by DEFAULT. Use --no-pot for the frozen baseline (A/B):
    python code/pipeline.py --s1-results data/results/s1_eval_20260603_224526.jsonl --split train --no-pot
"""

import argparse
import ast
import json
import operator
import os
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

# ── Config ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

SPLIT_CONFIG = {
    "train": {
        "qa_file": DATA_DIR / "qa" / "secqa_test_train.jsonl",
        "vector_db": DATA_DIR / "src" / "train" / "vector_db",
        "merged_dir": DATA_DIR / "src" / "train" / "20260603_002740_merged",
    },
    "test": {
        "qa_file": DATA_DIR / "qa" / "secqa_test_test.jsonl",
        "vector_db": DATA_DIR / "src" / "test" / "vector_db",
        "merged_dir": DATA_DIR / "src" / "test" / "20260603_043105_merged",
    },
}

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-31b"
TEMPERATURE = 0.1
LLM_MAX_TOKENS = 16000
TOP_PAGES = 10
TOP_K_CHUNKS = 25
RRF_K = 60
FACT_BATCH_SIZE = 5
EMBEDDING_MODEL = "BAAI/bge-m3"


# ── Prompts ─────────────────────────────────────────────────

FACT_EXTRACTION_PROMPT = """\
ROLE: You are a text scanner. You are NOT a reasoner.
Your job: read each page once and list every piece of information that could plausibly \
be relevant to the question — extract generously, duplicates are okay.

## Output format — exactly this, nothing else:

FACTS:
- quote: "<verbatim snippet from the page>" -> [Entity] metric | period | value unit | source: table/prose
- quote: "<verbatim snippet from the page>" -> [Entity] metric | period | value unit | source: table/prose

## Field rules:
- quote: copy a short verbatim snippet (5-20 words) from the page that contains the data. \
Must be an exact substring of the input.
- Entity: the company, segment, fund, or counterparty the data belongs to. \
Use the entity name as it appears in the document.
- metric: the line item, KPI, program name, or descriptive label \
(e.g. "Total revenue", "Net income", "Employee training programs").
- period: the fiscal year, quarter, or date the data refers to \
(e.g. "2020", "Q3 2019", "Dec 31 2020"). If not determinable, write "n/a".
- value unit: the numeric value with unit, or a descriptive value for qualitative facts \
(e.g. "12.4 billion USD", "START Program, Apprenticeships, Internships").
- source: "table" if from a structured table row/column, \
"prose" if from running text or a narrative sentence.

## Extraction rules:
- Extract EVERY number, metric, name, or fact that might be needed — it is better to include one \
too many than to miss one.
- For financial data: extract raw values, percentages, totals, subtotals, growth rates.
- For qualitative questions: extract program names, descriptions, lists, categories.
- When the same data is available both in a table and in prose, extract BOTH with their source type.
- For portion/percentage questions: you MUST also extract the relevant total or subtotal.
- For change/trend questions: extract values for ALL relevant periods.
- For multi-entity pages: label each fact with the correct entity — do not mix entities.
- Prefer raw table values over rounded numbers from prose text.

## FORBIDDEN in your output:
- "Wait", "Let me re-read", "Actually", "Hmm", "Let me check again"
- Discussion of table layout, OCR, column alignment
- Any "Self-Correction" section
- Any text before "FACTS:"
- Any explanation after the FACTS: block

## Example:
Question: What percentage of Fictional Bank's total assets were held as loans in 2020?
Context: "As of December 31, 2020, Fictional Bank reported total assets of $12.4 billion. \
The loan portfolio stood at $8.2 billion, while investment securities totaled $3.1 billion."

FACTS:
- quote: "total assets of $12.4 billion" -> [Fictional Bank] Total assets | 2020 | \
12.4 billion USD | source: prose
- quote: "loan portfolio stood at $8.2 billion" -> [Fictional Bank] Loan portfolio | 2020 | \
8.2 billion USD | source: prose"""


ANSWER_GENERATION_PROMPT = """\
You are a financial analyst answering questions about SEC filings (10-K and 10-Q).
Use ONLY the extracted facts provided below. Do NOT add information not present in the facts.

## Answer format — match the question type:
- Single numeric value → number only, no units (e.g. "133848")
- Multi-entity or multi-period comparison → "Label: Value" per line
- "Which company/quarter?" question → Entity name + value + one context sentence
- Complex multi-step question → Full sentence(s) with all relevant numbers
- Trend/time-series → "Year: Value" per line

## Calculation rules:
- Percentage change: ((new - old) / |old|) * 100
- CAGR: ((end / start) ^ (1/n) - 1) * 100
- Margin/share: (part / whole) * 100
- Q4 derivation: Q4 = full-year (10-K) minus nine-month cumulative (Q3 10-Q)
- Round to 2 decimal places unless the question specifies otherwise

## Important:
- If the extracted facts do not contain enough data to answer, respond with "INSUFFICIENT DATA"
- Keep monetary values in their original units unless conversion is needed for comparison
- Do NOT show your work or calculations — just give the final answer
- Be concise: match the style and detail level the question expects"""


# Appended to the answer system prompt ONLY when PoT supplies computed values, so the
# --pot-off path keeps the baseline system prompt byte-for-byte identical (clean A/B).
COMPUTED_VALUES_HINT = """

## Computed values (block below) — rules:
- Use these exact numbers; do NOT recompute them.
- Answer in the FORM the question asks for: if it asks for a percentage, share, ratio, \
growth rate, or contribution, the computed percentage/ratio IS the answer. Never convert \
it back into absolute amounts and never list the raw operands instead of the computed \
result. Include the % sign or unit with every value.
- Multi-entity / multi-period questions: report EVERY asked entity and EVERY asked period \
as its own labeled value, even if values repeat. Do not merge entities or periods into \
one number unless the question explicitly asks for a single combined figure. Do not add \
values, periods, or totals that were not asked for.
- BUT if the question DOES ask for a total / sum / combined / cumulative figure (over years, \
quarters, or entities), you MUST state the injected total explicitly as its own labeled value, \
in addition to any per-period values asked for. Omitting the requested total is a wrong answer.
- "How did X change" / trend questions: state BOTH endpoint values (e.g. "from 5.2% in 2021 to \
2.5% in 2024"), not only the delta between them.
- Do NOT answer "INSUFFICIENT DATA" when computed values are supplied below or the requested \
metric's values are present in the facts: give the best supported answer from them, even if \
some periods or entities are missing.
- "How much higher/lower" / "percentage difference" comparisons: state the magnitude as \
a positive number with a direction word (e.g. "4.74% lower"), not as a signed number."""


COMPUTE_PROMPT = """\
ROLE: You set up arithmetic. You do NOT execute it and you do NOT write prose.

Read the question and the extracted facts. Identify every numeric quantity the final answer must \
COMPUTE — percentage change, growth, CAGR, margin, ratio, difference, sum. For each, output a \
named arithmetic expression a calculator can evaluate.

## Output — strict JSON object, nothing else:
{"snake_case_name": "<arithmetic expression>", ...}

Rules for each expression:
- ONLY numbers and the operators + - * / ** ( ) and abs(). No variable names, no words, no units, \
no % signs, no commas inside numbers (write 3846 not 3,846).
- Substitute the actual numbers from the facts directly into the expression.
- Do NOT round inside the expression — full precision is kept by the calculator.
- One entry per quantity the answer needs. If the answer needs NO calculation (every value can be \
read off directly), output exactly {}.

## Standard formulas:
- percentage change: (new - old) / abs(old) * 100
- percentage difference / how much higher or lower A is than B: (A - B) / abs(B) * 100, \
where A = the entity or value named FIRST in the question (the sign encodes the direction)
- CAGR over n years:  (end / start) ** (1/n) * 100 - 100
- margin / share:     part / whole * 100
- debt-to-equity:     total liabilities / total equity
- difference:         a - b

## Choosing the right numbers (most errors are HERE, not in the arithmetic):
- PERIOD: use exactly the periods the question names. "2024 vs 2022" -> use 2024 and 2022, \
not an adjacent year.
- SCOPE: prefer the consolidated / company-wide TOTAL line item. Use a segment or sub-total \
ONLY when the question explicitly names that segment.
- DISAMBIGUATION: when several facts share a metric name but differ in value, pick the one whose \
label and scope match the question most directly (e.g. company-wide "Total liabilities", \
not a partial subtotal).
- LABEL MATCH: use the line item whose wording matches the question; for a named segment use that \
segment's reported total, not one sub-line.
- This is selection guidance, NOT a restriction: still use any fact you need — just pick the \
RIGHT one when several compete.

## Example
Question: By what percent did revenue grow from 7575 (2022) to 8000 (2023)?
Facts: ... [Co] revenue | 2022 | 7575 ... [Co] revenue | 2023 | 8000 ...
Output: {"revenue_pct_change_2022_2023": "(8000 - 7575) / abs(7575) * 100"}"""


# ── Document Store ──────────────────────────────────────────

class DocumentStore:
    """Loads vector DB + parent pages on demand with LRU eviction."""

    def __init__(self, vector_db_dir, pages_dir, max_cached=5, use_colbert=True):
        self._vector_db_dir = Path(vector_db_dir)
        self._pages_dir = Path(pages_dir)
        self._max_cached = max_cached
        self._use_colbert = use_colbert
        self._cache_order = []
        self._metadata = {}
        self._dense_vecs = {}
        self._colbert_vecs = {}
        self._sparse_vecs = {}
        self._bm25 = {}
        self._parent_pages = {}
        self._available_docs = {
            fp.stem.replace("_meta", "")
            for fp in self._vector_db_dir.glob("*_meta.json")
        }

    @property
    def available_docs(self):
        return self._available_docs

    def _evict_if_needed(self):
        while len(self._cache_order) > self._max_cached:
            old = self._cache_order.pop(0)
            self._metadata.pop(old, None)
            self._dense_vecs.pop(old, None)
            self._colbert_vecs.pop(old, None)
            self._sparse_vecs.pop(old, None)
            self._bm25.pop(old, None)
            self._parent_pages.pop(old, None)

    def _touch(self, doc_name):
        if doc_name in self._cache_order:
            self._cache_order.remove(doc_name)
        self._cache_order.append(doc_name)
        self._evict_if_needed()

    def _load_vectors(self, doc_name):
        if doc_name in self._metadata:
            return
        vdb = self._vector_db_dir
        with open(vdb / f"{doc_name}_meta.json", "r", encoding="utf-8") as f:
            self._metadata[doc_name] = json.load(f)
        self._dense_vecs[doc_name] = np.load(
            str(vdb / f"{doc_name}_dense.npy")
        ).astype(np.float32)
        if self._use_colbert:
            colbert_data = np.load(
                str(vdb / f"{doc_name}_colbert.npz"), allow_pickle=True
            )
            self._colbert_vecs[doc_name] = list(colbert_data["vecs"])
        with open(vdb / f"{doc_name}_sparse.pkl", "rb") as f:
            self._sparse_vecs[doc_name] = pickle.load(f)
        corpus = [m["text"].lower().split() for m in self._metadata[doc_name]]
        self._bm25[doc_name] = BM25Okapi(corpus)

    def _load_pages(self, doc_name):
        if doc_name in self._parent_pages:
            return
        fp = self._pages_dir / f"{doc_name}.json"
        if not fp.exists():
            self._parent_pages[doc_name] = {}
            return
        with open(fp, "r", encoding="utf-8") as f:
            doc = json.load(f)
        pages_list = doc.get("content", {}).get("pages", []) or []
        self._parent_pages[doc_name] = {p["page"]: p["text"] for p in pages_list}

    def get(self, doc_name):
        if doc_name not in self._available_docs:
            return None
        self._load_vectors(doc_name)
        self._load_pages(doc_name)
        self._touch(doc_name)
        result = {
            "metadata": self._metadata[doc_name],
            "dense_vecs": self._dense_vecs[doc_name],
            "sparse_vecs": self._sparse_vecs[doc_name],
            "bm25": self._bm25[doc_name],
            "parent_pages": self._parent_pages[doc_name],
        }
        if self._use_colbert and doc_name in self._colbert_vecs:
            result["colbert_vecs"] = self._colbert_vecs[doc_name]
        return result


# ── Retrieval Helpers ───────────────────────────────────────

def sparse_sim(q_sparse, d_sparse):
    score = 0.0
    for token, qw in q_sparse.items():
        if token in d_sparse:
            score += float(qw) * float(d_sparse[token])
    return score


def colbert_maxsim(q_colbert, d_colbert):
    if (
        q_colbert is None
        or d_colbert is None
        or len(q_colbert) == 0
        or len(d_colbert) == 0
    ):
        return 0.0
    q_norm = q_colbert / (np.linalg.norm(q_colbert, axis=1, keepdims=True) + 1e-8)
    d_norm = d_colbert / (np.linalg.norm(d_colbert, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ d_norm.T
    return float(sim.max(axis=1).sum())


def rrf_fuse(rankings, k):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True), dict(scores)


def strip_think(text):
    original = text
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
    text = re.sub(r"<think>[\s\S]*$", "", text).strip()
    if not text and original.strip():
        return original.strip()
    return text


def extract_think(text):
    """Extract content of <think> blocks. Returns concatenated think text or None."""
    blocks = re.findall(r"<think>([\s\S]*?)</think>", text)
    if blocks:
        return "\n".join(b.strip() for b in blocks if b.strip())
    # Unclosed think block (model got cut off mid-think)
    m = re.search(r"<think>([\s\S]*$)", text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


# ── Retrieval ───────────────────────────────────────────────

def retrieve_pages(doc_store, embed_model, semantic_query, resolved_docs,
                   top_pages, use_colbert=True, return_full_ranking=False):
    """Retrieve top pages across multiple documents for a single target."""

    # Encode query
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

    # Score chunks across all resolved docs, keyed by (doc_name, page)
    page_dense = {}
    page_bm25 = {}
    page_sparse = {}
    page_colbert = {}

    for doc_name in resolved_docs:
        doc_data = doc_store.get(doc_name)
        if doc_data is None:
            print(f"  WARNING: {doc_name} not found in vector_db, skipping")
            continue

        meta = doc_data["metadata"]
        all_dense = doc_data["dense_vecs"]
        all_sparse = doc_data["sparse_vecs"]
        bm25 = doc_data["bm25"]
        all_colbert = doc_data.get("colbert_vecs")

        # Dense scores
        dense_scores = (query_dense @ all_dense.T)[0]
        # BM25 scores
        bm25_scores = bm25.get_scores(semantic_query.lower().split())
        # Sparse scores
        sparse_scores = np.array([sparse_sim(query_sparse, s) for s in all_sparse])
        # ColBERT scores
        if use_colbert and all_colbert is not None:
            colbert_scores = np.array(
                [colbert_maxsim(query_colbert, c) for c in all_colbert]
            )
        else:
            colbert_scores = np.zeros(len(meta))

        # Aggregate to (doc_name, page) — best chunk score per page
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
        return [], [] if return_full_ranking else []

    # RRF fusion
    dense_ranked = sorted(page_dense, key=page_dense.get, reverse=True)
    bm25_ranked = sorted(page_bm25, key=page_bm25.get, reverse=True)
    sparse_ranked = sorted(page_sparse, key=page_sparse.get, reverse=True)
    rankings = [dense_ranked, bm25_ranked, sparse_ranked]
    if use_colbert:
        colbert_ranked = sorted(page_colbert, key=page_colbert.get, reverse=True)
        rankings.append(colbert_ranked)

    rrf_ranked, rrf_scores = rrf_fuse(rankings, k=RRF_K)

    # Full ranking (without page text, for gold-page lookup)
    full_ranking = [
        {"doc_name": dn, "page": pg, "rank": rank + 1,
         "score": round(rrf_scores[(dn, pg)], 6)}
        for rank, (dn, pg) in enumerate(rrf_ranked)
    ]

    # Build result pages (top N with text)
    result = []
    for doc_name, pg in rrf_ranked[:top_pages]:
        doc_data = doc_store.get(doc_name)
        if doc_data is None:
            continue
        page_text = doc_data["parent_pages"].get(pg)
        if page_text is None:
            continue
        result.append({
            "doc_name": doc_name,
            "page": pg,
            "text": page_text,
            "score": round(rrf_scores[(doc_name, pg)], 6),
        })

    if return_full_ranking:
        return result, full_ranking
    return result


# ── LLM Calls ──────────────────────────────────────────────

def llm_call(system, user, temperature=None, model=None, think=None):
    """Single LLM call via LM Studio. Returns raw content string."""
    body = {
        "model": model or LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature if temperature is not None else TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    if think is not None:
        body["think"] = think
    resp = requests.post(
        LM_STUDIO_URL,
        json=body,
        timeout=600,
    )
    if not resp.ok:
        # raise_for_status() would discard the response body — keep it, it carries
        # the real reason behind e.g. an HTTP 400 at the n_ctx limit
        raise requests.HTTPError(
            f"HTTP {resp.status_code} from LM Studio: {resp.text[:500]}", response=resp
        )
    data = resp.json()
    choice = data["choices"][0]
    raw = (choice["message"].get("content") or "").strip()
    usage = data.get("usage", {})
    return {
        "content": strip_think(raw),
        "raw": raw,
        "think": extract_think(raw),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def extract_facts(question, needed_info, pages, temperature=None, model=None):
    """Call LLM to extract facts from retrieved pages."""
    context_parts = []
    for p in pages:
        context_parts.append(
            f"=== {p['doc_name']} · page {p['page']} ===\n{p['text']}"
        )
    context_str = "\n\n".join(context_parts)

    user_prompt = (
        f"Question: {question}\n"
        f"Needed info: {needed_info}\n\n"
        f"{context_str}\n\n"
        f"FACTS:"
    )

    result = llm_call(FACT_EXTRACTION_PROMPT, user_prompt,
                       temperature=temperature, model=model, think=False)
    return result["content"], user_prompt, result


def generate_answer(question, all_facts_by_target, task_type, aggregation,
                    temperature=None, model=None, computed_values=None):
    """Call LLM to generate final answer from combined extracted facts.

    When PoT is on, `computed_values` (name -> exact float) is appended as a
    block so the model substitutes the pre-computed numbers instead of doing
    the arithmetic itself."""
    facts_parts = []
    for entry in all_facts_by_target:
        target = entry["target"]
        company = target.get("company", "Unknown")
        docs = ", ".join(entry["docs"])
        facts_parts.append(
            f"=== Facts from {company} ({docs}) ===\n{entry['facts']}"
        )
    facts_str = "\n\n".join(facts_parts)

    user_prompt = (
        f"Question: {question}\n"
        f"Task type: {task_type}\n"
        f"Aggregation: {aggregation}\n\n"
        f"{facts_str}"
    )
    system_prompt = ANSWER_GENERATION_PROMPT
    if computed_values:
        lines = "\n".join(f"- {k} = {v}" for k, v in computed_values.items())
        user_prompt += (
            f"\n\nComputed values (exact — use these, do NOT recompute):\n{lines}"
        )
        # Steer the model only when values are actually present; with PoT off the
        # system prompt stays byte-for-byte the baseline (keeps the A/B clean).
        system_prompt = ANSWER_GENERATION_PROMPT + COMPUTED_VALUES_HINT

    result = llm_call(system_prompt, user_prompt,
                       temperature=temperature, model=model, think=True)
    return result["content"], system_prompt, user_prompt, result


# ── Program-of-Thought (PoT) ────────────────────────────────

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)):     # bool is an int subclass — reject it
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent out of range")  # block int-tower CPU/memory blowups
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "abs" and len(node.args) == 1 and not node.keywords):
        return abs(_eval_node(node.args[0]))          # abs() is the ONLY allowed function
    raise ValueError(f"disallowed: {type(node).__name__}")


def safe_eval(expr):
    """Eval an arithmetic string (+ - * / ** %, unary +/-, abs(), parens).
    Returns float, or None if invalid/forbidden/division-by-zero."""
    try:
        return float(_eval_node(ast.parse(expr, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError, OverflowError):
        return None


def _balanced_object(text, start):
    """Substring of the balanced {...} beginning at index `start`, or None.

    Tracks JSON string literals + escapes so braces inside string values don't
    miscount (e.g. {"k": "a{b}c"} is captured whole)."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_formulas(content):
    """Extract a flat {name: expr} dict from the compute-call output.

    Tolerant: strips ```json … ``` fences, then returns the FIRST balanced {...}
    object that parses to a dict — so trailing prose (even with stray braces),
    code fences, or a second object don't drop the formulas. Returns {} on any
    failure — PoT must never crash the run. A bare {} (no calc needed) is honoured."""
    if not content:
        return {}
    text = content.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    idx = text.find("{")
    while idx != -1:
        obj_str = _balanced_object(text, idx)
        if obj_str is not None:
            try:
                obj = json.loads(obj_str)
            except (json.JSONDecodeError, ValueError):
                obj = None
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        idx = text.find("{", idx + 1)
    return {}


def compute_values(question, all_facts_by_target, temperature=None, model=None):
    """PoT step: LLM proposes named formulas, Python evaluates them safely.
    Returns (computed_values: dict[str,float], formulas: dict[str,str], user_prompt, llm_result)."""
    facts_parts = []
    for entry in all_facts_by_target:
        company = entry["target"].get("company", "Unknown")
        docs = ", ".join(entry["docs"])
        facts_parts.append(f"=== Facts from {company} ({docs}) ===\n{entry['facts']}")
    user_prompt = f"Question: {question}\n\n" + "\n\n".join(facts_parts)

    result = llm_call(COMPUTE_PROMPT, user_prompt, temperature=temperature,
                      model=model, think=False)
    formulas, computed = _parse_formulas(result["content"]), {}
    for name, expr in formulas.items():
        val = safe_eval(str(expr))
        if val is not None:
            computed[name] = round(val, 6)
    return computed, formulas, user_prompt, result


# ── Pipeline Orchestration ──────────────────────────────────

def match_gold_pages(evidences, resolved_docs):
    """Return gold evidences whose doc_name is in resolved_docs."""
    return [
        ev for ev in (evidences or [])
        if ev["doc_name"] in resolved_docs
    ]


def prune_target_docs(target, docs):
    """--prune-docs: collapse FY-only targets spanning two consecutive years to the
    newest 10-K. Its comparative statements cover the prior year on every statement
    (income/cash flow show 3 years, balance sheet 2), so the older filing is
    redundant. Longer spans keep all years — the balance sheet's 2-year window no
    longer covers them (measured on train: pruning 3-year spans loses gold docs)."""
    years = [int(y) for y in (target.get("years") or [])]
    periods = target.get("periods") or []
    if set(periods) == {"FY"} and len(set(years)) == 2 and max(years) - min(years) == 1:
        keep = [d for d in docs if d.endswith(f"_{max(years)}_10K")]
        return keep or docs
    return docs


def compute_hits(full_ranking, gold_evidences):
    """Compute hit@1, hit@5, hit@10 for a target given its full ranking and gold pages."""
    gold_set = {(ev["doc_name"], ev["page_num"]) for ev in gold_evidences}
    if not gold_set:
        return {"hit@1": None, "hit@5": None, "hit@10": None, "gold_ranks": []}
    gold_ranks = []
    for entry in full_ranking:
        key = (entry["doc_name"], entry["page"])
        if key in gold_set:
            gold_ranks.append({
                "doc_name": entry["doc_name"],
                "page": entry["page"],
                "rank": entry["rank"],
                "score": entry["score"],
            })
    ranks = [g["rank"] for g in gold_ranks]
    return {
        "hit@1": any(r <= 1 for r in ranks),
        "hit@5": any(r <= 5 for r in ranks),
        "hit@10": any(r <= 10 for r in ranks),
        "gold_ranks": gold_ranks,
    }


def process_question(s1_record, qa_entry, doc_store, embed_model, config):
    """Process a single question: retrieve per target, extract facts, generate answer."""
    question = qa_entry["question"]
    plan = s1_record["plan"]
    resolution_log = s1_record["resolution_log"]
    evidences = qa_entry.get("evidences", [])

    all_facts = []
    targets_detail = []
    retrieval_targets = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    n_fact_calls = 0

    # Per-target -> per-document: expand each target into one retrieval unit per
    # resolved doc, so every year/document gets its OWN full top-N. Previously all
    # docs of a target competed in one shared RRF pool, so near-identical pages
    # (e.g. the cash-flow page repeated across yearly filings) pushed some years
    # out of the shared top-N. The body below is unchanged and now operates on a
    # single-document `available` list; dict.fromkeys dedups (preserving order)
    # against a plan mapping overlapping years/periods onto the same doc name.
    retrieval_units = []
    for target, res_entry in zip(plan["targets"], resolution_log):
        resolved_docs = res_entry.get("docs", res_entry.get("matched", []))
        if config.get("prune_docs"):
            pruned = prune_target_docs(target, resolved_docs)
            if len(pruned) < len(resolved_docs):
                dropped = sorted(set(resolved_docs) - set(pruned))
                print(f"    Target '{target.get('company', '?')}': --prune-docs dropped {dropped}")
                resolved_docs = pruned
        if not resolved_docs:
            print(f"    Target '{target.get('company', '?')}': no matched docs, skipping")
            continue

        # Filter to docs available in vector_db
        avail = [d for d in resolved_docs if d in doc_store.available_docs]
        if not avail:
            print(f"    Target '{target.get('company', '?')}': none of {resolved_docs} in vector_db")
            continue

        for doc_name in dict.fromkeys(avail):
            retrieval_units.append((target, [doc_name]))

    # Each retrieval unit is a single document with its own full top-N budget.
    for target, available in retrieval_units:

        # Retrieve pages (with full ranking for gold-page lookup)
        pages, full_ranking = retrieve_pages(
            doc_store, embed_model,
            target.get("needed_info") or target.get("semantic_query", question),
            available,
            config["top_pages"],
            use_colbert=config["use_colbert"],
            return_full_ranking=True,
        )

        # Gold-page matching for this target
        target_gold = match_gold_pages(evidences, available)
        hits = compute_hits(full_ranking, target_gold)

        # Print retrieval table
        company = target.get("company", "?")
        print(f"\n  ── Target: {company} ({', '.join(available)}) ──")
        print(f"  {'#':>3}  {'Doc':<25} {'Page':>5}  {'RRF-Score':>10}")
        top_entries = full_ranking[:config["top_pages"]]
        gold_set = {(ev["doc_name"], ev["page_num"]) for ev in target_gold}
        for entry in top_entries:
            is_gold = (entry["doc_name"], entry["page"]) in gold_set
            marker = "  \u2605 GOLD" if is_gold else ""
            print(f"  {entry['rank']:>3}  {entry['doc_name']:<25} {entry['page']:>5}  {entry['score']:>10.6f}{marker}")
        for gr in hits["gold_ranks"]:
            if gr["rank"] <= config["top_pages"]:
                print(f"  Gold {gr['doc_name']} p.{gr['page']} \u2192 Hit@{gr['rank']} \u2713")
            else:
                print(f"  Gold {gr['doc_name']} p.{gr['page']} \u2192 Rank {gr['rank']} (NOT in top {config['top_pages']})")
        for ev in target_gold:
            if not any(gr["doc_name"] == ev["doc_name"] and gr["page"] == ev["page_num"] for gr in hits["gold_ranks"]):
                print(f"  Gold {ev['doc_name']} p.{ev['page_num']} \u2192 NOT RANKED")

        retrieval_targets.append({
            "company": company,
            "docs": available,
            "pages": [
                {"rank": e["rank"], "doc_name": e["doc_name"], "page": e["page"],
                 "score": e["score"],
                 "is_gold": (e["doc_name"], e["page"]) in gold_set}
                for e in top_entries
            ],
            "gold_hits": {
                "hit@1": hits["hit@1"],
                "hit@5": hits["hit@5"],
                "hit@10": hits["hit@10"],
            },
            "gold_ranks": hits["gold_ranks"],
        })

        if not pages:
            print(f"    Target '{company}': no pages retrieved")
            continue

        # LLM fact extraction (batched)
        batch_size = config["fact_batch_size"]
        page_batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]

        fact_batch_results = []
        combined_facts_parts = []
        batch_prompt_tokens = 0
        batch_completion_tokens = 0

        for batch_idx, batch_pages in enumerate(page_batches):
            facts_text, fact_user_prompt, fact_llm_result = extract_facts(
                question, target.get("needed_info", ""), batch_pages,
                temperature=config["temperature"], model=config["model"],
            )
            fact_batch_results.append({
                "batch_idx": batch_idx,
                "n_pages": len(batch_pages),
                "fact_prompt_user": fact_user_prompt,
                "extracted_facts": facts_text,
                "think": fact_llm_result["think"],
                "finish_reason": fact_llm_result["finish_reason"],
                "prompt_tokens": fact_llm_result["prompt_tokens"],
                "completion_tokens": fact_llm_result["completion_tokens"],
            })
            combined_facts_parts.append(facts_text)
            batch_prompt_tokens += fact_llm_result["prompt_tokens"]
            batch_completion_tokens += fact_llm_result["completion_tokens"]
            n_fact_calls += 1

        combined_facts = "\n".join(combined_facts_parts)
        total_prompt_tokens += batch_prompt_tokens
        total_completion_tokens += batch_completion_tokens

        all_facts.append({
            "target": target,
            "facts": combined_facts,
            "docs": available,
        })
        targets_detail.append({
            "target": target,
            "resolved_docs": available,
            "retrieved_pages": [
                {"doc_name": p["doc_name"], "page": p["page"],
                 "text": p["text"], "score": p["score"]}
                for p in pages
            ],
            "fact_prompt_system": FACT_EXTRACTION_PROMPT,
            "extracted_facts": combined_facts,
            "fact_batches": fact_batch_results,
        })

    if not all_facts:
        return {
            "qid": qa_entry["qid"],
            "question": question,
            "predicted_answer": "INSUFFICIENT DATA",
            "gold_answer": qa_entry.get("answer", ""),
            "gold_answer2": qa_entry.get("answer2"),
            "evidences": evidences,
            "s1_plan": plan,
            "n_targets": len(plan["targets"]),
            "n_retrieval_units": len(retrieval_targets),
            "n_fact_calls": 0,
            "error": "No facts extracted from any target",
            "targets_detail": targets_detail,
            "retrieval_targets": retrieval_targets,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
        }

    # PoT (optional): a small compute call sets up named formulas, Python evaluates
    # them exactly, and the exact results are handed to the answer call as a block.
    computed_values, pot_detail = {}, None
    if config.get("pot"):
        computed_values, formulas, compute_user, compute_res = compute_values(
            question, all_facts,
            temperature=config["temperature"], model=config["model"],
        )
        total_prompt_tokens += compute_res["prompt_tokens"]
        total_completion_tokens += compute_res["completion_tokens"]
        pot_detail = {
            "computed_values": computed_values,
            "formulas": formulas,
            "compute_prompt_user": compute_user,
            "raw": compute_res["content"],
            "finish_reason": compute_res["finish_reason"],
            "prompt_tokens": compute_res["prompt_tokens"],
            "completion_tokens": compute_res["completion_tokens"],
        }

    # LLM answer generation (all facts combined)
    answer, answer_system_prompt, answer_user_prompt, answer_llm_result = generate_answer(
        question, all_facts,
        plan.get("task_type", "lookup"),
        plan.get("aggregation", "none"),
        temperature=config["temperature"], model=config["model"],
        computed_values=computed_values,
    )
    total_prompt_tokens += answer_llm_result["prompt_tokens"]
    total_completion_tokens += answer_llm_result["completion_tokens"]

    # Collect all retrieved pages for the compact results file
    all_retrieved = []
    for td in targets_detail:
        for rp in td["retrieved_pages"]:
            all_retrieved.append({"doc_name": rp["doc_name"], "page": rp["page"]})

    return {
        "qid": qa_entry["qid"],
        "question": question,
        "predicted_answer": answer,
        "gold_answer": qa_entry.get("answer", ""),
        "gold_answer2": qa_entry.get("answer2"),
        "evidences": evidences,
        "s1_plan": plan,
        "n_targets": len(plan["targets"]),
        "n_retrieval_units": len(retrieval_targets),
        "n_fact_calls": n_fact_calls,
        "retrieved_pages": all_retrieved,
        "targets_detail": targets_detail,
        "retrieval_targets": retrieval_targets,
        "answer_prompt_system": answer_system_prompt,
        "answer_prompt_user": answer_user_prompt,
        "answer_think": answer_llm_result["think"],
        "answer_finish_reason": answer_llm_result["finish_reason"],
        "pot": pot_detail,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }


# ── Output ──────────────────────────────────────────────────

def build_results_line(r):
    """Compact results line (one per question) — the file Stage 3 run_judge.py reads."""
    return {
        "qid": r["qid"],
        "question": r["question"],
        "predicted_answer": r["predicted_answer"],
        "gold_answer": r["gold_answer"],
        "gold_answer2": r.get("gold_answer2"),
        "s1_plan": r["s1_plan"],
        "n_targets": r["n_targets"],
        "n_retrieval_units": r.get("n_retrieval_units", 0),
        "n_fact_calls": r["n_fact_calls"],
        "retrieved_pages": r.get("retrieved_pages", []),
        "latency_s": r.get("latency_s", 0),
        "error": r.get("error"),
        "timestamp": r.get("timestamp", ""),
    }


def _overall_hits(evidences, retrieval_targets):
    """hit@1/5/10 for one question: hit@k iff ALL gold evidences rank <= k."""
    overall = {"hit@1": None, "hit@5": None, "hit@10": None}
    if not evidences:
        return overall
    all_gold_ranks = []
    for t in retrieval_targets:
        all_gold_ranks.extend(t.get("gold_ranks", []))
    # Per evidence: best rank (or None if not found)
    ev_ranks = []
    for ev in evidences:
        key = (ev["doc_name"], ev["page_num"])
        ranks_for_ev = [gr["rank"] for gr in all_gold_ranks
                        if (gr["doc_name"], gr["page"]) == key]
        ev_ranks.append(min(ranks_for_ev) if ranks_for_ev else None)
    for k_name, k_val in [("hit@1", 1), ("hit@5", 5), ("hit@10", 10)]:
        overall[k_name] = all(rk is not None and rk <= k_val for rk in ev_ranks)
    return overall


def build_retrieval_line(r):
    """Per-question retrieval detail + overall gold hits."""
    rt = r.get("retrieval_targets", [])
    all_gold_ev = r.get("evidences", [])
    return {
        "qid": r["qid"],
        "question": r["question"],
        "gold_answer": r["gold_answer"],
        "predicted_answer": r.get("predicted_answer", ""),
        "evidences": all_gold_ev,
        "targets": rt,
        "overall_hits": _overall_hits(all_gold_ev, rt),
    }


def build_recheck_line(r):
    """Detailed recheck line (full prompts + per-target detail + token counts)."""
    return {
        "qid": r["qid"],
        "question": r["question"],
        "predicted_answer": r["predicted_answer"],
        "gold_answer": r["gold_answer"],
        "gold_answer2": r.get("gold_answer2"),
        "targets_detail": r.get("targets_detail", []),
        "answer_prompt_system": r.get("answer_prompt_system", ""),
        "answer_prompt_user": r.get("answer_prompt_user", ""),
        "pot": r.get("pot"),
        "latency_s": r.get("latency_s", 0),
        "total_prompt_tokens": r.get("total_prompt_tokens", 0),
        "total_completion_tokens": r.get("total_completion_tokens", 0),
    }


def build_progress_line(r):
    """Lightweight per-question sidecar line — the resume source-of-truth.

    Holds exactly the fields compute_summary() consumes (plus qid/error), so a
    resumed run can recompute the combined summary without re-reading the heavy
    results/retrieval/recheck files. See progress_line_to_seed() for the inverse.
    """
    gold_ranks = []
    for t in r.get("retrieval_targets", []):
        gold_ranks.extend(t.get("gold_ranks", []))
    return {
        "qid": r["qid"],
        "error": r.get("error"),
        "n_targets": r["n_targets"],
        "n_retrieval_units": r.get("n_retrieval_units", 0),
        "n_fact_calls": r["n_fact_calls"],
        "n_retrieved_pages": len(r.get("retrieved_pages", [])),
        "latency_s": r.get("latency_s", 0),
        "total_prompt_tokens": r.get("total_prompt_tokens", 0),
        "total_completion_tokens": r.get("total_completion_tokens", 0),
        "evidences": r.get("evidences", []),
        "gold_ranks": gold_ranks,
    }


def progress_line_to_seed(p):
    """Rebuild a minimal result dict from a progress line, exposing exactly the
    keys compute_summary() reads — used to re-seed prior results on resume."""
    return {
        "qid": p["qid"],
        "error": p.get("error"),
        "n_targets": p.get("n_targets", 0),
        "n_retrieval_units": p.get("n_retrieval_units", 0),
        "n_fact_calls": p.get("n_fact_calls", 0),
        "retrieved_pages": [None] * p.get("n_retrieved_pages", 0),
        "latency_s": p.get("latency_s", 0),
        "total_prompt_tokens": p.get("total_prompt_tokens", 0),
        "total_completion_tokens": p.get("total_completion_tokens", 0),
        "evidences": p.get("evidences", []),
        "retrieval_targets": [{"gold_ranks": p.get("gold_ranks", [])}],
    }


def compute_summary(all_results, config):
    """Aggregate summary over all results (config block + metrics + retrieval hits)."""
    n_completed = sum(1 for r in all_results if not r.get("error"))
    n_errors = sum(1 for r in all_results if r.get("error"))
    latencies = [r["latency_s"] for r in all_results if r.get("latency_s")]
    total_targets = sum(r["n_targets"] for r in all_results)
    total_retrieval_units = sum(r.get("n_retrieval_units", 0) for r in all_results)
    total_fact_calls = sum(r["n_fact_calls"] for r in all_results)
    total_pages = sum(len(r.get("retrieved_pages", [])) for r in all_results)

    summary = {
        "config": {
            "model": config["model"],
            "temperature": config["temperature"],
            "top_pages": config["top_pages"],
            "fact_batch_size": config["fact_batch_size"],
            "split": config["split"],
            "use_colbert": config["use_colbert"],
            "prune_docs": config.get("prune_docs", False),
            "s1_results": config["s1_results"],
            "embedding_model": config["embedding_model"],
        },
        "n_questions": len(all_results),
        "n_completed": n_completed,
        "n_errors": n_errors,
        "avg_targets_per_question": round(total_targets / len(all_results), 2) if all_results else 0,
        "avg_retrieval_units_per_question": round(total_retrieval_units / len(all_results), 2) if all_results else 0,
        "avg_fact_calls_per_question": round(total_fact_calls / len(all_results), 2) if all_results else 0,
        "avg_pages_per_question": round(total_pages / len(all_results), 2) if all_results else 0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "total_time_s": round(sum(latencies), 2) if latencies else 0,
        "total_prompt_tokens": sum(r.get("total_prompt_tokens", 0) for r in all_results),
        "total_completion_tokens": sum(r.get("total_completion_tokens", 0) for r in all_results),
    }

    # Aggregate hit@1/5/10 across all questions
    hit_counts = {"hit@1": 0, "hit@5": 0, "hit@10": 0}
    n_with_evidence = 0
    for r in all_results:
        rt = r.get("retrieval_targets", [])
        evs = r.get("evidences", [])
        if not evs:
            continue
        n_with_evidence += 1
        all_gold_ranks = []
        for t in rt:
            all_gold_ranks.extend(t.get("gold_ranks", []))
        ev_ranks = []
        for ev in evs:
            key = (ev["doc_name"], ev["page_num"])
            ranks_for_ev = [gr["rank"] for gr in all_gold_ranks
                            if (gr["doc_name"], gr["page"]) == key]
            ev_ranks.append(min(ranks_for_ev) if ranks_for_ev else None)
        for k_name, k_val in [("hit@1", 1), ("hit@5", 5), ("hit@10", 10)]:
            if all(rk is not None and rk <= k_val for rk in ev_ranks):
                hit_counts[k_name] += 1

    summary["retrieval_hits"] = {
        "n_with_evidence": n_with_evidence,
        "hit@1": hit_counts["hit@1"],
        "hit@5": hit_counts["hit@5"],
        "hit@10": hit_counts["hit@10"],
        "hit@1_rate": round(hit_counts["hit@1"] / n_with_evidence, 4) if n_with_evidence else 0,
        "hit@5_rate": round(hit_counts["hit@5"] / n_with_evidence, 4) if n_with_evidence else 0,
        "hit@10_rate": round(hit_counts["hit@10"] / n_with_evidence, 4) if n_with_evidence else 0,
    }
    return summary


def read_jsonl_safe(path):
    """Read a JSONL file, tolerating a truncated final line (hard crash mid-append).

    Returns the list of parsed objects, stopping at the first unparseable line
    (which can only be the tail of an append-only log)."""
    path = Path(path)
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  WARNING: skipping truncated/corrupt tail line in {path.name}")
                break
    return out


class IncrementalWriter:
    """Writes pipeline results incrementally — one set of lines per question,
    flushed + fsync'd to disk, with the summary refreshed after each write.

    The results / retrieval / recheck / progress files are append-only logs; the
    summary is rewritten atomically (temp file + os.replace) so a crash never
    leaves a half-written summary. On resume, prior_results re-seeds the in-memory
    accumulator so the recomputed summary covers old + new questions.

    Files are reopened in append mode for each write (rather than holding handles
    open for the whole multi-hour run): on win32 this avoids share-lock collisions
    when the user inspects the files / antivirus / backup agents touch them, and
    the reopen cost is negligible next to the 30-60 s LLM latency per question.
    """

    def __init__(self, output_dir, timestamp, config, prior_results=None, summary_every=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = timestamp
        self.config = config
        self.prefix = f"rag_{timestamp}"
        self.summary_every = max(1, summary_every)
        self.all_results = list(prior_results or [])
        self._since_summary = 0
        self.results_path = self.output_dir / f"{self.prefix}_results.jsonl"
        self.retrieval_path = self.output_dir / f"{self.prefix}_retrieval.jsonl"
        self.recheck_path = self.output_dir / f"{self.prefix}_recheck.jsonl"
        self.progress_path = self.output_dir / f"{self.prefix}_progress.jsonl"
        self.summary_path = self.output_dir / f"{self.prefix}_summary.json"

    @staticmethod
    def _append_line(path, obj):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _write_summary(self):
        summary = compute_summary(self.all_results, self.config)
        tmp = self.summary_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.summary_path)

    def write_result(self, result):
        """Append one question's lines to all logs (durable) + refresh summary."""
        self.all_results.append(result)
        self._append_line(self.results_path, build_results_line(result))
        self._append_line(self.retrieval_path, build_retrieval_line(result))
        self._append_line(self.recheck_path, build_recheck_line(result))
        self._append_line(self.progress_path, build_progress_line(result))
        self._since_summary += 1
        if self._since_summary >= self.summary_every:
            self._write_summary()
            self._since_summary = 0

    def finalize(self):
        """Force a final summary write; return the 4 main output paths."""
        self._write_summary()
        self._since_summary = 0
        return self.results_path, self.recheck_path, self.retrieval_path, self.summary_path


def save_results(all_results, output_dir, timestamp, config):
    """One-shot batch save — kept for back-compat. The live pipeline saves via
    IncrementalWriter; this delegates to the same builders for identical output.
    (Does not write the _progress.jsonl sidecar; resume falls back to joining
    the three logs for runs saved this way.)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"rag_{timestamp}"

    results_path = output_dir / f"{prefix}_results.jsonl"
    retrieval_path = output_dir / f"{prefix}_retrieval.jsonl"
    recheck_path = output_dir / f"{prefix}_recheck.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"

    with open(results_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(build_results_line(r), ensure_ascii=False) + "\n")
    with open(retrieval_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(build_retrieval_line(r), ensure_ascii=False) + "\n")
    with open(recheck_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(build_recheck_line(r), ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compute_summary(all_results, config), f, indent=2, ensure_ascii=False)

    return results_path, recheck_path, retrieval_path, summary_path


# ── Printing ────────────────────────────────────────────────

_SEP = "\u2500" * 80


def print_question_result(idx, total, result):
    qid = result["qid"]
    print(f"\n{_SEP}")
    print(f"  {idx}/{total}  {qid}")
    print(f"{_SEP}")
    print(f"  Q: {result['question']}")
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
    else:
        pred = str(result["predicted_answer"])
        gold = str(result["gold_answer"])
        print(f"  A (pred): {pred[:300]}{'...' if len(pred) > 300 else ''}")
        print(f"  A (gold): {gold[:300]}{'...' if len(gold) > 300 else ''}")
        # Hit summary
        rt = result.get("retrieval_targets", [])
        evs = result.get("evidences", [])
        if evs:
            all_gold_ranks = []
            for t in rt:
                all_gold_ranks.extend(t.get("gold_ranks", []))
            ev_ranks = []
            for ev in evs:
                key = (ev["doc_name"], ev["page_num"])
                ranks_for_ev = [gr["rank"] for gr in all_gold_ranks
                                if (gr["doc_name"], gr["page"]) == key]
                ev_ranks.append(min(ranks_for_ev) if ranks_for_ev else None)
            h1 = all(r is not None and r <= 1 for r in ev_ranks)
            h5 = all(r is not None and r <= 5 for r in ev_ranks)
            h10 = all(r is not None and r <= 10 for r in ev_ranks)
            print(f"  Hits: @1={'\u2713' if h1 else '\u2717'}  @5={'\u2713' if h5 else '\u2717'}  @10={'\u2713' if h10 else '\u2717'}")
    if result.get("latency_s"):
        print(f"  Latency: {result['latency_s']:.1f}s")


# ── Main ────────────────────────────────────────────────────

def load_inputs(s1_results_path, split_config_entry):
    """Load S1 plan records + QA data for a split. Returns (s1_records, qa_data).

    Gold evidences are stored 0-indexed but the vector DB is 1-indexed, so
    page_num is bumped by +1 on load (matches the original main() behaviour)."""
    print(f"Loading S1 results from {s1_results_path}")
    s1_records = {}
    with open(s1_results_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "plan" in rec:  # skip error records
                s1_records[rec["qid"]] = rec
    print(f"  {len(s1_records)} plans loaded")

    qa_path = split_config_entry["qa_file"]
    print(f"Loading QA data from {qa_path}")
    qa_data = {}
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            # Gold annotations use 0-indexed pages; vector DB uses 1-indexed
            for ev in q.get("evidences", []):
                ev["page_num"] += 1
            qa_data[q["qid"]] = q
    print(f"  {len(qa_data)} questions loaded")
    return s1_records, qa_data


def build_config_from_args(args):
    """Build the run config dict from parsed argparse args."""
    return {
        "model": args.model,
        "temperature": args.temperature,
        "top_pages": args.top_pages,
        "fact_batch_size": args.fact_batch_size,
        "use_colbert": not args.no_colbert,
        "pot": args.pot,
        "prune_docs": args.prune_docs,
        "split": args.split,
        "s1_results": args.s1_results,
        "embedding_model": args.embedding_model,
    }


def compute_common_qids(s1_records, qa_data, qids=None, start=0, n=None):
    """Questions present in both S1 plans and QA data, filtered by qids/start/n."""
    common = sorted(set(s1_records.keys()) & set(qa_data.keys()))
    if qids:
        common = [q for q in common if q in set(qids)]
    else:
        common = common[start:]
        if n is not None:
            common = common[:n]
    return common


def run_pipeline(config, s1_records, qa_data, common_qids, *,
                 output_dir, timestamp, prior_results=None, no_save=False):
    """Retrieve + extract + answer for each qid, saving incrementally.

    Loads the embedding model + DocumentStore ONCE, then loops, writing every
    result to disk immediately via IncrementalWriter. On resume, prior_results
    re-seeds the summary accumulator so the on-disk summary covers old + new.
    Returns the list of results computed in THIS invocation."""
    split = SPLIT_CONFIG[config["split"]]

    writer = None
    if not no_save:
        writer = IncrementalWriter(output_dir, timestamp, config,
                                   prior_results=prior_results)

    print(f"Processing {len(common_qids)} questions")
    if not common_qids:
        if writer is not None:
            writer.finalize()
        return []

    # Init embedding model
    print(f"Loading embedding model: {config['embedding_model']}")
    embed_model = BGEM3FlagModel(config["embedding_model"], use_fp16=True)

    # Init document store
    print(f"Initializing DocumentStore (colbert={'ON' if config['use_colbert'] else 'OFF'})")
    doc_store = DocumentStore(
        split["vector_db"], split["merged_dir"],
        max_cached=8, use_colbert=config["use_colbert"],
    )
    print(f"  {len(doc_store.available_docs)} docs available in vector_db")

    # Process questions
    run_results = []
    for i, qid in enumerate(common_qids, 1):
        s1_rec = s1_records[qid]
        qa_entry = qa_data[qid]

        print(f"\n{'=' * 40} {i}/{len(common_qids)} {qid} {'=' * 40}")

        t0 = time.time()
        try:
            result = process_question(s1_rec, qa_entry, doc_store, embed_model, config)
            result["latency_s"] = round(time.time() - t0, 2)
            result["timestamp"] = datetime.now().isoformat()
        except Exception as e:
            result = {
                "qid": qid,
                "question": qa_entry["question"],
                "predicted_answer": "",
                "gold_answer": qa_entry.get("answer", ""),
                "gold_answer2": qa_entry.get("answer2"),
                "evidences": qa_entry.get("evidences", []),
                "s1_plan": s1_rec.get("plan", {}),
                "n_targets": 0,
                "n_retrieval_units": 0,
                "n_fact_calls": 0,
                "retrieval_targets": [],
                "error": f"{type(e).__name__}: {e}",
                "latency_s": round(time.time() - t0, 2),
                "timestamp": datetime.now().isoformat(),
            }
            import traceback
            traceback.print_exc()

        run_results.append(result)
        if writer is not None:
            writer.write_result(result)
        print_question_result(i, len(common_qids), result)

    # Summary (over results computed in this invocation)
    n_done = sum(1 for r in run_results if not r.get("error"))
    n_errors = sum(1 for r in run_results if r.get("error"))
    latencies = [r["latency_s"] for r in run_results if r.get("latency_s")]

    print(f"\n{'=' * 80}")
    print(f"  RAG PIPELINE COMPLETE  ({n_done} completed, {n_errors} errors)")
    print(f"{'=' * 80}")
    if latencies:
        print(f"  Mean latency: {sum(latencies)/len(latencies):.1f}s")
        print(f"  Total time:   {sum(latencies):.1f}s")

    if writer is not None:
        results_path, recheck_path, retrieval_path, summary_path = writer.finalize()
        print(f"\n  Results:    {results_path}")
        print(f"  Recheck:    {recheck_path}")
        print(f"  Retrieval:  {retrieval_path}")
        print(f"  Summary:    {summary_path}")
        print(f"  Progress:   {writer.progress_path}")

    return run_results


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline — Stage 2")
    parser.add_argument("--s1-results", required=True, help="Path to S1 eval JSONL")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--n", type=int, default=None, help="Number of questions")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--qids", nargs="+", help="Specific question IDs")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-pages", type=int, default=TOP_PAGES)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--fact-batch-size", type=int, default=FACT_BATCH_SIZE,
                        help="Max pages per fact extraction LLM call")
    parser.add_argument("--no-colbert", action="store_true", help="Skip ColBERT (saves memory)")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--pot", action=argparse.BooleanOptionalAction, default=True,
                        help="Program-of-Thought compute step before answering "
                             "(default: ON; use --no-pot for the frozen baseline / A/B path)")
    parser.add_argument("--prune-docs", action=argparse.BooleanOptionalAction, default=False,
                        help="Collapse FY-only targets spanning two consecutive years to the "
                             "newest 10-K (its comparative statements cover the prior year). "
                             "Default: OFF (baseline behaviour).")
    parser.add_argument("--qids-file",
                        help="File with one qid per line (# comments and blank lines ignored)")
    args = parser.parse_args()

    config = build_config_from_args(args)
    s1_records, qa_data = load_inputs(args.s1_results, SPLIT_CONFIG[args.split])

    qids = list(args.qids or [])
    if args.qids_file:
        with open(args.qids_file, encoding="utf-8") as f:
            qids += [ln.split("#", 1)[0].strip() for ln in f
                     if ln.split("#", 1)[0].strip()]
    common_qids = compute_common_qids(s1_records, qa_data, qids or None, args.start, args.n)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_pipeline(config, s1_records, qa_data, common_qids,
                 output_dir=RESULTS_DIR, timestamp=timestamp,
                 prior_results=None, no_save=args.no_save)


if __name__ == "__main__":
    main()
