"""Evaluate S1 Query Planner against SecQA ground truth.

Single-file pipeline: config, entity resolution, LLM planner, evaluation.

Usage:

    python code/eval_s1.py --file data/secqa_test_train.jsonl     # custom file
    python eval_s1.py --n 20                                # first 20
    python eval_s1.py --start 50 --n 10
    python eval_s1.py --errors                                 # only 4 known error cases
    python eval_s1.py --qids openqa_308 openqa_138             # specific question IDs
"""

import argparse
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
SECQA_PATH = DATA_DIR / "qa" / "secqa_test_train.jsonl"

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LLM_MODEL = "google/gemma-4-31b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 16000

REL_TOLERANCE = 0.01
TOP_K = 8

ERROR_QIDS = ["openqa_308", "openqa_138", "openqa_12", "openqa_266"]

_DOC_NAME_RE = re.compile(r"^([A-Z]+)_(\d{4})(Q[123])?_(10[KQ])$")


def parse_doc_name(doc_name: str) -> dict | None:
    m = _DOC_NAME_RE.match(doc_name)
    if not m:
        return None
    return {
        "ticker": m.group(1),
        "year": int(m.group(2)),
        "period": m.group(3) if m.group(3) else "FY",
        "form": m.group(4),
        "doc_name": doc_name,
    }


def generate_doc_name(ticker: str, year: int, period: str) -> str:
    if period == "FY":
        return f"{ticker}_{year}_10K"
    else:
        return f"{ticker}_{year}{period}_10Q"


# ── Entity Resolution ──────────────────────────────────────

_TICKER_DATA: dict[str, tuple[str, list[str]]] = {
    "AAL":  ("American Airlines", ["american airlines group"]),
    "AAP":  ("Advance Auto Parts", []),
    "AAPL": ("Apple", ["apple inc"]),
    "ABBV": ("AbbVie", ["abbvie inc"]),
    "ABNB": ("Airbnb", ["airbnb inc"]),
    "ADBE": ("Adobe", ["adobe inc", "adobe systems"]),
    "AEP":  ("American Electric Power", []),
    "AMAT": ("Applied Materials", ["applied materials inc"]),
    "AMD":  ("AMD", ["advanced micro devices"]),
    "AMGN": ("Amgen", ["amgen inc"]),
    "AMZN": ("Amazon", ["amazon.com", "amazon inc"]),
    "ANET": ("Arista Networks", ["arista"]),
    "APD":  ("Air Products & Chemicals", ["air products", "air products and chemicals"]),
    "AVGO": ("Broadcom", ["broadcom inc"]),
    "AWK":  ("American Water Works", ["american water"]),
    "AXP":  ("American Express", ["amex"]),
    "AZO":  ("AutoZone", ["auto zone"]),
    "BA":   ("Boeing", ["the boeing company"]),
    "BAC":  ("Bank of America", ["bofa", "bank of america corporation"]),
    "BKNG": ("Booking Holdings", ["booking.com", "priceline"]),
    "BKR":  ("Baker Hughes", ["baker hughes company"]),
    "BLK":  ("BlackRock", ["blackrock inc"]),
    "BMY":  ("Bristol-Myers Squibb", ["bristol myers squibb", "bms"]),
    "BX":   ("Blackstone", ["blackstone inc", "blackstone group"]),
    "C":    ("Citigroup", ["citi", "citibank"]),
    "CBOE": ("Cboe Global Markets", ["cboe", "chicago board options exchange"]),
    "CEG":  ("Constellation Energy", ["constellation"]),
    "CL":   ("Colgate-Palmolive", ["colgate palmolive", "colgate"]),
    "CME":  ("CME Group", ["cme", "chicago mercantile exchange"]),
    "COP":  ("ConocoPhillips", ["conoco phillips"]),
    "COST": ("Costco", ["costco wholesale"]),
    "CRM":  ("Salesforce", ["salesforce.com", "salesforce inc"]),
    "CSCO": ("Cisco Systems", ["cisco"]),
    "CVX":  ("Chevron", ["chevron corporation"]),
    "DAL":  ("Delta Air Lines", ["delta airlines", "delta"]),
    "DD":   ("DuPont", ["du pont", "dupont de nemours"]),
    "DIS":  ("Walt Disney", ["disney", "the walt disney company"]),
    "DLR":  ("Digital Realty", ["digital realty trust"]),
    "DOW":  ("Dow", ["dow inc", "dow chemical"]),
    "EBAY": ("eBay", ["ebay inc"]),
    "ELV":  ("Elevance Health", ["elevance", "anthem"]),
    "EMR":  ("Emerson Electric", ["emerson"]),
    "EQIX": ("Equinix", ["equinix inc"]),
    "EXPE": ("Expedia", ["expedia group"]),
    "F":    ("Ford", ["ford motor", "ford motor company"]),
    "FI":   ("Fiserv", ["fiserv inc"]),
    "FIS":  ("FIS", ["fidelity national information services"]),
    "GE":   ("General Electric", ["ge aerospace"]),
    "GM":   ("General Motors", []),
    "GOOGL": ("Alphabet", ["google", "alphabet inc"]),
    "GPC":  ("Genuine Parts Company", ["genuine parts"]),
    "GS":   ("Goldman Sachs", ["goldman sachs group"]),
    "HAL":  ("Halliburton", ["halliburton company"]),
    "HD":   ("Home Depot", ["the home depot"]),
    "HLT":  ("Hilton", ["hilton worldwide", "hilton hotels"]),
    "HON":  ("Honeywell", ["honeywell international"]),
    "HPQ":  ("HP", ["hp inc", "hewlett-packard", "hewlett packard"]),
    "HUM":  ("Humana", ["humana inc"]),
    "IBM":  ("IBM", ["international business machines"]),
    "IDXX": ("IDEXX", ["idexx laboratories"]),
    "INTC": ("Intel", ["intel corporation"]),
    "IR":   ("Ingersoll Rand", ["ingersoll-rand"]),
    "IRM":  ("Iron Mountain", ["iron mountain inc"]),
    "ISRG": ("Intuitive Surgical", ["intuitive"]),
    "JBL":  ("Jabil", ["jabil inc"]),
    "JNJ":  ("Johnson & Johnson", ["johnson and johnson", "j&j"]),
    "JNPR": ("Juniper Networks", ["juniper"]),
    "JPM":  ("JPMorgan Chase", ["jp morgan chase", "jpmorgan", "jp morgan", "j.p. morgan"]),
    "KDP":  ("Keurig Dr Pepper", ["dr pepper snapple group", "dr pepper", "dr. pepper"]),
    "KEYS": ("Keysight Technologies", ["keysight"]),
    "KKR":  ("KKR", ["kkr & co"]),
    "KO":   ("Coca-Cola", ["coca cola", "coke", "the coca-cola company"]),
    "LLY":  ("Eli Lilly", ["lilly", "eli lilly and company"]),
    "LNT":  ("Alliant Energy", ["alliant"]),
    "LOW":  ("Lowe's", ["lowes", "lowe's companies"]),
    "LUV":  ("Southwest Airlines", ["southwest"]),
    "LYB":  ("LyondellBasell", ["lyondell basell"]),
    "MA":   ("Mastercard", ["mastercard inc"]),
    "MAR":  ("Marriott", ["marriott international"]),
    "MCD":  ("McDonald's", ["mcdonalds", "mcdonald"]),
    "MDT":  ("Medtronic", ["medtronic plc"]),
    "META": ("Meta", ["facebook", "meta platforms", "meta platforms inc"]),
    "MMM":  ("3M", ["3m company"]),
    "MO":   ("Altria Group", ["altria"]),
    "MRK":  ("Merck", ["merck & co", "merck and co"]),
    "MS":   ("Morgan Stanley", []),
    "MSFT": ("Microsoft", ["microsoft corporation"]),
    "MU":   ("Micron Technology", ["micron"]),
    "NFLX": ("Netflix", ["netflix inc"]),
    "NKE":  ("Nike", ["nike inc"]),
    "NVDA": ("NVIDIA", ["nvidia corporation", "nvidia corp"]),
    "ORCL": ("Oracle", ["oracle corporation"]),
    "ORLY": ("O'Reilly Automotive", ["o'reilly", "oreilly", "oreilly automotive", "o'reilly auto parts"]),
    "PARA": ("Paramount Global", ["viacomcbs", "viacom cbs", "paramount", "viacom"]),
    "PEP":  ("PepsiCo", ["pepsi", "pepsico inc"]),
    "PFE":  ("Pfizer", ["pfizer inc"]),
    "PG":   ("Procter & Gamble", ["procter and gamble", "p&g"]),
    "PH":   ("Parker-Hannifin", ["parker hannifin", "parker"]),
    "PM":   ("Philip Morris International", ["philip morris"]),
    "PNR":  ("Pentair", ["pentair plc"]),
    "POOL": ("Pool Corporation", ["pool corp"]),
    "PPG":  ("PPG Industries", []),
    "PYPL": ("PayPal", ["paypal holdings"]),
    "QCOM": ("Qualcomm", ["qualcomm inc"]),
    "SBUX": ("Starbucks", ["starbucks corporation"]),
    "STT":  ("State Street Corporation", ["state street", "state street corp"]),
    "T":    ("AT&T", ["at&t inc"]),
    "TGT":  ("Target", ["target corporation"]),
    "TSLA": ("Tesla", ["tesla inc"]),
    "UAA":  ("Under Armour", ["under armor"]),
    "UAL":  ("United Airlines", ["united airlines holdings"]),
    "UNH":  ("UnitedHealth Group", ["unitedhealth", "united health", "united health group"]),
    "V":    ("Visa", ["visa inc"]),
    "VZ":   ("Verizon", ["verizon communications"]),
    "WBD":  ("Warner Bros. Discovery", ["warner bros discovery", "warner brothers discovery"]),
    "WMT":  ("Walmart", ["walmart inc", "wal-mart"]),
    "XEL":  ("Xcel Energy", ["xcel"]),
    "XOM":  ("ExxonMobil", ["exxon mobil", "exxon", "exxonmobil corporation"]),
    "XYL":  ("Xylem", ["xylem inc"]),
    "YUM":  ("Yum! Brands", ["yum brands", "yum"]),
    "ZTS":  ("Zoetis", ["zoetis inc"]),
}

_ALIASES: dict[str, str] = {}
for _ticker, (_primary, _aliases) in _TICKER_DATA.items():
    _ALIASES[_primary.lower()] = _ticker
    _ALIASES[_ticker.lower()] = _ticker
    for _a in _aliases:
        _ALIASES[_a.lower()] = _ticker


def resolve_company(name: str) -> str | None:
    key = name.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    for suffix in ["\u2019s", "'s", "s"]:
        if key.endswith(suffix):
            candidate = key[: -len(suffix)]
            if candidate in _ALIASES:
                return _ALIASES[candidate]
    for suffix in [
        " inc", " inc.", " corp", " corp.", " co", " co.",
        " group", " holdings", " ltd", " plc",
        " corporation", " company",
    ]:
        if key.endswith(suffix):
            candidate = key[: -len(suffix)].strip()
            if candidate in _ALIASES:
                return _ALIASES[candidate]
    return None


# ── S1 Query Planner ───────────────────────────────────────

SYSTEM_PROMPT = r"""You are a financial document routing agent.  Given a question about SEC filings (10-K annual reports and 10-Q quarterly reports), identify which documents are needed to answer it.

## Task
Output a JSON plan specifying which companies, fiscal years, and periods to search, plus what information to extract.

## Output Format
Output ONLY a single valid JSON object.  No markdown fences, no explanation text before or after the JSON.

```
{
  "targets": [
    {
      "company": "<company name as written in the question>",
      "years": [2023],
      "periods": ["FY"],
      "semantic_query": "<concise search query for vector retrieval>",
      "needed_info": "<what value or fact to extract>"
    }
  ],
  "task_type": "<lookup | compare_companies | trend | compare_quarters | arithmetic>",
  "aggregation": "<none | sum | diff | ratio | growth | argmax | argmin | avg>"
}
```

## Field Reference
- **company**: Exact company name from the question (or the well-known name for the company).
- **years**: The fiscal years whose filings you will search — the **minimal** set per the Year Rules below (list of integers).
- **periods**: Which periods to search inside each year.
  - `"FY"` = full fiscal year → 10-K annual filing.
  - `"Q1"`, `"Q2"`, `"Q3"` = fiscal quarter → 10-Q quarterly filing.
  - There is NO separate Q4 filing.  See Rule 2 below.
- **semantic_query**: A detailed retrieval query (10-20 words) used to find the right pages via vector search. Include: (1) the specific line item or metric name, (2) the financial statement or section where it appears (e.g. "consolidated statements of cash flows", "income statement", "balance sheet"), and (3) related synonyms or terms that appear near the data (e.g. "capital expenditures" alongside "free cash flow"). Do NOT include company names or years — those are already handled by document filtering.
- **needed_info**: What specific metric, value, or fact must be extracted.
- **task_type**: One of `lookup` (single value), `compare_companies` (compare across firms), `trend` (multi-period for one firm), `compare_quarters` (compare quarters within a year), `arithmetic` (calculation-heavy).
- **aggregation**: How to combine extracted values.  `none` if a single lookup; `sum`, `diff`, `ratio`, `growth`, `argmax`, `argmin`, or `avg` otherwise.

## Critical Period Rules (MUST follow)

**Rule 1 — "which quarter" / superlative over quarters, no specific quarter named**
→ periods = `["Q1", "Q2", "Q3", "FY"]`.  All four are needed so every quarter can be compared.  FY (10-K) gives the full-year total from which Q4 is derived.
→ aggregation = `"argmax"` or `"argmin"`.

**Rule 2 — Q4 / fourth quarter explicitly mentioned**
→ There is NO standalone Q4 filing.  Q4 = full-year (10-K) minus cumulative nine months (Q3 10-Q).
→ periods MUST include `["Q3", "FY"]` for the relevant year.

**Rule 3 — "nine months ended" / "first nine months"**
→ The Q3 10-Q contains cumulative 9-month data.
→ periods = `["Q3"]`.

**Rule 4 — Fiscal year ≠ calendar year**
→ Some companies (e.g. HP, Nike, Microsoft) have non-calendar fiscal years.
→ Use the fiscal year as stated in the question.  If the question says "FY2025 Q1", output year=2025, period="Q1".

**Rule 5 — Different years need different periods → one target per year**
→ A target's years × periods expand as a full cross-product: EVERY listed period is fetched for EVERY listed year.
→ If the question needs different periods in different years (e.g. Q3 of FY2024 but Q1 of FY2025), do NOT merge them into one target — the cross-product would fetch documents nobody asked for (2024 Q1, 2025 Q3, 2025 FY).
→ Output one target per year instead, each listing only that year's needed periods (same company, same semantic_query).

## Year Rules
- "in 2023" or "for 2023" → `[2023]`.
- A 10-K prints **comparative prior-year columns** on the same page as the current year: the income
  statement and the cash-flow statement show **3 fiscal years** (filing year + 2 prior); the balance
  sheet shows **2 fiscal years** (filing year + 1 prior).
- For a **FY-only** question spanning several consecutive fiscal years, list the **smallest set of
  10-Ks whose comparative columns already cover every requested year** — do NOT emit one 10-K per year
  out of caution. Start at the newest requested year and step backwards by the comparative window:
  - income / cash-flow metrics (revenue, net income, operating income, margins, free cash flow,
    operating/investing/financing cash flow): step **3** → an FY2020-2024 revenue trend = years
    `[2024, 2021]` (2024 covers 2024/2023/2022, 2021 covers 2021/2020), NOT all five.
  - balance-sheet metrics (total assets, total liabilities, equity, cash, debt, inventory): step **2**
    → an FY2020-2024 total-assets trend = years `[2024, 2022, 2020]`.
- **If you are unsure the metric appears in a comparative column** (segment breakdowns, unusual or
  restated line items), list that fiscal year's own 10-K to be safe — recall matters more than a
  perfectly minimal list.
- This minimisation applies to **FY / 10-K** targets only. Quarterly (10-Q) targets keep exactly the
  periods required by the Critical Period Rules above.
- Always name the full requested year range in `needed_info` (e.g. "annual revenue for FY2020-2024")
  so extraction reads every comparative column from the chosen filings.

## Multi-Company Questions
- "Among A, B, and C …" → create one target per company, all with the same semantic_query / needed_info.
- "Compare A and B" → two targets.

## Semantic Query Guidelines
The semantic_query is matched against document page chunks via embedding similarity. Write queries that would appear on the same page as the data you need:
- Include the **financial statement name** where the data lives (e.g. "consolidated statements of cash flows", "consolidated balance sheets", "consolidated statements of operations").
- Include **synonyms and related line items** that typically appear near the target data (e.g. for free cash flow: "operating cash flow", "capital expenditures", "cash provided by operations").
- For segment data, include the **segment name** and "segment results" or "segment information".
- For ratios/margins, include both the **numerator and denominator** terms.
- Do NOT include company names or years — document filtering already handles those.

## Examples

### Single lookup
Q: "What was Costco's total revenue in the third quarter of fiscal year 2024?"
{"targets":[{"company":"Costco","years":[2024],"periods":["Q3"],"semantic_query":"total revenue net sales consolidated statements of operations","needed_info":"total revenue value"}],"task_type":"lookup","aggregation":"none"}

### Two-company comparison
Q: "What is the percentage difference of Netflix's operating income compared to that of Walt Disney in 2024?"
{"targets":[{"company":"Netflix","years":[2024],"periods":["FY"],"semantic_query":"operating income consolidated statements of operations income from operations","needed_info":"operating income"},{"company":"Walt Disney","years":[2024],"periods":["FY"],"semantic_query":"operating income consolidated statements of operations income from operations","needed_info":"operating income"}],"task_type":"compare_companies","aggregation":"diff"}

### Multi-year trend (minimal filings via comparative columns)
Q: "As of 2024, what is the overall free cash flow trend of Procter & Gamble over the recent 5-year period?"
{"targets":[{"company":"Procter & Gamble","years":[2024,2021],"periods":["FY"],"semantic_query":"free cash flow operating cash flow capital expenditures consolidated statements of cash flows","needed_info":"annual free cash flow for fiscal years 2020 through 2024"}],"task_type":"trend","aggregation":"none"}

### Three-company superlative
Q: "Among Intel, AMD, and Qualcomm, what is the operating income of the company that has the highest R&D in 2023?"
{"targets":[{"company":"Intel","years":[2023],"periods":["FY"],"semantic_query":"research and development expenses operating income consolidated statements of operations R&D","needed_info":"R&D expenses and operating income"},{"company":"AMD","years":[2023],"periods":["FY"],"semantic_query":"research and development expenses operating income consolidated statements of operations R&D","needed_info":"R&D expenses and operating income"},{"company":"Qualcomm","years":[2023],"periods":["FY"],"semantic_query":"research and development expenses operating income consolidated statements of operations R&D","needed_info":"R&D expenses and operating income"}],"task_type":"compare_companies","aggregation":"argmax"}

### Which-quarter (Rule 1)
Q: "In which quarter of 2023 did Ford report the highest net income?"
{"targets":[{"company":"Ford","years":[2023],"periods":["Q1","Q2","Q3","FY"],"semantic_query":"net income consolidated statements of operations earnings loss","needed_info":"net income per quarter"}],"task_type":"compare_quarters","aggregation":"argmax"}

### Q4 derivation (Rule 2)
Q: "What was Starbucks' revenue in Q4 of fiscal year 2023?"
{"targets":[{"company":"Starbucks","years":[2023],"periods":["Q3","FY"],"semantic_query":"total net revenues consolidated statements of earnings quarterly revenue","needed_info":"Q4 revenue"}],"task_type":"lookup","aggregation":"none"}

### Mixed periods across years (Rule 5)
Q: "What was Nike's revenue in Q4 of fiscal 2023 and in Q1 of fiscal 2024?"
{"targets":[{"company":"Nike","years":[2023],"periods":["Q3","FY"],"semantic_query":"total revenues consolidated statements of income quarterly revenue","needed_info":"Q4 FY2023 revenue"},{"company":"Nike","years":[2024],"periods":["Q1"],"semantic_query":"total revenues consolidated statements of income quarterly revenue","needed_info":"Q1 FY2024 revenue"}],"task_type":"lookup","aggregation":"none"}

### Nine months cumulative (Rule 3)
Q: "What was Tesla's net income for the first nine months of 2024?"
{"targets":[{"company":"Tesla","years":[2024],"periods":["Q3"],"semantic_query":"net income nine months ended consolidated statements of operations","needed_info":"cumulative net income for nine months ended"}],"task_type":"lookup","aggregation":"none"}

### Multi-year growth (minimal filings)
Q: "What is Amazon's overall revenue growth over the last 4-year period as of 2024?"
{"targets":[{"company":"Amazon","years":[2024,2021],"periods":["FY"],"semantic_query":"total net sales revenue consolidated statements of operations","needed_info":"annual total revenue for fiscal years 2021 through 2024"}],"task_type":"trend","aggregation":"growth"}

### Multi-year balance-sheet trend (2-year comparative window → step 2)
Q: "How did Microsoft's total assets change across fiscal years 2021 to 2024?"
{"targets":[{"company":"Microsoft","years":[2024,2022],"periods":["FY"],"semantic_query":"total assets consolidated balance sheets","needed_info":"total assets for fiscal years 2021 through 2024"}],"task_type":"trend","aggregation":"none"}
"""


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
    text = re.sub(r"<think>[\s\S]*$", "", text).strip()
    return text


def _extract_json(text: str) -> dict:
    clean = _strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", clean)
    if m:
        return json.loads(m.group(1))
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(clean[start : end + 1])
    raise ValueError(f"No JSON found in LLM response:\n{clean[:500]}")


def _validate_plan(plan: dict) -> dict:
    if "targets" not in plan or not isinstance(plan["targets"], list):
        raise ValueError("Plan missing 'targets' list")
    for i, t in enumerate(plan["targets"]):
        if "company" not in t:
            raise ValueError(f"Target {i} missing 'company'")
        if "years" not in t or not isinstance(t["years"], list):
            raise ValueError(f"Target {i} missing or invalid 'years'")
        t["years"] = [int(y) for y in t["years"]]
        if "periods" not in t or not isinstance(t["periods"], list):
            t["periods"] = ["FY"]
        valid_periods = {"FY", "Q1", "Q2", "Q3"}
        t["periods"] = [p.upper() for p in t["periods"]]
        t["periods"] = [p for p in t["periods"] if p in valid_periods]
        if not t["periods"]:
            t["periods"] = ["FY"]
        t.setdefault("semantic_query", "")
        t.setdefault("needed_info", "")
    plan.setdefault("task_type", "lookup")
    plan.setdefault("aggregation", "none")
    return plan


def llm_plan(question: str, *, temperature: float | None = None) -> dict:
    resp = requests.post(
        LM_STUDIO_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    parsed = _extract_json(raw)
    return _validate_plan(parsed)


# ── Evaluation ─────────────────────────────────────────────

def load_questions(path: Path = SECQA_PATH) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions



def resolve_plan_to_docs(plan: dict) -> tuple[set[str], list[dict]]:
    resolved = set()
    log = []
    for target in plan.get("targets", []):
        company = target["company"]
        ticker = resolve_company(company)
        entry = {
            "company": company,
            "ticker": ticker,
            "years": target["years"],
            "periods": target["periods"],
            "docs": [],
        }
        if ticker is None:
            log.append(entry)
            continue
        for year in target["years"]:
            for period in target["periods"]:
                doc = generate_doc_name(ticker, year, period)
                entry["docs"].append(doc)
                resolved.add(doc)
        log.append(entry)
    return resolved, log


def compute_metrics(resolved_docs: set[str], gt_docs: set[str]) -> dict:
    if not gt_docs:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0, "perfect": True}
    hits = resolved_docs & gt_docs
    recall = len(hits) / len(gt_docs) if gt_docs else 0.0
    precision = len(hits) / len(resolved_docs) if resolved_docs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    perfect = hits == gt_docs
    return {"recall": recall, "precision": precision, "f1": f1, "perfect": perfect}


def compute_company_recall(plan: dict, gt_docs: set[str]) -> dict:
    gt_tickers = {parse_doc_name(d)["ticker"] for d in gt_docs if parse_doc_name(d)}
    plan_tickers = set()
    for t in plan.get("targets", []):
        ticker = resolve_company(t["company"])
        if ticker:
            plan_tickers.add(ticker)
    if not gt_tickers:
        return {"company_recall": 1.0, "gt_tickers": [], "plan_tickers": []}
    hits = plan_tickers & gt_tickers
    return {
        "company_recall": len(hits) / len(gt_tickers),
        "gt_tickers": sorted(gt_tickers),
        "plan_tickers": sorted(plan_tickers),
        "missed_tickers": sorted(gt_tickers - plan_tickers),
    }


# ── Printing ───────────────────────────────────────────────

_SEP = "\u2500" * 80


def print_question_result(
    idx, total, qobj, plan, resolved_docs, gt_docs, metrics, company_info, error=None
):
    qid = qobj["qid"]
    print(f"\n{_SEP}")
    print(f"  {idx}/{total}  {qid}")
    print(f"{_SEP}")
    print(f"  Q: {qobj['question']}")

    if error:
        print(f"  ERROR: {error}")
        return

    for t in plan.get("targets", []):
        ticker = resolve_company(t["company"]) or "???"
        print(f"  Plan: {t['company']} ({ticker})  years={t['years']}  periods={t['periods']}")
    print(f"  task_type={plan.get('task_type')}  aggregation={plan.get('aggregation')}")

    print(f"  Resolved docs: {sorted(resolved_docs)}")
    print(f"  GT docs:       {sorted(gt_docs)}")

    r_icon = "\u2713" if metrics["recall"] == 1.0 else "\u2717"
    p_icon = "\u2713" if metrics["precision"] == 1.0 else "\u2717"
    c_icon = "\u2713" if company_info["company_recall"] == 1.0 else "\u2717"
    print(
        f"  Recall: {metrics['recall']:.0%} {r_icon}  "
        f"Precision: {metrics['precision']:.0%} {p_icon}  "
        f"Company: {company_info['company_recall']:.0%} {c_icon}"
    )

    if company_info.get("missed_tickers"):
        print(f"  Missed companies: {company_info['missed_tickers']}")
    missed_docs = gt_docs - resolved_docs
    if missed_docs:
        print(f"  Missed docs: {sorted(missed_docs)}")
    extra_docs = resolved_docs - gt_docs
    if extra_docs:
        print(f"  Extra docs: {sorted(extra_docs)}")


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate S1 Query Planner")
    parser.add_argument("--file", type=str, default=None, help="Path to JSONL question file")
    parser.add_argument("--n", type=int, default=None, help="Number of questions to process")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--save", action="store_true", help="Save results to JSONL", default=True)
    parser.add_argument("--errors", action="store_true", help="Only run the 4 known error cases")
    parser.add_argument("--qids", nargs="+", help="Only run specific question IDs")
    args = parser.parse_args()

    input_path = Path(args.file) if args.file else SECQA_PATH
    all_questions = load_questions(input_path)
    print(f"Loaded {len(all_questions)} questions from {input_path.name}")

    # Filter by QIDs (--errors or --qids)
    filter_qids = None
    if args.errors:
        filter_qids = set(ERROR_QIDS)
    elif args.qids:
        filter_qids = set(args.qids)

    if filter_qids:
        questions = [q for q in all_questions if q["qid"] in filter_qids]
        missing = filter_qids - {q["qid"] for q in questions}
        if missing:
            print(f"WARNING: QIDs not found in dataset: {sorted(missing)}")
        print(f"Processing {len(questions)} filtered questions: {[q['qid'] for q in questions]}")
    else:
        questions = all_questions[args.start:]
        if args.n is not None:
            questions = questions[: args.n]
        print(f"Processing questions {args.start}..{args.start + len(questions) - 1}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"s1_eval_{ts}.jsonl"

    total = len(questions)
    n_done = 0
    n_errors = 0
    sum_recall = 0.0
    sum_precision = 0.0
    sum_company_recall = 0.0
    n_perfect = 0

    for i, qobj in enumerate(questions, 1):
        qid = qobj["qid"]
        gt_docs = {ev["doc_name"] for ev in qobj["evidences"]}

        try:
            result_plan = llm_plan(qobj["question"])
            error = None
        except Exception as e:
            result_plan = None
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

        if error:
            n_errors += 1
            print_question_result(i, total, qobj, None, set(), gt_docs, {}, {}, error)
            if args.save:
                record = {"qid": qid, "error": error, "timestamp": datetime.now().isoformat()}
                with open(results_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            continue

        resolved_docs, resolution_log = resolve_plan_to_docs(result_plan)
        metrics = compute_metrics(resolved_docs, gt_docs)
        company_info = compute_company_recall(result_plan, gt_docs)

        n_done += 1
        sum_recall += metrics["recall"]
        sum_precision += metrics["precision"]
        sum_company_recall += company_info["company_recall"]
        if metrics["perfect"]:
            n_perfect += 1

        print_question_result(
            i, total, qobj, result_plan, resolved_docs, gt_docs, metrics, company_info
        )

        if args.save:
            record = {
                "qid": qid,
                "question": qobj["question"],
                "plan": result_plan,
                "resolved_docs": sorted(resolved_docs),
                "gt_docs": sorted(gt_docs),
                "metrics": metrics,
                "company_info": company_info,
                "resolution_log": resolution_log,
                "timestamp": datetime.now().isoformat(),
            }
            with open(results_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 80}")
    print(f"  S1 PLANNER EVALUATION  ({n_done} evaluated, {n_errors} errors)")
    print(f"{'=' * 80}")

    if n_done > 0:
        avg_recall = sum_recall / n_done
        avg_precision = sum_precision / n_done
        avg_company = sum_company_recall / n_done
        print(f"  Avg doc recall:      {avg_recall:.1%}")
        print(f"  Avg doc precision:   {avg_precision:.1%}")
        print(f"  Avg company recall:  {avg_company:.1%}")
        print(f"  Perfect plan rate:   {n_perfect}/{n_done} ({n_perfect/n_done:.1%})")

    if args.save:
        print(f"\n  Results saved to: {results_file}")


if __name__ == "__main__":
    main()
