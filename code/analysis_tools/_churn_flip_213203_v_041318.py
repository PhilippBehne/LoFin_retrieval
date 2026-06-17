"""Wegwerf: Churn/Flip-Diff Run A (213203) vs Run B (041318) auf train (143 qids).

Baut die volle 143-qid-Status-Matrix (correct/partial/incorrect/skip-Subtyp) fuer
beide Laeufe. Skip-Logik EXAKT wie code/run_judge.py:
  error gesetzt -> crash ; predicted_answer leer -> empty ;
  predicted_answer normalisiert (strip, rstrip("."), strip, casefold) == "insufficient data" -> insuff.
Status correct = judge_verdict == "correct"; partial separat; sonst incorrect.
Gibt Uebergangsmatrix A->B, Brutto-Churn vs Netto, Signal-Anteil (Ziel-qids), Regressionen.
Read-only.
"""
import json
from collections import defaultdict, Counter

DIR = "data/results"
A = "20260613_213203"   # Baseline, OHNE 3 Regeln
B = "20260615_041318"   # MIT 3 Regeln
TARGET_QIDS = {"openqa_267", "openqa_155", "openqa_126", "openqa_144"}


def load_jsonl(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["qid"]] = r
    return d


def is_insufficient(pa):
    return (pa or "").strip().rstrip(".").strip().casefold() == "insufficient data"


def status_map(ts):
    """Pro qid genau ein Status, ueber ALLE results-qids (143)."""
    results = load_jsonl(f"{DIR}/rag_{ts}_results.jsonl")
    judged = load_jsonl(f"{DIR}/rag_{ts}_judged.jsonl")
    out = {}
    for qid, r in results.items():
        if qid in judged:
            v = judged[qid].get("judge_verdict")
            if v == "correct":
                out[qid] = "correct"
            elif v == "partially_correct":
                out[qid] = "partial"
            else:
                out[qid] = "incorrect"
        else:  # vom Judge uebersprungen -> Subtyp wie run_judge.py
            pa = r.get("predicted_answer")
            if r.get("error"):
                out[qid] = "crash"
            elif not pa:
                out[qid] = "empty"
            elif is_insufficient(pa):
                out[qid] = "insuff"
            else:
                out[qid] = "skip?"  # sollte nie passieren
    return out, results, judged


a_st, a_res, a_jud = status_map(A)
b_st, b_res, b_jud = status_map(B)

print(f"qids: A={len(a_st)} B={len(b_st)}  shared={len(set(a_st)&set(b_st))}")
print(f"A {A}: {dict(Counter(a_st.values()))}")
print(f"B {B}: {dict(Counter(b_st.values()))}")

# Cross-check gegen Summaries
for tag, st in [("A", a_st), ("B", b_st)]:
    c = sum(1 for s in st.values() if s == "correct")
    p = sum(1 for s in st.values() if s == "partial")
    inc = sum(1 for s in st.values() if s == "incorrect")
    sk = sum(1 for s in st.values() if s in ("crash", "insuff", "empty", "skip?"))
    print(f"  {tag}: correct={c} partial={p} incorrect={inc} skip={sk}  (sum={c+p+inc+sk})")

# Skip-bucket gibt es als correct=Treffer-Bool fuer Flip-Vergleich
def hit(s):  # nur correct zaehlt als Treffer
    return s == "correct"

# Uebergangsmatrix
all_q = sorted(set(a_st) | set(b_st))
trans = defaultdict(list)
for q in all_q:
    trans[(a_st.get(q, "MISS"), b_st.get(q, "MISS"))].append(q)

order = ["correct", "partial", "incorrect", "insuff", "crash", "empty", "skip?", "MISS"]
def k(s): return order.index(s) if s in order else 99

print("\n=== Uebergangsmatrix A -> B (nur Wechsel) ===")
churn_qids = []
for (sa, sb) in sorted(trans, key=lambda x: (k(x[0]), k(x[1]))):
    if sa == sb:
        continue
    qs = sorted(trans[(sa, sb)])
    churn_qids += qs
    mark = ""
    if sa == "correct" and sb != "correct":
        mark = "   <== REGRESSION"
    elif sb == "correct" and sa != "correct":
        mark = "   <== GEWONNEN"
    tgt = [q for q in qs if q in TARGET_QIDS]
    tgt_s = f"  [Ziel: {tgt}]" if tgt else ""
    print(f"  {sa:>9} -> {sb:<9}: {len(qs):2}  {qs}{mark}{tgt_s}")

print("\n=== unveraendert (Status gleich) ===")
for (sa, sb) in sorted(trans, key=lambda x: (k(x[0]), k(x[1]))):
    if sa == sb:
        print(f"  {sa:>9} == {sb:<9}: {len(trans[(sa, sb)]):3}")

# Kernzahlen
brutto = len(set(churn_qids))
a_corr = sum(1 for s in a_st.values() if s == "correct")
b_corr = sum(1 for s in b_st.values() if s == "correct")
netto = b_corr - a_corr

up_flips = [q for q in all_q if not hit(a_st.get(q)) and hit(b_st.get(q))]      # ->correct
down_flips = [q for q in all_q if hit(a_st.get(q)) and not hit(b_st.get(q))]    # correct->
up_on_target = [q for q in up_flips if q in TARGET_QIDS]
up_off_target = [q for q in up_flips if q not in TARGET_QIDS]

print("\n=== KERNZAHLEN ===")
print(f"Brutto-Churn (qids mit IRGENDEINEM Statuswechsel): {brutto}")
print(f"Netto correct: A={a_corr} -> B={b_corr}  ({netto:+d})")
print(f"Aufwaerts-Flips (->correct): {len(up_flips)}  {up_flips}")
print(f"  davon auf ZIEL-qids ({sorted(TARGET_QIDS)}): {len(up_on_target)}  {up_on_target}")
print(f"  davon VERSTREUT (off-target): {len(up_off_target)}  {up_off_target}")
print(f"Abwaerts-Flips (correct->): {len(down_flips)}  {down_flips}")
print(f"Ziel-qids Status A->B:")
for q in sorted(TARGET_QIDS):
    print(f"    {q}: {a_st.get(q,'MISS')} -> {b_st.get(q,'MISS')}")

# Regressionen + correct->skip: Texte zeigen
print("\n=== REGRESSIONEN / correct->skip (Detail) ===")
regress = [q for q in all_q if a_st.get(q) == "correct" and b_st.get(q) != "correct"]
for q in sorted(regress):
    pa_a = (a_jud.get(q) or a_res.get(q) or {}).get("predicted_answer", "")
    pa_b = (b_jud.get(q) or b_res.get(q) or {}).get("predicted_answer", "")
    gold = (a_res.get(q) or {}).get("gold_answer", "")
    gold2 = (a_res.get(q) or {}).get("gold_answer2")
    print(f"\n--- {q}  [{a_st.get(q)} -> {b_st.get(q)}]  {'(ZIEL)' if q in TARGET_QIDS else ''}")
    print(f"  GOLD : {gold}")
    if gold2:
        print(f"  GOLD2: {gold2}")
    print(f"  PRED A: {pa_a}")
    print(f"  PRED B: {pa_b}")
    if q in b_jud:
        print(f"  Judge-B reason: {b_jud[q].get('judge_reason','')}")

# partial rein/raus
print("\n=== partial-Wechsel ===")
into_partial = [q for q in all_q if a_st.get(q) != "partial" and b_st.get(q) == "partial"]
out_partial = [q for q in all_q if a_st.get(q) == "partial" and b_st.get(q) != "partial"]
print(f"  ->partial: {into_partial}")
print(f"  partial->: {out_partial}")
for q in into_partial:
    print(f"    {q}: {a_st.get(q)} -> partial | PRED_B: {(b_jud.get(q) or {}).get('predicted_answer','')[:120]}")
