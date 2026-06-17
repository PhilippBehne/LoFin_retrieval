"""Wegwerf: qid-Level-Diff zweier Laeufe (Baseline vs PoT). Kein Modell.

Kombiniert _results.jsonl (error / INSUFFICIENT-Kapitulation) mit _judged.jsonl
(Judge-Verdikt) zu EINEM Status pro qid und stellt die Uebergangsmatrix auf.
Beantwortet: welche Crashes wurden gerettet (n_ctx?), welche INSUFFICIENT hat PoT
in correct/incorrect verwandelt, und gab es Regressionen (correct -> schlechter).
"""
import json
from collections import defaultdict, Counter

DIR = "data/results"
BASE = "20260606_020413"  # Baseline (kein PoT)
POT = "20260607_013732"   # voller PoT-Lauf


def load_jsonl(path):
    d = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    d[r["qid"]] = r
    except FileNotFoundError:
        print(f"  (fehlt: {path})")
    return d


def status_map(ts):
    """Pro qid ein disjunkter Status. Prioritaet bei Skip: crash > insuff > empty."""
    results = load_jsonl(f"{DIR}/rag_{ts}_results.jsonl")
    judged = load_jsonl(f"{DIR}/rag_{ts}_judged.jsonl")
    out = {}
    for qid, r in results.items():
        if qid in judged:
            j = judged[qid]
            if j.get("eval_correct"):
                out[qid] = "correct"
            elif j.get("eval_partially_correct"):
                out[qid] = "partial"
            else:
                out[qid] = "incorrect"
        else:  # vom Judge uebersprungen
            if r.get("error"):
                out[qid] = "crash"
            elif r.get("predicted_answer") == "INSUFFICIENT DATA":
                out[qid] = "insuff"
            elif not r.get("predicted_answer"):
                out[qid] = "empty"
            else:
                out[qid] = "skip?"
    return out, results


base_st, base_res = status_map(BASE)
pot_st, pot_res = status_map(POT)

print(f"Baseline {BASE}: {dict(Counter(base_st.values()))}  (n={len(base_st)})")
print(f"PoT      {POT}: {dict(Counter(pot_st.values()))}  (n={len(pot_st)})")

# Overlap error & INSUFFICIENT (erklaert die +/-1 in der Skip-Bilanz)
for tag, res in [("BASE", base_res), ("POT", pot_res)]:
    ov = [q for q, r in res.items()
          if r.get("error") and r.get("predicted_answer") == "INSUFFICIENT DATA"]
    print(f"{tag} overlap (error UND INSUFFICIENT): {len(ov)} {ov}")

# Uebergangsmatrix
order = ["correct", "partial", "incorrect", "insuff", "crash", "empty", "skip?"]
def k(s): return order.index(s) if s in order else 99

trans = defaultdict(list)
for q in sorted(set(base_st) | set(pot_st)):
    trans[(base_st.get(q, "MISS"), pot_st.get(q, "MISS"))].append(q)

print("\n=== Uebergaenge (Baseline -> PoT) ===")
for (b, p) in sorted(trans, key=lambda x: (k(x[0]), k(x[1]))):
    qs = trans[(b, p)]
    mark = ""
    if b in ("correct", "partial") and p in ("incorrect", "insuff", "crash", "empty"):
        mark = "  <-- REGRESSION"
    elif b in ("incorrect", "insuff", "crash", "empty") and p == "correct":
        mark = "  <-- GEWONNEN"
    print(f"  {b:>9} -> {p:<9}: {len(qs):2}  {qs}{mark}")

print("\n=== Baseline-Crashes -> wohin? (n_ctx-Kandidaten) ===")
for q in sorted(q for q, s in base_st.items() if s == "crash"):
    still = "  (crasht weiter)" if pot_st.get(q) == "crash" else ""
    print(f"  {q}: crash -> {pot_st.get(q, '?')}{still}")

print("\n=== Baseline-INSUFFICIENT -> wohin? (PoT-Kandidaten) ===")
for q in sorted(q for q, s in base_st.items() if s == "insuff"):
    print(f"  {q}: insuff -> {pot_st.get(q, '?')}")

bc = sum(1 for s in base_st.values() if s == "correct")
pc = sum(1 for s in pot_st.values() if s == "correct")
print(f"\ncorrect: base={bc} -> pot={pc}  (netto {pc - bc:+d})")
