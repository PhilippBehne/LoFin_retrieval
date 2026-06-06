"""Qualitative Stichprobe aus gold.csv: echte Gold-Seiten + echte Fragen, mit
den per-Signal-Raengen direkt aus der CSV. Zum manuellen Reingucken — bestaetigt,
dass _analyze_weighting.py reale Zeilen verarbeitet hat. Unvoreingenommen:
openqa_11 (vom Nutzer markiert) + jede 30. reachable Gold-Seite.
"""
import csv
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN = "data/results/retrieval_debug_20260605_170423"
SIG = ["dense", "sparse", "bm25", "colbert"]

gold = list(csv.DictReader(open(f"{RUN}_gold.csv", encoding="utf-8")))
qa = {}
for line in open("data/qa/secqa_test_train.jsonl", encoding="utf-8"):
    d = json.loads(line)
    qa[d["qid"]] = d

reach = [g for g in gold if g["reachable"] == "True"]

# Deterministische, verteilte Stichprobe + die markierte openqa_11.
picks = [g for g in reach if g["qid"] == "openqa_11"]
picks += [reach[i] for i in range(0, len(reach), 30)
          if reach[i]["qid"] != "openqa_11"]

print(f"Stichprobe: {len(picks)} Gold-Seiten von {len(reach)} reachable\n")
print("Lesart: r = Rang des Signals fuer DIESE Gold-Seite (1 = perfekt). "
      "★ = bestes Signal. RRF-Cutoff in Produktion = top-10.\n")

dc_best = top10 = 0
for g in picks:
    ranks = {s: int(g[f"{s}_rank"]) for s in SIG}
    best = min(ranks.values())
    winners = {s for s, r in ranks.items() if r == best}
    if winners & {"dense", "colbert"}:
        dc_best += 1
    rrf = int(g["rrf_rank"])
    in10 = rrf <= 10
    top10 += in10

    q = qa.get(g["qid"], {})
    question = (q.get("question", "?"))[:78]
    answer = str(q.get("answer", "?"))[:78]

    cells = "  ".join(
        f"{('★' if s in winners else ' ')}{s} r{ranks[s]:<3}" for s in SIG
    )
    flag = "✓ top-10" if in10 else "✗ ausserhalb top-10"
    print(f"{g['qid']:<11} {g['company'][:22]:<22} {g['doc_name']:<16} "
          f"p.{g['page']:<4} RRF r{rrf:<3} [{flag}]")
    print(f"   Q:    {question}")
    print(f"   Gold: {answer}")
    print(f"   {cells}")
    # Anschaulich: wo liegt RRF gegenueber dem besten Einzelsignal?
    if rrf > best:
        print(f"   → ungewichtete RRF (r{rrf}) schlechter als bestes Signal "
              f"(r{best}) — lexikal. Signale ziehen runter")
    print()

print("─" * 70)
print(f"In {dc_best}/{len(picks)} Stichproben ist dense oder colbert das beste Signal.")
print(f"In {top10}/{len(picks)} bringt die (ungewichtete) RRF die Gold-Seite in top-10.")
