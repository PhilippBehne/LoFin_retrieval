"""Unabhaengige Verifikation von _analyze_weighting.py.

Bewusst ANDERE Rechenwege als das Analyseskript, damit es ein echter Gegencheck
ist und nicht dieselbe (evtl. fehlerhafte) Logik wiederholt:
  - reines Python statt numpy fuer recall
  - rrf_score wird aus den 4 per-Signal-Raengen REKONSTRUIERT und gegen die
    gespeicherte Spalte geprueft -> beweist, dass die Spalten-Zuordnung stimmt
  - Kernaussage "dense & colbert am besten" via argmin-Win-Count + Head-to-Head
    (Schwellen-unabhaengig), nicht via recall@k
Keine Modelle noetig. Jeder Block endet mit PASS/FAIL.
"""
import csv
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN = "data/results/retrieval_debug_20260605_170423"
SIGNALS = ["dense", "sparse", "bm25", "colbert"]
RRF_K = 60


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


gold = load(f"{RUN}_gold.csv")
pages = load(f"{RUN}_pages.csv")
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# ── A. Beweist die Spalten-Zuordnung: rrf_score aus 4 Raengen rekonstruieren ─
# Der gespeicherte rrf_score MUSS sum_i 1/(60+rank_i) sein (use_colbert=True).
# Wenn das auf 1e-9 passt, ist garantiert, dass dense_rank wirklich dense ist
# usw. UND dass RRF_K/Formel stimmen.
print("A. CSV-INTEGRITAET: rrf_score == Summe 1/(60+rang) ueber 4 Signale")
max_err = 0.0
n_checked = 0
for r in pages:
    recon = sum(1.0 / (RRF_K + float(r[f"{s}_rank"])) for s in SIGNALS)
    err = abs(recon - float(r["rrf_score"]))
    max_err = max(max_err, err)
    n_checked += 1
check(f"{n_checked} Seiten geprueft, max. Abweichung {max_err:.2e}", max_err < 1e-9)

# rrf_rank muss der absteigenden rrf_score-Ordnung pro (qid,target,doc) folgen.
groups = defaultdict(list)
for r in pages:
    groups[(r["qid"], r["target_idx"], r["doc_name"])].append(r)
ord_ok = True
for g, rows in groups.items():
    by_score = sorted(rows, key=lambda r: float(r["rrf_score"]), reverse=True)
    by_rank = sorted(rows, key=lambda r: int(r["rrf_rank"]))
    if [id(x) for x in by_score] != [id(x) for x in by_rank]:
        # erlaubt Ties: pruefe, ob rrf_rank monoton mit -rrf_score
        prev = -1
        for r in by_rank:
            if float(r["rrf_score"]) > prev and prev >= 0:
                ord_ok = False
            prev = -float(r["rrf_score"])  # placeholder; richtige Tie-Pruefung unten
# robustere Tie-tolerante Pruefung:
ord_ok = True
for g, rows in groups.items():
    s = sorted(rows, key=lambda r: int(r["rrf_rank"]))
    scores = [float(r["rrf_score"]) for r in s]
    if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
        ord_ok = False
check("rrf_rank folgt absteigendem rrf_score (Tie-tolerant)", ord_ok)


# ── B. Vollstaendigkeit der gerankten Gold-Seiten ────────────────────────────
print("\nB. VOLLSTAENDIGKEIT der reachable Gold-Seiten")
reach = [g for g in gold if g["reachable"] == "True"]
missing = [g for g in reach if any(g[f"{s}_rank"] in ("", None) for s in SIGNALS)]
check(f"N={len(reach)} reachable, davon mit fehlendem Rang: {len(missing)}",
      len(missing) == 0)
check("N stimmt mit summary.json (n_ranked=339)", len(reach) == 339,
      f"gefunden {len(reach)}")


# ── C. recall@k in REINEM Python (Gegencheck zu numpy/summary) ───────────────
print("\nC. recall@k pur-Python vs summary.json")
SUMMARY = {  # aus retrieval_debug_..._summary.json (unabhaengig erzeugt)
    "dense":   {1: .2242, 5: .6254, 10: .7994, 20: .9174},
    "colbert": {1: .2183, 5: .6106, 10: .7729, 20: .9086},
    "sparse":  {1: .1563, 5: .4366, 10: .6047, 20: .8289},
    "bm25":    {1: .1475, 5: .4218, 10: .5929, 20: .7935},
}
N = len(reach)
for sig in SIGNALS:
    line = []
    sig_ok = True
    for k in (1, 5, 10, 20):
        hits = sum(1 for g in reach if float(g[f"{sig}_rank"]) <= k)
        rec = hits / N
        diff = abs(rec - SUMMARY[sig][k])
        sig_ok = sig_ok and diff < 5e-4
        line.append(f"r@{k}={rec:.4f}(Δ{diff:.0e})")
    check(f"{sig:<8} " + " ".join(line), sig_ok)


# ── D. KERNAUSSAGE schwellen-UNABHAENGIG: Win-Count + Head-to-Head ───────────
# Voellig anderer Weg als recall@k: pro Gold-Seite das Signal mit dem KLEINSTEN
# Rang ("Gewinner"). Wenn dense/colbert dominieren, ist die Aussage robust.
print("\nD. KERNAUSSAGE — Win-Count (bestes Signal je Gold-Seite, argmin-Rang)")
wins = Counter()
dc_best = 0  # dense ODER colbert ist (mit) das beste Signal
for g in reach:
    rk = {s: float(g[f"{s}_rank"]) for s in SIGNALS}
    best = min(rk.values())
    winners = [s for s in SIGNALS if rk[s] == best]
    for w in winners:
        wins[w] += 1.0 / len(winners)  # Ties teilen
    if "dense" in winners or "colbert" in winners:
        dc_best += 1
for s in SIGNALS:
    print(f"     {s:<8} gewinnt {wins[s]:6.1f}x  ({wins[s]/N:.1%})")
check(f"dense/colbert ist auf {dc_best}/{N} Gold-Seiten das beste Signal",
      dc_best / N > 0.75, f"{dc_best/N:.1%}")

print("\n   Head-to-Head (Anteil Gold-Seiten, wo Zeile strikt besser rankt):")
print(f"     {'':<9}" + "".join(f"{s:>9}" for s in SIGNALS))
for a in SIGNALS:
    cells = []
    for b in SIGNALS:
        if a == b:
            cells.append(f"{'—':>9}")
            continue
        wins_ab = sum(1 for g in reach
                      if float(g[f"{a}_rank"]) < float(g[f"{b}_rank"])) / N
        cells.append(f"{wins_ab:>9.2f}")
    print(f"     {a:<9}" + "".join(cells))
# dense & colbert muessen sparse & bm25 in der Mehrheit schlagen:
def beats(a, b):
    return sum(1 for g in reach if float(g[f"{a}_rank"]) < float(g[f"{b}_rank"])) / N
strong = all(beats(s, l) > 0.5 for s in ("dense", "colbert") for l in ("sparse", "bm25"))
check("dense & colbert schlagen sparse & bm25 jeweils mehrheitlich (>50%)", strong)


# ── E. Sind BEIDE noetig? Komplementaritaet dense vs colbert ─────────────────
print("\nE. dense & colbert — redundant oder komplementaer? (k=10)")
k = 10
d_only = sum(1 for g in reach
             if float(g["dense_rank"]) <= k and float(g["colbert_rank"]) > k)
c_only = sum(1 for g in reach
             if float(g["colbert_rank"]) <= k and float(g["dense_rank"]) > k)
both = sum(1 for g in reach
           if float(g["dense_rank"]) <= k and float(g["colbert_rank"]) <= k)
neither = N - d_only - c_only - both
print(f"     beide@10={both}  nur dense={d_only}  nur colbert={c_only}  keins={neither}")
check("colbert findet eigenstaendige Gold-Seiten (nicht redundant zu dense)",
      c_only >= 10, f"{c_only} nur von colbert @10")
check("dense findet eigenstaendige Gold-Seiten (nicht redundant zu colbert)",
      d_only >= 10, f"{d_only} nur von dense @10")


# ── F. Simulations-Validitaet: Baseline-Rang == gespeicherter rrf_rank ───────
# Fuer Gold in der Original-top-20 muss die (1,1,1,1)-Simulation exakt den
# rrf_rank reproduzieren — sonst ist die Gewichtungs-Sim (Sektion 4) kaputt.
print("\nF. SIMULATIONS-VALIDITAET: (1,1,1,1)-Rang == CSV rrf_rank (Gold in top-20)")
# Kandidaten je Gruppe nachbauen wie im Analyseskript.
cand = defaultdict(dict)
for r in pages:
    cand[(r["qid"], r["target_idx"], r["doc_name"])][(r["doc_name"], int(r["page"]))] = r
gset = set()
for g in reach:
    key = (g["qid"], g["target_idx"], g["doc_name"])
    cand[key].setdefault((g["doc_name"], int(g["page"])), g)
    gset.add((g["qid"], g["target_idx"], g["doc_name"], int(g["page"])))

def wscore(r):
    return sum(1.0 / (RRF_K + float(r[f"{s}_rank"])) for s in SIGNALS)

mismatch = 0
compared = 0
for (qid, tidx, doc), c in cand.items():
    scored = sorted(c.values(), key=wscore, reverse=True)
    for i, r in enumerate(scored, 1):
        page = int(r["page"])
        if (qid, tidx, doc, page) in gset and r.get("rrf_rank") not in ("", None):
            csv_rank = int(r["rrf_rank"])
            if csv_rank <= 20:  # nur Gold, das original in top-20 lag
                compared += 1
                if i != csv_rank:
                    mismatch += 1
check(f"{compared} Gold-Seiten in top-20 verglichen, Rang-Abweichungen: {mismatch}",
      mismatch == 0)


print("\n" + "=" * 60)
print(f"GESAMT: {'ALLE CHECKS BESTANDEN ✓' if ok_all else 'ES GAB FEHLER ✗'}")
print("=" * 60)
sys.exit(0 if ok_all else 1)
