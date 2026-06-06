"""Explorative Analyse: Welches Signal findet Gold am besten, und hilft eine
Gewichtung der RRF-Fusion? Liest die CSVs eines retrieval_debug-Runs (keine
Modelle, kein Rerun). Approximiert die Weighted-RRF-Raenge auf der Kandidaten-
menge (top-N aus pages.csv) ∪ (alle Gold-Seiten aus gold.csv), gruppiert pro
(qid, target_idx, doc_name) — das ist die isoliert gerankte retrieval unit.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

RUN = sys.argv[1] if len(sys.argv) > 1 else "data/results/retrieval_debug_20260605_170423"
SIGNALS = ["dense", "sparse", "bm25", "colbert"]
KS = [1, 3, 5, 10, 20]
RRF_K = 60


def pf(x):
    return None if x in ("", None) else float(x)


def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


gold = load_csv(f"{RUN}_gold.csv")
pages = load_csv(f"{RUN}_pages.csv")

# Nur erreichbare (gerankte) Gold-Seiten zaehlen fuer recall.
gold_ranked = [g for g in gold if g["reachable"] == "True"]
N = len(gold_ranked)
print(f"Run: {RUN}")
print(f"Gold-Seiten gesamt={len(gold)}  gerankt(reachable)={N}  "
      f"doc-not-retrieved={sum(1 for g in gold if g['in_retrieved_doc']!='True')}")


def ranks(rows, sig):
    return np.array([float(r[f"{sig}_rank"]) for r in rows if pf(r[f"{sig}_rank"])])


# ── 1. Reproduktion: recall@k pro Einzelsignal (Sanity vs summary.json) ──────
print("\n" + "=" * 78)
print("1. EINZELSIGNAL — Gold-Rang-Statistik (Reproduktion der summary.json)")
print("=" * 78)
hdr = f"{'Signal':<9}{'mean':>7}{'med':>5}" + "".join(f"{'r@'+str(k):>8}" for k in KS)
print(hdr)
print("-" * len(hdr))
for sig in SIGNALS + ["rrf"]:
    arr = ranks(gold_ranked, sig)
    rec = "".join(f"{np.mean(arr <= k):>8.3f}" for k in KS)
    print(f"{sig:<9}{arr.mean():>7.1f}{np.median(arr):>5.0f}{rec}")


# ── 2. Komplementaritaet: best-of-N (Oracle) ────────────────────────────────
# Fuer jede Gold-Seite der MINIMALE Rang ueber eine Signalmenge = was ein
# perfekter Signal-Selektor erreichen wuerde. Zeigt, ob ein Signal-Set genuegt.
print("\n" + "=" * 78)
print("2. KOMPLEMENTARITAET — Oracle best-of-N (min. Rang ueber Signalmenge)")
print("=" * 78)
combos = {
    "dense allein":          ["dense"],
    "colbert allein":        ["colbert"],
    "dense+colbert":         ["dense", "colbert"],
    "sparse+bm25 (lexik.)":  ["sparse", "bm25"],
    "ALLE 4 (oracle)":       SIGNALS,
}
hdr = f"{'Kombination':<22}" + "".join(f"{'r@'+str(k):>8}" for k in KS)
print(hdr)
print("-" * len(hdr))
best = {}
for name, sigs in combos.items():
    mins = []
    for g in gold_ranked:
        rs = [float(g[f"{s}_rank"]) for s in sigs if pf(g[f"{s}_rank"])]
        mins.append(min(rs) if rs else 1e9)
    arr = np.array(mins)
    best[name] = arr
    rec = "".join(f"{np.mean(arr <= k):>8.3f}" for k in KS)
    print(f"{name:<22}{rec}")

# Was bringen die lexikalischen Signale ZUSAETZLICH zu dense+colbert?
dc = best["dense+colbert"]
allf = best["ALLE 4 (oracle)"]
print("\nZusatznutzen sparse+bm25 ueber dense+colbert (oracle-Delta):")
for k in KS:
    gain = np.mean(allf <= k) - np.mean(dc <= k)
    print(f"  r@{k:<2}: +{gain:.3f}  ({int(round(gain*N))} Gold-Seiten)")


# ── 3. Einzigartige Rettungen durch die lexikalischen Signale ───────────────
# Gold-Seiten, die NUR sparse/bm25 in top-k bringen, waehrend dense UND colbert
# sie verfehlen (>k). Das ist der einzige Grund, sie ueberhaupt zu behalten.
print("\n" + "=" * 78)
print("3. EINZIGARTIGE RETTUNGEN — nur lexikalisch in top-k, dense&colbert raus")
print("=" * 78)
for k in [5, 10, 20]:
    lex_only = dense_only = both_dense = 0
    for g in gold_ranked:
        d = float(g["dense_rank"]); c = float(g["colbert_rank"])
        s = float(g["sparse_rank"]); b = float(g["bm25_rank"])
        dense_side = (d <= k) or (c <= k)
        lex_side = (s <= k) or (b <= k)
        if lex_side and not dense_side:
            lex_only += 1
        if dense_side and not lex_side:
            dense_only += 1
    print(f"  k={k:<3} nur lexik. rettet: {lex_only:>3} Gold-Seiten   |   "
          f"nur dense/colbert rettet: {dense_only:>3}")


# ── 4. Weighted-RRF-Simulation ──────────────────────────────────────────────
# Kandidaten pro (qid,target_idx,doc) = top-N(pages.csv) ∪ gold(gold.csv).
# Approx: nicht-Gold-Kandidaten ausserhalb top-N fehlen -> leicht optimistisch
# fuer Gold ausserhalb top-N (~7% der Faelle bei r@20). Baseline (1,1,1,1) wird
# gegen die echte summary-RRF-recall kalibriert.
print("\n" + "=" * 78)
print("4. WEIGHTED-RRF-SIMULATION  (Kandidaten = top-N ∪ gold pro doc)")
print("=" * 78)

groups = defaultdict(dict)  # (qid,tidx,doc) -> {(doc,page): row}
for r in pages:
    groups[(r["qid"], r["target_idx"], r["doc_name"])][(r["doc_name"], int(r["page"]))] = r
# Gold-Seiten als Kandidaten ergaenzen (falls ausserhalb top-N) + markieren.
gold_keys = set()
for g in gold_ranked:
    key = (g["qid"], g["target_idx"], g["doc_name"])
    pk = (g["doc_name"], int(g["page"]))
    groups[key].setdefault(pk, g)
    gold_keys.add((g["qid"], g["target_idx"], g["doc_name"], int(g["page"])))


def wscore(row, w):
    s = 0.0
    for wi, sig in zip(w, SIGNALS):
        rk = pf(row[f"{sig}_rank"])
        if wi and rk:
            s += wi / (RRF_K + rk)
    return s


def sim_recall(w):
    """Gibt recall@k-Dict ueber alle N gerankten Gold-Seiten zurueck."""
    hits = {k: 0 for k in KS}
    for (qid, tidx, doc), cand in groups.items():
        scored = sorted(cand.values(), key=lambda r: wscore(r, w), reverse=True)
        rank_of = {(r["doc_name"], int(r["page"])): i + 1
                   for i, r in enumerate(scored)}
        for (cdoc, cpage), rnk in rank_of.items():
            if (qid, tidx, doc, cpage) in gold_keys:
                for k in KS:
                    if rnk <= k:
                        hits[k] += 1
    return {k: hits[k] / N for k in KS}


weight_sets = {
    "(1,1,1,1) Baseline":      (1, 1, 1, 1),
    "(1,.5,.5,1) lex halb":    (1, .5, .5, 1),
    "(1,.25,.25,1) lex viertel":(1, .25, .25, 1),
    "(1,0,0,1) nur dense+cb":  (1, 0, 0, 1),
    "(2,1,1,2) dense/cb x2":   (2, 1, 1, 2),
    "(1,.5,.5,.8) cb leicht runter": (1, .5, .5, .8),
    "(1,0,0,0) nur dense":     (1, 0, 0, 0),
}
hdr = f"{'Gewichte (d,s,b,c)':<30}" + "".join(f"{'r@'+str(k):>8}" for k in KS)
print(hdr)
print("-" * len(hdr))
base = None
for name, w in weight_sets.items():
    rec = sim_recall(w)
    if base is None:
        base = rec
    deltas = "".join(f"{rec[k]:>8.3f}" for k in KS)
    print(f"{name:<30}{deltas}")

print(f"\n  (Kalibrierung: simulierte Baseline r@10={base[10]:.3f} vs "
      f"echte summary RRF r@10=0.785 — Differenz zeigt Approx-Fehler)")
print(f"  N={N} gerankte Gold-Seiten, RRF_K={RRF_K}")
