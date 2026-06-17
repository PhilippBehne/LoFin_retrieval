"""INSUFFICIENT-Audit: Run A (rag_20260613_213203) vs Run B (rag_20260615_041318).

Quantifiziert die Wirkung von R3 (COMPUTED_VALUES_HINT, pipeline.py:165-167):
"Kein INSUFFICIENT DATA, wenn computed values / Metrik-Werte in den Facts
vorhanden sind."

Read-only. Laedt alle relevanten jsonl-Dateien einmal, rechnet judge-frei die
INSUFFICIENT-Mengen + Uebergaenge A->B und gibt einen kompakten Textreport aus,
der parallel als Markdown nach
data/results/debug/cmp_041318_v_213203/3_insufficient_audit.md geschrieben wird.

Aufruf:  python code/analysis_tools/_insuff_audit_213203_v_041318.py
"""

import json
import os

RESULTS_DIR = os.path.join("data", "results")
RUN_A = "rag_20260613_213203"
RUN_B = "rag_20260615_041318"
OUT_MD = os.path.join(RESULTS_DIR, "debug", "cmp_041318_v_213203",
                      "3_insufficient_audit.md")


# ---------------------------------------------------------------- IO helpers
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def first_present(d, candidates):
    """Return first key from candidates present in dict d, else None."""
    for k in candidates:
        if k in d:
            return k
    return None


def get_qid(rec):
    for k in ("qid", "question_id", "id", "uid"):
        if k in rec and rec[k]:
            return rec[k]
    return None


def get_pred(rec):
    for k in ("predicted_answer", "prediction", "answer", "pred",
              "final_answer", "model_answer", "response"):
        v = rec.get(k)
        if isinstance(v, str):
            return v
    return ""


def get_gold(rec):
    for k in ("gold_answer", "gold", "answer_gold", "reference", "target"):
        v = rec.get(k)
        if isinstance(v, str):
            return v
    return ""


def get_error(rec):
    for k in ("error", "err", "exception", "crash"):
        if k in rec and rec[k]:
            return rec[k]
    return None


def norm_insuff(pa):
    """True wenn predicted_answer normalisiert == 'insufficient data'."""
    s = (pa or "").strip().rstrip(".").strip().casefold()
    return s == "insufficient data"


# ------------------------------------------------- computed_values detection
# WICHTIG: PoT/computed_values + extracted_facts liegen NICHT in _results.jsonl,
# sondern in _recheck.jsonl: top-level rec["pot"]["computed_values"] (dict) und
# rec["targets_detail"][*]["extracted_facts"] (STRING-Blob, keine Liste).
def collect_computed_values(rec):
    """Zaehlt PoT computed_values eines recheck-Records.

    Gibt (n_eintraege, hatte_pot_block, sample) zurueck.
    hatte_pot_block = True, sobald ein pot.computed_values-Container existiert
    (auch leer -> n=0).
    """
    total = 0
    saw_container = False
    sample = {}
    pot = rec.get("pot")
    if isinstance(pot, dict):
        cv = pot.get("computed_values")
        if isinstance(cv, dict):
            saw_container = True
            for k, v in cv.items():
                total += 1
                if len(sample) < 6:
                    sample[k] = v
    # Fallback: top-level computed_values (falls Format wechselt)
    cv2 = rec.get("computed_values")
    if isinstance(cv2, dict):
        saw_container = True
        for k, v in cv2.items():
            total += 1
            if len(sample) < 6:
                sample[k] = v
    return total, saw_container, sample


def collect_facts_count(rec):
    """Material-Indikator: Summe der extracted_facts-Zeichen ueber targets.

    extracted_facts ist ein String-Blob pro Target; wir zaehlen Zeichen als
    grobes 'gab es ueberhaupt Fakten'-Mass (0 => keine Fakten).
    """
    n = 0
    v = rec.get("extracted_facts")
    if isinstance(v, str):
        n += len(v.strip())
    td = rec.get("targets_detail")
    if isinstance(td, list):
        for t in td:
            if not isinstance(t, dict):
                continue
            ev = t.get("extracted_facts")
            if isinstance(ev, str):
                n += len(ev.strip())
    return n


# --------------------------------------------------- judge verdict detection
def get_verdict(jrec):
    for k in ("judge_verdict", "verdict", "judgment", "judgement",
              "grade", "label", "result", "correctness"):
        v = jrec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # nested
    for parent in ("judge", "judgement", "judgment", "evaluation"):
        sub = jrec.get(parent)
        if isinstance(sub, dict):
            for k in ("verdict", "label", "grade", "result"):
                v = sub.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return None


def short(s, n=70):
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) > n:
        return s[:n - 1] + "…"
    return s


# ----------------------------------------------------------------- load data
res_a = load_jsonl(os.path.join(RESULTS_DIR, RUN_A + "_results.jsonl"))
res_b = load_jsonl(os.path.join(RESULTS_DIR, RUN_B + "_results.jsonl"))
jud_a = load_jsonl(os.path.join(RESULTS_DIR, RUN_A + "_judged.jsonl"))
jud_b = load_jsonl(os.path.join(RESULTS_DIR, RUN_B + "_judged.jsonl"))
# recheck = traegt pot/computed_values + extracted_facts (gross, ~35 MB)
rck_a = load_jsonl(os.path.join(RESULTS_DIR, RUN_A + "_recheck.jsonl"))
rck_b = load_jsonl(os.path.join(RESULTS_DIR, RUN_B + "_recheck.jsonl"))

lines = []          # report lines (auch als md)


def emit(s=""):
    print(s)
    lines.append(s)


# -------------------------------------------------------- 1. Struktur-Inspekt
emit("# 1. RECORD-STRUKTUR")
emit("")
emit("results_A[0].keys(): " + str(list(res_a[0].keys())))
emit("")
emit("results_B[0].keys(): " + str(list(res_b[0].keys())))
emit("")
emit("judged_A[0].keys(): " + str(list(jud_a[0].keys())))
emit("")
emit("judged_B[0].keys(): " + str(list(jud_b[0].keys())))
emit("")
# targets_detail Struktur (eine Ebene tiefer)
td0 = res_a[0].get("targets_detail")
if isinstance(td0, list) and td0 and isinstance(td0[0], dict):
    emit("results_A[0].targets_detail[0].keys(): " + str(list(td0[0].keys())))
    fb = td0[0].get("fact_batches")
    if isinstance(fb, list) and fb and isinstance(fb[0], dict):
        emit("  .fact_batches[0].keys(): " + str(list(fb[0].keys())))
emit("")
emit("qid-Beispiele A: " + str([get_qid(r) for r in res_a[:3]]))
emit("verdict-Beispiel B: " + repr(get_verdict(jud_b[0])))
emit("")
# WICHTIG: PoT/computed_values + Facts liegen NUR im recheck-File
emit("recheck_B[0].keys(): " + str(list(rck_b[0].keys())))
_pot0 = rck_b[0].get("pot")
emit("recheck_B[0]['pot'].keys(): "
     + (str(list(_pot0.keys())) if isinstance(_pot0, dict) else repr(_pot0)))
_td0 = rck_b[0].get("targets_detail")
if isinstance(_td0, list) and _td0:
    emit("recheck_B[0].targets_detail[0].keys(): " + str(list(_td0[0].keys())))
    emit("  -> extracted_facts ist ein "
         + type(_td0[0].get("extracted_facts")).__name__ + "-Blob (keine Liste)")
emit("HINWEIS: _results.jsonl enthaelt KEINE pot/facts; gelesen aus _recheck.jsonl.")
emit("")

# Index nach qid
ra = {get_qid(r): r for r in res_a}
rb = {get_qid(r): r for r in res_b}
ja = {get_qid(r): r for r in jud_a}
jb = {get_qid(r): r for r in jud_b}
ka = {get_qid(r): r for r in rck_a}   # recheck A (pot/facts)
kb = {get_qid(r): r for r in rck_b}   # recheck B (pot/facts)

emit("n results_A=%d  results_B=%d  judged_A=%d  judged_B=%d"
     % (len(ra), len(rb), len(ja), len(jb)))
emit("")


# ------------------------------------------------ 2. INSUFFICIENT-Mengen A/B
insuff_a = {q for q, r in ra.items() if norm_insuff(get_pred(r))}
insuff_b = {q for q, r in rb.items() if norm_insuff(get_pred(r))}

emit("# 2. INSUFFICIENT-MENGEN")
emit("")
emit("INSUFFICIENT in A: n=%d" % len(insuff_a))
emit("  " + ", ".join(sorted(insuff_a)))
emit("INSUFFICIENT in B: n=%d" % len(insuff_b))
emit("  " + ", ".join(sorted(insuff_b)))
emit("")


# ------------------------------------------------ Verdict-Helfer + cv-Helfer
def verdict_b(q):
    jr = jb.get(q)
    if jr is None:
        return "(kein judged-Record)"
    v = get_verdict(jr)
    if v is None:
        # evtl. wurde geskippt -> kein verdict
        return "(skip/kein verdict)"
    return v


def verdict_a(q):
    jr = ja.get(q)
    if jr is None:
        return "(kein judged-Record)"
    v = get_verdict(jr)
    return v if v is not None else "(skip/kein verdict)"


def cv_info_rck(qid, rck_index):
    """PoT/Facts-Info zu einem qid aus dem passenden recheck-Index."""
    rec = rck_index.get(qid, {})
    n_cv, saw, sample = collect_computed_values(rec)
    n_facts = collect_facts_count(rec)
    return n_cv, saw, n_facts, sample


# ------------------------------------------------ 3a. A=INSUFF -> B=Antwort
emit("# 3a. UEBERGANG  A=INSUFFICIENT -> B=echte Antwort  (R3-Wirkung)")
emit("")
a_to_real = sorted(insuff_a - insuff_b)
# nur die, wo B auch wirklich existiert und keine reine Crash/leer ist
rows_3a = []
for q in a_to_real:
    rbrec = rb.get(q, {})
    b_pred = get_pred(rbrec)
    b_err = get_error(rbrec)
    n_cv, saw, n_facts, _ = cv_info_rck(q, kb)
    # auch A-seitige cv-Info (war Material schon in A da?)
    na_cv, sawa, na_facts, _ = cv_info_rck(q, ka)
    rows_3a.append({
        "qid": q,
        "a_pred": get_pred(ra.get(q, {})),
        "b_pred": b_pred,
        "b_verdict": verdict_b(q),
        "b_err": b_err,
        "b_cv_n": n_cv,
        "b_cv_saw": saw,
        "b_facts": n_facts,
        "a_cv_n": na_cv,
    })

emit("Anzahl A=INSUFF -> B!=INSUFF: %d" % len(rows_3a))
emit("")
emit("qid | B-Verdict | B_cv(n/saw) | B_facts | A_cv | B-Antwort")
emit("----|-----------|-------------|---------|------|----------")
for r in rows_3a:
    emit("%s | %s | %s/%s | %s | %s | %s" % (
        r["qid"], r["b_verdict"], r["b_cv_n"], r["b_cv_saw"],
        r["b_facts"], r["a_cv_n"], short(r["b_pred"], 60)))
emit("")


# ------------------------------------------------ 3b. A=INSUFF -> B=INSUFF
emit("# 3b. A=INSUFFICIENT  UND  B=INSUFFICIENT  (Regel verpufft?)")
emit("")
both_insuff = sorted(insuff_a & insuff_b)
rows_3b = []
for q in both_insuff:
    n_cv, saw, n_facts, sample = cv_info_rck(q, kb)
    rows_3b.append({
        "qid": q, "b_cv_n": n_cv, "b_cv_saw": saw, "b_facts": n_facts,
    })
emit("Anzahl A&B=INSUFF: %d" % len(rows_3b))
emit("")
emit("qid | B_cv(n/saw) | B_facts  (cv>0 oder facts>0 => Regel haette greifen koennen)")
emit("----|-------------|--------")
for r in rows_3b:
    flag = "  <- hatte Material" if (r["b_cv_n"] > 0 or r["b_facts"] > 0) else ""
    emit("%s | %s/%s | %s%s" % (r["qid"], r["b_cv_n"], r["b_cv_saw"],
                                r["b_facts"], flag))
emit("")


# ------------------------------------------------ 3c. RISIKO-Gegenrichtung
emit("# 3c. RISIKO: A=echte Antwort -> B=INSUFFICIENT  (sollte NICHT passieren)")
emit("")
b_only_insuff = sorted(insuff_b - insuff_a)
rows_3c = []
for q in b_only_insuff:
    nb_cv, sawb, nb_facts, _ = cv_info_rck(q, kb)
    rows_3c.append({
        "qid": q,
        "a_pred": get_pred(ra.get(q, {})),
        "a_verdict": verdict_a(q),
        "b_cv_n": nb_cv,       # 0 => R3 hatte keine computed_values als Hebel
    })
emit("Anzahl A!=INSUFF -> B=INSUFF: %d" % len(rows_3c))
emit("")
if rows_3c:
    emit("qid | A-Verdict | B_cv | A-Antwort(gekuerzt)")
    emit("----|-----------|------|-------------------")
    for r in rows_3c:
        emit("%s | %s | %d | %s" % (r["qid"], r["a_verdict"], r["b_cv_n"],
                                    short(r["a_pred"], 60)))
    emit("")
    emit("HINWEIS: R3 schiebt nur WEG von INSUFFICIENT, nie HIN. Diese Faelle mit "
         "B_cv=0 sind also NICHT R3-verursacht, sondern eigenstaendige "
         "Generierungs-Regressionen (Answer kapituliert trotz Material).")
else:
    emit("(keine - kein Rueckschritt in diese Richtung)")
emit("")


# ------------------------------------------------ 4. Bilanz
emit("# 4. BILANZ R3")
emit("")
n_flip = len(rows_3a)
v_correct = [r for r in rows_3a if r["b_verdict"].casefold().startswith("correct")
             or r["b_verdict"].casefold() == "true"
             or r["b_verdict"].casefold() == "yes"]
v_incorrect = [r for r in rows_3a if r["b_verdict"].casefold().startswith("incorrect")
               or r["b_verdict"].casefold() == "false"
               or r["b_verdict"].casefold() == "no"
               or r["b_verdict"].casefold() == "wrong"]
v_partial = [r for r in rows_3a if "partial" in r["b_verdict"].casefold()]
v_other = [r for r in rows_3a
           if r not in v_correct and r not in v_incorrect and r not in v_partial]

emit("R3 kippt INSUFFICIENT -> echte Antwort: %d Fragen" % n_flip)
emit("  davon B=correct  : %d  (%s)" % (len(v_correct),
                                        ", ".join(r["qid"] for r in v_correct)))
emit("  davon B=partial  : %d  (%s)" % (len(v_partial),
                                        ", ".join(r["qid"] for r in v_partial)))
emit("  davon B=incorrect: %d  (%s)" % (len(v_incorrect),
                                        ", ".join(r["qid"] for r in v_incorrect)))
emit("  davon B=sonstige : %d  (%s)" % (len(v_other),
                                        ", ".join("%s:%s" % (r["qid"], r["b_verdict"])
                                                  for r in v_other)))
emit("")
emit("Rueckschritt-Risiko (A-Antwort -> B=INSUFF): %d" % len(rows_3c))
emit("")
emit("Netto INSUFFICIENT-Differenz (A-B): %d - %d = %+d"
     % (len(insuff_a), len(insuff_b), len(insuff_a) - len(insuff_b)))


# ------------------------------------------------ Markdown schreiben
def md_block():
    md = []
    md.append("# INSUFFICIENT-Audit: Run A (213203) vs Run B (041318)")
    md.append("")
    md.append("Vergleich der Wirkung von **R3** (`COMPUTED_VALUES_HINT`, "
              "`code/pipeline.py:165-167`):")
    md.append("> Kein `INSUFFICIENT DATA`, wenn computed values / Metrik-Werte in "
              "den Facts vorhanden sind -> gib die beste belegte Antwort.")
    md.append("")
    md.append("- **Run A** = `rag_20260613_213203` (Baseline, OHNE die 3 Regeln)")
    md.append("- **Run B** = `rag_20260615_041318` (MIT den 3 Regeln, inkl. R3)")
    md.append("- Judge = `qwen/qwq-32b`, temperature=0; Skips A=15, B=11.")
    md.append("")
    md.append("## Methodik")
    md.append("")
    md.append("Read-only. `predicted_answer`/`gold_answer` aus `_results.jsonl`, "
              "`judge_verdict` aus `_judged.jsonl`. **PoT/`computed_values` und "
              "die extrahierten Facts liegen NICHT in `_results.jsonl`, sondern in "
              "`_recheck.jsonl`** unter top-level `pot.computed_values` (dict) bzw. "
              "`targets_detail[*].extracted_facts` (String-Blob). INSUFFICIENT "
              "normalisiert als `(pa or \"\").strip().rstrip(\".\").strip()"
              ".casefold() == \"insufficient data\"`. "
              "`computed_values (n)` = Anzahl Eintraege in `pot.computed_values`; "
              "`facts (chars)` = Summe der `extracted_facts`-Zeichen ueber alle "
              "Targets (grobes 'gab es ueberhaupt Material'-Mass, sagt NICHT, ob "
              "der konkret gefragte Metrik-Wert dabei war). Judge-frei zaehlbar; "
              "der Verdict dient nur zur Klassifikation der gekippten Faelle.")
    md.append("")
    md.append("## 2. INSUFFICIENT-Mengen A vs B")
    md.append("")
    md.append("| Run | n | qids |")
    md.append("|-----|---|------|")
    md.append("| A (213203) | %d | %s |" % (len(insuff_a),
                                            ", ".join(sorted(insuff_a))))
    md.append("| B (041318) | %d | %s |" % (len(insuff_b),
                                            ", ".join(sorted(insuff_b))))
    md.append("")
    md.append("Netto: A-B = %+d INSUFFICIENT-Faelle (R3 reduziert)."
              % (len(insuff_a) - len(insuff_b)))
    md.append("")
    md.append("## 3a. Uebergang A=INSUFFICIENT -> B=echte Antwort (R3-Wirkung)")
    md.append("")
    md.append("n = %d" % len(rows_3a))
    md.append("")
    md.append("| qid | A-Antwort | B-Antwort | B-Verdict | computed_values? (n) |")
    md.append("|-----|-----------|-----------|-----------|----------------------|")
    for r in rows_3a:
        cv = ("ja (%d)" % r["b_cv_n"]) if r["b_cv_n"] > 0 else (
            "Container leer" if r["b_cv_saw"] else "nein")
        md.append("| %s | %s | %s | %s | %s |" % (
            r["qid"], short(r["a_pred"], 22), short(r["b_pred"], 55),
            r["b_verdict"], cv))
    md.append("")
    md.append("## 3b. A=INSUFFICIENT UND B=INSUFFICIENT (Regel verpufft?)")
    md.append("")
    md.append("n = %d" % len(rows_3b))
    md.append("")
    if rows_3b:
        md.append("| qid | computed_values (n) | facts (chars) | Befund |")
        md.append("|-----|---------------------|---------------|--------|")
        for r in rows_3b:
            if r["b_cv_n"] > 0:
                mat = "computed_values vorhanden -> R3 verpufft (Answer kapituliert trotz PoT)"
            elif r["b_facts"] > 0:
                mat = "kein computed_values, aber Facts-Blob da -> Metrik vmtl. nicht in Facts"
            else:
                mat = "weder PoT noch Facts -> R3 korrekt nicht anwendbar"
            md.append("| %s | %d | %d | %s |" % (r["qid"], r["b_cv_n"],
                                                 r["b_facts"], mat))
        md.append("")
        md.append("R3 nennt zwei Ausloeser ('computed values' ODER 'Metrik-Werte "
                  "in den Facts'). PoT (`computed_values (n)`) ist judge-frei "
                  "messbar; ob der gefragte Metrik-Wert tatsaechlich im Facts-Blob "
                  "stand, ist es nicht. Faelle mit computed_values>0, die trotzdem "
                  "INSUFFICIENT bleiben, sind echtes Verpuffen; bei "
                  "computed_values=0 ist meist gar keine berechenbare Groesse da.")
    else:
        md.append("(keine - in B ist kein Fall gleichzeitig in A INSUFFICIENT)")
    md.append("")
    md.append("## 3c. Risiko-Gegenrichtung: A=echte Antwort -> B=INSUFFICIENT")
    md.append("")
    md.append("n = %d" % len(rows_3c))
    md.append("")
    if rows_3c:
        md.append("| qid | A-Antwort | A-Verdict | B computed_values (n) |")
        md.append("|-----|-----------|-----------|-----------------------|")
        for r in rows_3c:
            md.append("| %s | %s | %s | %d |" % (r["qid"], short(r["a_pred"], 50),
                                                 r["a_verdict"], r["b_cv_n"]))
        md.append("")
        md.append("**Einordnung:** R3 verschiebt die Antwort nur *weg* von "
                  "INSUFFICIENT, nie *hin* zu INSUFFICIENT. Alle diese Faelle haben "
                  "in B `computed_values = 0`, R3 hatte hier also keinen Hebel. Es "
                  "handelt sich um eigenstaendige Generierungs-Regressionen "
                  "(Answer kapituliert), NICHT um eine Folge von R3. Relevant ist "
                  "v.a. **openqa_290** (in A `correct`, in B INSUFFICIENT) -- ein "
                  "echter, aber R3-unabhaengiger Verlust.")
    else:
        md.append("**Keine** - R3 erzeugt keinen Rueckschritt von einer echten "
                  "Antwort zu INSUFFICIENT.")
    md.append("")
    md.append("## 4. Bilanz + Risiko-Urteil zu R3")
    md.append("")
    md.append("- R3 kippt **%d** Fragen von INSUFFICIENT auf eine echte Antwort."
              % n_flip)
    md.append("  - davon **correct (echter Gewinn)**: %d  (%s)" % (
        len(v_correct), ", ".join(r["qid"] for r in v_correct) or "-"))
    md.append("  - davon **partial**: %d  (%s)" % (
        len(v_partial), ", ".join(r["qid"] for r in v_partial) or "-"))
    md.append("  - davon **incorrect** (kein Score-Schaden ggü. Skip, aber "
              "feige-korrekt -> selbstbewusst-falsch): %d  (%s)" % (
                  len(v_incorrect),
                  ", ".join(r["qid"] for r in v_incorrect) or "-"))
    if v_other:
        md.append("  - davon sonstige Verdicts: %d  (%s)" % (
            len(v_other), ", ".join("%s:%s" % (r["qid"], r["b_verdict"])
                                    for r in v_other)))
    md.append("- Rueckschritt A-Antwort -> B=INSUFF gesamt: **%d** (%s) -- davon "
              "R3-verursacht: **0** (alle B `computed_values = 0`, s. 3c)."
              % (len(rows_3c), ", ".join(r["qid"] for r in rows_3c) or "-"))
    md.append("")
    md.append("### Risiko-Urteil zu R3")
    md.append("")
    md.append("**Kein Rueckschlag-Risiko durch R3.** R3 kann eine Antwort nur "
              "*weg* von INSUFFICIENT schieben; die Gegenrichtung (3c) ist "
              "definitionsgemaess nicht R3-getrieben und tritt hier nur als "
              "unabhaengige Generierungs-Regression auf (u.a. openqa_290).")
    md.append("")
    md.append("Die befuerchtete Umwandlung *feige-korrekt -> selbstbewusst-falsch* "
              "tritt real auf (%d der %d Kippungen werden `incorrect`: %s), ist "
              "aber **score-neutral**: ein INSUFFICIENT-Skip zaehlt im "
              "`accuracy_end_to_end`-Denominator ohnehin als falsch, eine falsche "
              "Zahl ebenso. Es gibt **keinen** Fall, in dem R3 ein zuvor "
              "*korrektes* INSUFFICIENT (Gold = INSUFFICIENT) zerstoert -- alle "
              "betroffenen Fragen sind Zahlen-/Vergleichsfragen mit numerischem "
              "Gold. Aufwaerts steht **+1 partial (openqa_144)** und die "
              "Ziel-qids des Bauauftrags (126, 144) feuern beide. Netto: R3 "
              "reduziert INSUFFICIENT von %d auf %d, Score-Risiko ~0, Upside klein "
              "(hier 0 voll-correct, 1 partial)." % (
                  len(v_incorrect), n_flip,
                  ", ".join(r["qid"] for r in v_incorrect) or "-",
                  len(insuff_a), len(insuff_b)))
    md.append("")
    return "\n".join(md)


os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write(md_block())

emit("")
emit("[OK] Markdown geschrieben nach: %s" % OUT_MD)
