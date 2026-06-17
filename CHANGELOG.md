# Changelog

Nachvollziehbare Historie der Code-Änderungen an der SecQA-RAG-Pipeline.
Neueste Einträge oben. Zeitangaben lokal (Europe/Berlin).

---

## 2026-06-16 — `think` → `reasoning_effort` (toter Param ersetzt, per-Schritt-Steuerung)

**Was:** Der `think`-Bool in `code/pipeline.py` wurde von LM Studio still ignoriert (Memory
`lm-studio-think-param-dead`, am 15.06. bewiesen) — alle bisherigen Läufe liefen real im
„alles denkt"-Modus. Ersetzt durch echtes per-Schritt-`reasoning_effort`:
- **Config (`pipeline.py:61`)**: `REASONING_FACT="none"` · `REASONING_COMPUTE="high"` ·
  `REASONING_ANSWER="high"`. Bei gemma-4-31b ist Reasoning binär (`none`=aus, alles andere=an,
  keine Stufe; `off`/`on` werfen HTTP 400) — „high" liest sich nur als „an".
- **`llm_call`**: Parameter `think` → `reasoning`; `body["think"]` → `body["reasoning_effort"]`
  (nur wenn ≠ None). Return: `think`-Feld jetzt aus `message.reasoning_content` (Fallback
  `extract_think` für etwaige `<think>`-Tags); neues Feld `reasoning_tokens` aus
  `usage.completion_tokens_details.reasoning_tokens`.
- **Drei Aufrufe** auf die Konstanten umgestellt (`extract_facts`/`generate_answer`/`compute_values`).
- **`reasoning_tokens` additiv geloggt**: `fact_batches[].reasoning_tokens`, `pot.reasoning_tokens`,
  `answer_reasoning_tokens` (alte Result-Files bleiben lesbar).
- `CLAUDE.md` korrigiert (der Satz „fact extraction runs think=False; answer generation think=True"
  war faktisch falsch) + `reasoning_tokens` im Result-Format-Absatz ergänzt.
- `pipeline_resume.py`: kein Edit nötig (referenziert `think`/`reasoning` nicht, erbt via Import) — per grep bestätigt.

**Warum:** Was im Code steht, soll auch passieren. Non-thinking Fact-Extraktion (mechanisch) spart
Speed + n_ctx; Compute/Answer behalten Reasoning. Mechanismus vorab grün verifiziert
(`_test_reasoning_effort.py`, 16.06., Exit 0): `none`/`high` wirkt pro Call und schlägt den App-Toggle.

**Test:** `py_compile` (4 Skripte) ✓ · `_test_safe_eval.py` ✓ · `_test_prune_docs.py` ✓ (Replay
441→420 unverändert) · `_test_incremental_resume.py` T1–T9 ✓ (Byte-Identität trotz neuem
`reasoning_tokens`-Key gewahrt) · grep: kein `think=`/`body["think"]` mehr in `pipeline.py`.
Mess-Lauf (Nutzer, LM Studio/Gemma): voller train-Lauf, dann Judge vs. `041318` (= „alles denkt") —
über `accuracy_end_to_end` + Crash-/`length`-Zahl + Laufzeit.

## 2026-06-14 — Aufräumung Runde 2 (Re-Analyse-Scratch, Housekeeping)

**Was:** Den Arbeitsbaum nach der 213203-Re-Analyse wieder auf Vordermann gebracht — nichts gelöscht,
alles nach `archive/2026-06-14/` verschoben.
- **`code/analysis_tools/` aktiv geblieben:** nur die dokumentierten Dauer-Tools `_diff_runs.py`,
  `_test_incremental_resume.py`, `_test_safe_eval.py`, `_test_prune_docs.py`. Alle Einmal-`_ni_*.py`
  (needed_info-Chat + Re-Analyse: `_ni_anchors/_ni_check81/_ni_explore/_ni_facts/_ni_master/_ni_retr_cf/`
  `_ni_taxonomy/_ni_wrong_ids/_ni_dossier/_ni_lib/_ni_report`) → `archive/2026-06-14/scripts/` (jetzt 34).
- **`data/results/debug/` aktiv geblieben:** die Write-ups `analysis_history.md`,
  `reanalysis_213203_SYNTHESIS.md`, `_metric_addressable.md`, `wrong_ids_213203.txt`. Scratch
  (`dossiers/`, `_ni_wf_result.json`, `_ni_retr_ab.log`, `needed_info_master.jsonl`, `ni_retr_cf.jsonl`)
  → `archive/2026-06-14/debug_scratch/`. `__pycache__` entfernt (regenerierbar).
- **Docs synchronisiert:** `analysis_history.md` (Nachtrag mit Re-Analyse-Bilanz), `CLAUDE.md`
  (Archiv-Notiz + Data-layout + `_diff_runs`-Judge-Hinweis), `TODO.md` (alte Fassung →
  `archive/2026-06-14/TODO_pre-levers_20260614.md`).

## 2026-06-14 — Generierungs-Hebel (PoT-Block) + Judge-Nenner-Fix

**Hintergrund:** Re-Analyse der 45 Falschen aus `213203` (`data/results/debug/reanalysis_213203_SYNTHESIS.md`):
7 sind Gold-Fehler (Pipeline korrekt), ~13 GENUIN (unsere Lesart besser), nur ~6 echt gewinnbar. Davon die
3 risikoarmen, generellen Robustheits-Regeln gebaut (bewusst KEINE Konventions-Hebel D/E, gross-margin-$,
CAGR-Fenster — die regredieren so viel wie sie gewinnen).

**`code/pipeline.py` — 3 Regeln in `COMPUTED_VALUES_HINT`** (nur PoT-Pfad; `--no-pot`-Baseline byte-identisch):
1. Fragt die Frage nach total/sum/combined/cumulative → injizierten Total **explizit nennen** (`267`).
2. „How did X change" → **beide Endpunkte** nennen, nicht nur das Delta (`155`).
3. Keine `INSUFFICIENT DATA`, wenn Compute-Werte/Metrik-Werte in den Facts liegen — beste belegte Antwort
   geben statt kapitulieren (`126`,`144`).
Erwartung: ~+1…+3, **innerhalb des Judge-Rauschens** → nicht als Beweis lesen; gebaut wegen Korrektheit,
nicht wegen der Punkte.

**`code/run_judge.py` — Nenner-Fix (additive Keys, abwärtskompatibel):**
- INSUFFICIENT-Erkennung normalisiert (`_is_insufficient`: strip/casefold, Punkt am Ende toleriert) — vorher
  wurde `"INSUFFICIENT DATA."` gejudgt, `"INSUFFICIENT DATA"` geskippt.
- Skip-Aufschlüsselung `n_skipped_crash` / `n_skipped_insufficient` / `n_skipped_empty`.
- Neue Summary-Keys `accuracy_end_to_end` (= correct/143) und `accuracy_end_to_end_with_partial`; alte
  `accuracy` (judged-Nenner) bleibt, ist aber als „nicht laufübergreifend vergleichbar" markiert.
- Konsolen-Print zeigt beide Nenner; Lauf-Vergleich künftig auf `accuracy_end_to_end` (oder `_diff_runs.py`).

**Verifikation:** `py_compile` beide ✓ · `_test_safe_eval.py` ✓ · neue Keys/Regeln per grep bestätigt.

## 2026-06-14 — Aufräumung & Archivierung (Housekeeping, kein Pipeline-Code)

**Was:** Den Arbeitsbaum auf den ehrlichen Stand reduziert. Alle Alt-Läufe (`rag_*` außer
`rag_20260613_213203`), die Detail-Analysen (`run_analyses/`, Daten-Dumps) und ~22 Wegwerf-Analyse-
Skripte (+ `run_pot38.ps1`, `ANALYSIS_PROMPTS.md`) nach **`archive/2026-06-14/`** verschoben — nichts
gelöscht. `archive/` ist neu in `.gitignore` (lokal only, hält die ~168 MB Alt-Daten aus git).

**Konsolidierte Analyse:** `data/results/debug/analysis_history.md` neu geschrieben als **eine**
Datei — Chronologie aller Analysen (Datum · Lauf · Ergebnis · gilt-noch) + ehrlicher Aktualstand
`213203` (62-Fragen-Potenzial-Karte, Hebel-Ranking, Decke ~63–76 %). Ersetzt die alte Teil-1–8-Datei
+ die 10 run_analyses-Reports (beide vollständig im Archiv).

**Aktiv geblieben:** Core-Pipeline; die 3 Offline-Tests + `_diff_runs.py`; Lauf `213203` + S1
`173341` (+ `s1_eval_20260603_224526.jsonl` als Fixture für `_test_prune_docs.py`); `CLAUDE.md`
(Abschnitte Status/Tools/Data-layout aktualisiert), `CHANGELOG.md`, `TODO.md` (Aufräum-Punkt erledigt).

**Verifikation:** `py_compile` der 4 Core-Skripte ✓ · `_test_safe_eval.py` ✓ · `_test_prune_docs.py`
✓ (Replay 441→420, Verlierer {206,209}) · `_test_incremental_resume.py` T1–T9 ✓ · `git status` führt
`archive/` nicht (korrekt ignoriert).

## 2026-06-13 — S1-Doc-Precision: restriktiver Prompt (Comparative-Columns statt „all years")

**Was:** `SYSTEM_PROMPT` (`code/eval_s1.py`) Year-Rules umgebaut. Die bisherige defensive Sprache
(„always include all years to be safe", „still include all intermediate years — the retrieval stage
will filter") **entfernt** und durch eine **statement-aware Minimal-Regel** ersetzt: Ein 10-K trägt
Vergleichsspalten (GuV/CF **3 Jahre**, Bilanz **2 Jahre**). Für FY-only-Fragen über mehrere
aufeinanderfolgende Jahre → **kleinste Menge 10-Ks, deren Vergleichsspalten alle gefragten Jahre
abdecken**: vom neuesten Jahr rückwärts in Schritten von **3** (GuV/CF) bzw. **2** (Bilanz). Leitplanke:
bei Unsicherheit, ob die Kennzahl in einer Vergleichsspalte steht (Segmente, ungewöhnliche/restatete
Posten), das Jahr einzeln listen — **Recall vor minimaler Liste**. Gilt nur für **FY/10-K**; 10-Q
unberührt (Rules 1–5). `needed_info` nennt jetzt den vollen Jahresbereich (z. B. „FY2020-2024"), damit
S2 die Vergleichsspalten der gewählten Filings ausliest. Feld-Referenz `years` + drei Beispiele
konsistent gezogen: P&G-Trend `[2020…2024]`→`[2024, 2021]`, Amazon-Growth `[2021…2024]`→`[2024, 2021]`,
neues Microsoft-Bilanz-Beispiel `[2024, 2022]` (demonstriert Schritt 2).

**Warum:** Der dominante Über-Doc-Block der Baseline `s1_eval_20260603_224526.jsonl` (93 der 102
Über-Docs = **Jahres-Überschuss**) wurde von Rule 5 (Periode) und `--prune-docs` (nur zwei
aufeinanderfolgende FY) kaum erfasst. Die am 11.06. verworfenen **pauschalen** Trimmer kosteten Recall
(Endpunkte 90,8 %, 3-Jahres-Fenster 88,5 %), weil sie GT-zitierte Einzeljahre blind droppten. Der
Prompt verlagert die Entscheidung ins Modell — **pro Frage, statement-aware** — statt einer Heuristik,
die nicht zwischen „nötig" und „redundant" unterscheiden kann.

**Risiko & Messung:** Erwartung Precision ↑. Recall-Risiko ist real und teils ein **Metrik-Artefakt**:
die S1-Doc-Recall-Metrik wertet ein GT-zitiertes Einzeljahr-10-K als Miss, auch wenn der Wert in der
Vergleichsspalte eines behaltenen Filings steht (vgl. openqa_206/209 bei `--prune-docs`) — dort fände
S2 die Antwort trotzdem. Daher beim A/B **per-qid-Diff** und jeden verlorenen Recall als *Artefakt* vs.
*echt* einordnen; viele echte Misses (Wiederholung der 88,5 %-Story) = Prompt entschärfen/verwerfen.

**A/B-Invariante:** Nur der S1-Prompt geändert, **kein** S2/S3-Code; `--prune-docs` unberührt
(Default OFF). Die Baseline-S1-Datei `s1_eval_20260603_224526.jsonl` bleibt unverändert als Vergleich.

**Test:** `py_compile code\eval_s1.py` ✓ · Grep bestätigt: keine „all years to be safe"/„retrieval
stage will filter"-Reste mehr. Mess-Lauf (Nutzer, LM Studio/Gemma): **ein** voller train-S1-Lauf
(`python code\eval_s1.py`, eine Datei), dann gegen die Baseline diffen (avg doc precision/recall +
company recall, `_diff_s1_runs.py`/`_diff_s1_queries.py`).

## 2026-06-11 — S1-Doc-Precision: Rule 5 (ein Target pro Jahr) + `--prune-docs` in S2

**Befund (offline, train-S1-Baseline `s1_eval_20260603_224526.jsonl`):** 102 der 441 aufgelösten
Docs sind nicht in der GT (Macro-Precision 84,3 %). Davon 93 **Jahres-Überschuss** (defensives
„all years to be safe" bei trend/growth/argmax), 6 Perioden-Überschuss (4 Jahr×Perioden-
Kreuzprodukt-Fragen: openqa_70/167/168/316), 3 Company-„Extras" (openqa_113 — epistemisch nötig,
argmax braucht alle Kandidaten). Simulierte Pauschal-Trimmer wurden **verworfen**: Endpunkte bei
trend/growth bzw. 3-Jahres-Fenster→neuestes 10-K kosten je 23 Fragen S1-Recall (90,8 %/88,5 %) —
die Zwischenjahre werden real gebraucht. Einzig sicheres Fenster: **zwei aufeinanderfolgende
FY-Jahre** → das neuere 10-K deckt das Vorjahr auf jedem Statement ab (GuV/CF 3 Jahre, Bilanz 2).

**Maßnahme A — Rule 5 im S1-Prompt** (`code/eval_s1.py`, SYSTEM_PROMPT): Ein Target expandiert als
volles years×periods-Kreuzprodukt; brauchen verschiedene Jahre verschiedene Perioden → **ein Target
pro Jahr** (neue Rule 5 + Beispiel „Mixed periods across years"). Selektiver Re-Run **nur** der 4
betroffenen qids (`--qids …`, LM Studio/Gemma) → `s1_eval_20260611_175353.jsonl`; per neuem
Merge-Tool über die Baseline gelegt → **`s1_eval_20260611_175353_merged.jsonl`** (143 Records,
139 byte-identisch). Effekt: alle 4 Recall 1,0 behalten; HP-Fragen jetzt Precision 1,0; Docs der
4 Fragen 26→16; gesamt 441→431, Precision 84,3 %→85,2 %. Verbleibende 4 Extras (Oracle/eBay) sind
defensives Vorquartal für QoQ — vertretbar, kein Kreuzprodukt-Müll mehr.

**Maßnahme B — `--prune-docs` in S2** (`code/pipeline.py`): neues `prune_target_docs(target, docs)`
+ Flag `--prune-docs`/`--no-prune-docs` (BooleanOptionalAction, **Default OFF** = Baseline-Pfad
unverändert). Regel: FY-only-Target mit **genau zwei aufeinanderfolgenden Jahren** → nur das
neueste 10-K (Safety: nie auf leer prunen; Drop wird geloggt). Greift vor dem vector_db-Filter in
`process_question`; `build_config_from_args` + `_summary.json` speichern `prune_docs`.
`pipeline_resume.py`: gleiches Flag (Default OFF) + `prune_docs` in den Soft-Drift-Keys (alte
Summaries ohne den Key lösen keinen Fehlalarm aus). Gemessen auf train (Replay): Docs 441→420,
S1-Recall 97,8 %→97,1 % (**exakt** openqa_206/209 — deren Gold zitiert beide Jahres-10-Ks, die
Werte stehen aber auch in den Vergleichsspalten des neueren), Precision 84,3 %→87,4 %.
**Kombiniert** (merged S1 + `--prune-docs`): **410 Docs (−7 %), Precision 88,4 %, Recall 97,1 %.**

**Einordnung:** Kosten-/Noise-/n_ctx-Hebel (~−70 Fact-Calls pro Voll-Lauf, weniger periodenfremde
Facts im Answer-Prompt), **kein** direkter Accuracy-Hebel; den Crash-qids hilft es kaum (meist nur
2–3 Docs; openqa_73 ist ein Plan-Fehler mit Recall 0: FY2024-Q1-10-Qs geplant, GT in den 2023er
10-Ks). Bleibt in der Roadmap hinter PoT-Messlauf und Crash-Fix.

**A/B-Invariante gewahrt:** Ohne `--prune-docs` ist S2 byte-identisch; die Baseline-S1-Datei wurde
nicht verändert (Merge schreibt eine NEUE Datei, `--out` verweigert Überschreiben).

**Neue Tools:** `code/analysis_tools/_test_prune_docs.py` (Unit-Cases + Replay über die lokale
train-S1-Baseline, pinnt 441→420/97,1 %/87,4 %/Verlierer exakt {206, 209}; ohne Modell) und
`code/analysis_tools/_merge_s1_plans.py` (qid-weises Mergen von S1-Re-Plans in eine neue Datei).

**Test:** `py_compile` (5 Dateien) ✓ · `_test_prune_docs.py` ✓ · `_test_safe_eval.py` ✓ ·
`_test_incremental_resume.py` T1–T9 ✓ (Test-CONFIG um `prune_docs: False` ergänzt — Summaries
speichern den Key jetzt) · `--help`-Smoke beider CLIs ✓ · S1-Re-Run der 4 qids real gegen LM
Studio gefahren und Metriken verifiziert. Mess-Lauf (Nutzer, optional nach PoT-Messung): voller
train-Lauf mit `--s1-results …_merged.jsonl --prune-docs` + Judge vs. aktuellen PoT-Stand.

## 2026-06-11 — PoT ist jetzt Default an (`--pot` / `--no-pot`)

**Was:** PoT-Default umgedreht. Beide CLIs nutzen jetzt `argparse.BooleanOptionalAction`:
- `code/pipeline.py:1432` — `--pot` mit `default=True` (erzeugt `--pot` **und** `--no-pot`). PoT läuft
  ohne Flag; `--no-pot` schaltet auf den eingefrorenen Baseline-Pfad.
- `code/pipeline_resume.py:221` — `--pot` mit `default=None`. Ohne explizites Flag **erbt** der Resume
  jetzt das PoT-Setting des Original-Laufs (abgeleitet aus den `_recheck.jsonl`-`pot`-Feldern; Fallback
  ON, wenn unbestimmbar). Die bestehende Drift-Warnung greift nur noch bei explizitem `--pot`/`--no-pot`,
  das vom Original abweicht — Resume kann den A/B-Status nicht mehr versehentlich kippen.
- `CLAUDE.md` synchronisiert (Architektur-Abschnitt S2 + Flag-Liste): „default ON", `--no-pot` = Baseline.

**Warum:** Der Lauf `rag_20260611_042823` lief versehentlich **ohne** `--pot` (0/143 PoT-Formeln vs.
137/143 in `rag_20260607_013732`) und fiel damit still auf die Baseline zurück — strict 54,5 % statt
der 60,1 % von `013732`, und nicht mit ihm vergleichbar (einziger re-relevanter Unterschied war PoT,
nicht die Prompt-Fixes). Ein `store_true`-Flag, dessen Vergessen lautlos den Haupthebel abschaltet, ist
eine Falle. Default-on macht PoT zum Normalpfad; `--no-pot` behält das saubere A/B.

**A/B-Invariante gewahrt:** Nur der **Default** des Flags ändert sich, nicht die Logik. `--no-pot` ist
byte-identisch zum bisherigen `--pot`-aus-Pfad (Baseline `rag_20260606_020413` weiterhin reproduzierbar).

**Test:** Offline verifiziert, alles grün:
- `--help` beider CLIs zeigt `[--pot | --no-pot]` mit korrekten Defaults.
- `py_compile code\pipeline.py code\pipeline_resume.py` → fehlerfrei.
- `_test_safe_eval.py` → `safe_eval OK`; `_test_incremental_resume.py` → T1–T9 `ALL OFFLINE TESTS PASSED`.

## 2026-06-11 — Antwort-Prompt-Fixes 1–3 + HTTP-400-Diagnose

**Was:** Umsetzung des am 10.06. beschlossenen Pakets (`data/results/debug/analysis_history.md`, Teil 7;
vormals `plan_prompt_fixes_20260610.md`):

1. **`COMPUTED_VALUES_HINT`** (`code/pipeline.py:148`) von einem Satz zu einem Regelblock erweitert —
   trägt Fix 1 (berechneten Prozent-/Verhältnis-Wert in der **gefragten Form** ausgeben, nie in
   Rohwerte zurückrechnen, %-Zeichen/Einheit dran), Fix 2 (Multi-Entität/Multi-Perioden **nicht
   kollabieren**: jede gefragte Entität/Periode einzeln, nicht summieren außer explizit verlangt,
   nichts Ungefragtes) und den Ausgabe-Teil von Fix 3 (Vergleiche als **positiver Betrag +
   Richtungswort**, z. B. "4.74% lower", statt vorzeichenbehaftetem Wert).
2. **`COMPUTE_PROMPT`** (Standard-formulas-Block): "percentage difference"-Zeile um die
   Reihenfolge-Konvention ergänzt (A = die in der Frage **zuerst** genannte Entität → Vorzeichen
   deterministisch interpretierbar). `percentage change` bewusst unverändert (Vorzeichen ist dort
   gewollte Information, vgl. qid 116).
3. **HTTP-Diagnose** in `llm_call`: `raise_for_status()` durch expliziten `HTTPError`-Raise mit
   `resp.text[:500]` ersetzt — der echte 400-Grund landet damit über `main`s Fehlerbehandlung
   (`"error": f"{type(e).__name__}: {e}"`) automatisch im `error`-Feld der `_results.jsonl`.

**A/B-Invariante gewahrt:** `ANSWER_GENERATION_PROMPT` unverändert; der erweiterte Hint wird wie
bisher **nur** bei nicht-leeren `computed_values` angehängt → `--pot`-aus-Pfad byte-identisch zur
Baseline. `pipeline_resume.py` erbt alles via Import, kein Edit nötig.

**Warum:** Die Fixes reparieren überwiegend PoT-Regressionen und verwandte Format-Verluste ohne
Annahmen über Gold-Konventionen (Belege: Fix 1 → qid 35/177/181 · Fix 2 → 274/318/70/144 (111
Grenzfall: fragt wörtlich "combined") · Fix 3 → 53/304; qid 24 außer Reichweite, da LOOKUP ohne
computed values). Erwartung laut Beschluss: **~+5–7** auf 86/143 = 60,1 % (Lauf `013732`).

**Test:** Offline verifiziert (py_compile, `_test_safe_eval.py`, `_test_incremental_resume.py`,
A/B-Identitäts-Check ohne `computed_values`). Mess-Lauf (Nutzer): voller train-Lauf `--pot` + Judge
vs. `rag_20260607_013732`; Effekt-Trennung per qid s. o., Regressionen per `_diff_runs.py`.

## 2026-06-08 — Compute-Prompt: Operanden-/Metrik-Auswahl-Anleitung

**Was:** `COMPUTE_PROMPT` (`code/pipeline.py:154`) rein **additiv** erweitert: (1) zwei
Kennzahl-Definitionen (`percentage difference between A and B`, `debt-to-equity = total liabilities /
total equity`); (2) neuer Block „Choosing the right numbers" mit vier Auswahl-Regeln (PERIOD, SCOPE,
DISAMBIGUATION, LABEL MATCH) + explizitem Hinweis, dass es **Auswahl-Anleitung, keine Einschränkung**
ist. Output-Format (striktes JSON `{name: expr}`) und `_parse_formulas` **unverändert** → A/B bleibt
sauber, `--pot`-aus-Pfad unberührt.

**Warum:** Operanden-Tiefenanalyse des PoT-Laufs `rag_20260607_013732` zeigte: in ~16/18 geprüften
Fehlern lag die richtige Zahl in den Facts, aber der Compute-Call wählte den falschen von mehreren
gültigen Operanden / die falsche Metrik-Definition (z.B. `openqa_121`: Debt 61.473 statt Liabilities
174.801). Der Engpass ist die Auswahl, nicht das Rechnen oder Retrieval. Bewusst **keine** Restriktion
der Fact-Extraktion (Deep-Research-Befund: einschränken verliert wichtige Infos). Details:
`data/results/debug/analysis_history.md`, Teil 3 §7 (vormals `pot_run_analysis.md`).

**Test:** Voller train-`--pot`-Lauf gegen Baseline `rag_20260606_020413` (kein PoT) und
`rag_20260607_013732` (PoT ohne diese Anleitung). Vorab n_ctx/`LLM_MAX_TOKENS` abstimmen (n_ctx ≈ 33k).

## 2026-06-06 19:22 — Program-of-Thought (PoT) in S2

**Was:** Optionaler PoT-Schritt in `code/pipeline.py`, aktiviert per `--pot` (Default AUS). Zwischen
Fact-Extraction und Answer-Generierung stellt ein kleiner Compute-Call (`think=False`) die nötigen
Rechnungen als benannte JSON-Formeln auf; **Python** wertet sie über einen sicheren AST-Evaluator
exakt aus; die Ergebnisse gehen als Block in den bestehenden Answer-Call. Gemma formuliert dann nur
noch, rechnet nicht mehr.

**Warum:** Generierung ist der Engpass (end-to-end 52,4 % auf train). ~33–38 Fehler sind
Rechen-/Kapitulationsfehler bei vorhandenen Facts (Befund 3 + 6 in
`data/results/debug/analysis_history.md`, Teil 2; vormals `answer_generation_analysis.md`). PoT ist der größte Hebel.

**Geänderte/neue Symbole** (`code/pipeline.py`):
- `import ast`, `import operator` (Import-Block).
- Abschnitt „Program-of-Thought": `_BINOPS`/`_UNARY`, `_eval_node`, `safe_eval`, `_balanced_object`,
  `_parse_formulas`, `compute_values`.
- Prompt-Konstanten `COMPUTE_PROMPT` und `COMPUTED_VALUES_HINT`.
- `generate_answer(...)` nimmt `computed_values=None`, hängt den Block nur bei vorhandenen Werten an
  und gibt zusätzlich den **tatsächlich gesendeten** System-Prompt zurück (4-Tuple statt 3).
- PoT-Schritt in `process_question`, gegated auf `config.get("pot")`; `"pot"`-Detail im Ergebnis-Dict
  und in `build_recheck_line` (→ `_recheck.jsonl`).
- Flags `--pot` und `--qids-file` (argparse + `build_config_from_args` + `main`-Merge mit `--qids`,
  `#`-Kommentare/Leerzeilen werden ignoriert).

**`code/pipeline_resume.py`:** `--pot`-Flag ergänzt (teilt sich `build_config_from_args` mit der
Hauptpipeline — ohne das Flag wäre jeder Resume-Lauf an `args.pot` abgestürzt); zusätzlich Warnung bei
`--pot`-Mismatch (Original-Setting wird aus `_recheck.jsonl` abgeleitet, da `_summary.json` es nicht
speichert).

**Neuer Test:** `code/analysis_tools/_test_safe_eval.py` — Offline-Selbsttest für `safe_eval` +
`_parse_formulas` (kein Modell nötig).

**Sicherheit von `safe_eval`:** erlaubt ausschließlich Zahlen, `+ - * / ** %`, unäres `+/-`, Klammern
und `abs()`. Alles andere (Variablen, beliebige Calls, Attribute, Subscripts, Booleans) → `None`.
Niemals Pythons `eval()`. Exponent ist auf |x| ≤ 1000 begrenzt (verhindert hängende Integer-Türme wie
`10**10**8`). Parsing toleriert Code-Fences, nachgelagerten Prosa-Text mit Klammern und mehrere
Objekte (erstes valides Objekt gewinnt) und kann den Lauf nie zum Absturz bringen.

**Adverser Mehr-Agenten-Review (7/8 Befunde bestätigt) eingearbeitet:**
- `_parse_formulas`: greedy `{…}`-Regex → String-bewusste, balancierte Extraktion (major).
- `ANSWER_GENERATION_PROMPT` byte-identisch zur Baseline gehalten; Hinweis nur konditional injiziert,
  damit der `--pot`-aus-Pfad exakt reproduzierbar bleibt (major).
- `safe_eval`: Booleans abgelehnt; `**`-Exponent gecappt (nits).
- Resume: `--pot`-Drift-Warnung (minor).
- Bewusst NICHT geändert: `_recheck.jsonl` enthält in Baseline-Läufen ein additives `"pot": null`
  (vom Plan sanktioniert; `_results.jsonl` und Resume-Seed bleiben unberührt).

**Bewusste Abweichung vom ursprünglichen Plan (POT_PLAN.md, Schritt 5):** Der Plan verlangte den
Antwort-Prompt-Hinweis *unbedingt* in der Konstante. Das widerspricht der Architektur-Entscheidung #2
desselben Plans (sauberer A/B, Baseline `rag_20260606_020413` ohne Flag reproduzierbar). Entschieden
zugunsten #2 → Hinweis wird nur angehängt, wenn `computed_values` vorhanden sind.

**Verifikation (alles grün, kein Pipeline-Lauf gestartet):**
- `.venv\Scripts\python.exe code\analysis_tools\_test_safe_eval.py` → `safe_eval OK`
- `.venv\Scripts\python.exe -m py_compile code\pipeline.py code\pipeline_resume.py` → fehlerfrei
- bestehender Resume-Test → `ALL OFFLINE TESTS PASSED` (Byte-Identität trotz neuem `"pot"`-Key gewahrt)
- `git diff HEAD` bestätigt: `ANSWER_GENERATION_PROMPT` unverändert ggü. HEAD.

**Referenzen (übernommen aus dem gelöschten `POT_PLAN.md`):**
- Baseline-Lauf: `rag_20260606_020413` (in `data/results/`); zugehöriges S1:
  `data/results/s1_eval_20260603_224526.jsonl`.
- PoT-Testset: `data/eval_sets/pot_calc_train.txt` (38 CALC-qids, in der Baseline alle falsch/
  INSUFFICIENT → jede jetzt `correct` gejudgte Frage ist Netto-Gewinn von PoT).
- Crash-Testset (für späteren, separaten Crash-Fix): `data/eval_sets/crashes_train.txt`.

**Wie der Nutzer testet** (LM Studio + Gemma müssen laufen):
```powershell
.venv\Scripts\python.exe code\pipeline.py `
    --s1-results data\results\s1_eval_20260603_224526.jsonl `
    --split train --pot --qids-file data\eval_sets\pot_calc_train.txt

.venv\Scripts\python.exe code\run_judge.py `
    --results data\results\rag_<neuer_ts>_results.jsonl --concurrency 2
```
