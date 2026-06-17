# TODO — offene Arbeiten

Ein Abschnitt pro offenem Punkt; Erledigtes wandert mit Datum ins `CHANGELOG.md`.
Älterer Plan (Potenzial-Karte, ausführliche Judge-Fix-Spezifikation) archiviert unter
`archive/2026-06-14/TODO_pre-levers_20260614.md`.

---

## Stand 2026-06-16 — `think` → `reasoning_effort` verdrahtet (Code ✓ + Verifikation ✓); nur Mess-Lauf offen

**Erledigt seit 14.06.:** Lauf `041318` (84/143 = 58,7 % e2e) gefahren, gejudgt und 5-fach
tiefenanalysiert gegen `213203` (Reports lokal `data/results/debug/cmp_041318_v_213203/`, Bilanz in
Memory `secqa-041318-lever-ab-verified`). A/B-Befund: die 3 Generierungs-Regeln greifen mechanisch
(2 harte Gewinne `267`/`155`, R3 score-neutral), aber +3 liegt im Rauschen — Mikro-Hebel nur per
Mechanik-Attribution messbar, nicht am End-Score.

**Kritischer Nebenbefund (Memory `lm-studio-think-param-dead`, am 15.06. empirisch bewiesen):**
Der `think`-Parameter aus `pipeline.py` (`llm_call`, genutzt in `extract_facts`/`generate_answer`/
`compute_values`) wird von LM Studio **ignoriert**. Thinking steuert nur (a) der LM-Studio-App-Toggle
und (b) `reasoning_effort` (gültig `none/minimal/low/medium/high/xhigh`; bei Gemma binär — `none`=aus, jeder andere Wert=an, KEINE Stufe; `on`/`off` werfen HTTP 400). Reasoning kommt
in `message.reasoning_content`, nicht als `<think>` → der Code erfasst es nicht (`answer_think` war
immer None). Folge: alle bisherigen Läufe liefen real im **„alles denkt"-Modus** (Fact ~687, Compute
~722 für *1* Formel, Answer ~494 Median completion_tokens) — entgegen dem Code-Intent, und mit-Ursache
für Laufzeit (~10 h) + n_ctx-Crashes.

### 1. Code sauber machen — `think` → `reasoning_effort`  ✓ ERLEDIGT 16.06.
Umgesetzt in `code/pipeline.py` + `CLAUDE.md` korrigiert; `pipeline_resume.py` brauchte keinen Edit
(per grep bestätigt). Drei Konstanten `REASONING_FACT="none"`/`COMPUTE="high"`/`ANSWER="high"`
(`pipeline.py:61`), `llm_call` schickt `reasoning_effort` + loggt `reasoning_content` ins `think`-Feld
und neu `reasoning_tokens` pro Schritt. Offline verifiziert (py_compile + 3 Tests grün, Replay
unverändert, kein `think=`-Rest). Voller Eintrag: `CHANGELOG.md` (16.06.).
- **Nicht gebaut (bewusst):** Fallback `enable_thinking` via `chat_template_kwargs` — nur falls
  `reasoning_effort` bei einem LM-Studio-Update bricht; nicht offiziell für `/v1/chat/completions`
  dokumentiert, daher erst bei Bedarf messen.

### 2. Verifizieren — ✓ ERLEDIGT 16.06. (`code/analysis_tools/_test_reasoning_effort.py`, Exit 0)
Neues PASS/FAIL-Skript gegen `gemma-4-31b` gefahren, alle 4 Checks grün:
- `reasoning_effort="none"` → 0 reasoning_tokens, kein reasoning_content → denkt NICHT (Antwort trotzdem da).
- `reasoning_effort="high"` → reasoning_tokens>0 + reasoning_content → denkt.
- none ≠ high im SELBEN Lauf → der API-Parameter **überschreibt den App-Toggle** (Ein-Lauf-Beweis, Toggle-Stand egal).
- `off`/`on` → **HTTP 400** → der alte TODO-Wert `off`/`on` war falsch, `none`/`high` ist korrekt.
- Reasoning-Trägerfelder bestätigt: `reasoning_content` + `completion_tokens_details.reasoning_tokens` (kein `<think>`) → steuert den Logging-Fix oben.
Optional-Nachweis: Skript mit App-Toggle AUS wiederholen (muss wieder grün sein) — nicht zwingend.

### 3. Nachtlauf — non-thinking Extraction, thinking Compute + Answer
Nach 1 + 2: voller train-Lauf mit der neuen Konfiguration, dann judgen.
```bash
python code/pipeline.py --s1-results data/results/s1_eval_20260613_173341.jsonl --split train
python code/run_judge.py --results data/results/rag_<neuer_ts>_results.jsonl --concurrency 2
```
Auswerten gegen `041318` (= „alles denkt") **nur über `accuracy_end_to_end`**: drei Zahlen entscheiden
— Score, **Crash-Zahl / `finish_reason: length`**, Laufzeit. Bricht der Score ein, ist der
Compute-Schritt der Hauptverdächtige (Operanden-Auswahl). Beim Judgen sicherstellen, dass `qwq-32b`
sein Reasoning behält (prüfen, ob der Toggle global oder pro-Modell ist). Bei Abbruch:
`python code/pipeline_resume.py`.

---

## Bekannte Decken-Posten (NICHT weiter „adressieren")

7 filing-verifizierte Gold-Fehler aus der 213203-Re-Analyse (Pipeline war korrekt):
`73, 133, 156, 189, 199, 248, 316`. Plus ~13 GENUIN (unsere Lesart die bessere). Belege:
`data/results/debug/reanalysis_213203_SYNTHESIS.md` (+ `_metric_addressable.md` pro ID).
