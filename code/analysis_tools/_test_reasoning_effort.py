"""Verifiziert VOR dem `think` -> `reasoning_effort`-Umbau, dass der geplante
Mechanismus gegen den ECHTEN LM-Studio-Server wirklich greift. Faellt im Gegensatz
zu den explorativen Schwestern `_probe_think_modes.py` / `_probe_effort_levels.py`
ein hartes PASS/FAIL und setzt den Exit-Code (0 = Fix sicher, 1 = nicht).

Testet GENAU die zwei Werte, die der Pipeline-Fix verwenden wird:
  - reasoning_effort="none"  -> Fact-Extraktion  (soll NICHT denken)
  - reasoning_effort="high"  -> Compute + Answer  (sollen denken)

Drei harte Behauptungen:
  [A] "none" liefert KEIN Reasoning (reasoning_tokens==0, kein reasoning_content),
      aber trotzdem eine nicht-leere Antwort.
  [B] "high" liefert Reasoning (reasoning_tokens>0 / reasoning_content), Antwort da.
  [C] Per-Call-Steuerung: weil A und B im SELBEN Lauf (= selber App-Toggle-Stand)
      unterschiedlich ausfallen, ueberschreibt der API-Parameter den GUI-Toggle.
      -> Beweis in EINEM Lauf, egal wie der Toggle steht.

Sanity [D]: "off"/"on" muessen HTTP 400 werfen (dann sind none/high die einzig
gueltigen Werte, wie in Memory `lm-studio-think-param-dead` notiert). Schlaegt D
fehl, ist der TODO-Wert "off"/"on" doch nutzbar -> wird gemeldet, kippt aber A/B/C nicht.

Nebenprodukt: zeigt, WO das Reasoning ankommt (reasoning_content vs. <think>-Tag vs.
usage.…reasoning_tokens). Das ist exakt das Feld, das der Logging-Fix in `llm_call`
kuenftig erfassen muss, damit `answer_think` nicht mehr immer None ist.

ECHTE Calls. Voraussetzung: LM Studio laeuft, das S2-Modell (gemma-4-31b) ist geladen.
Optional zweimal fahren (App-Toggle AUS, dann AN) -> [C] muss beide Male PASS sein.

    python code/analysis_tools/_test_reasoning_effort.py
    python code/analysis_tools/_test_reasoning_effort.py --model google/gemma-4-31b --max-tokens 2500
"""

import argparse
import sys

import requests

URL = "http://localhost:1234/v1/chat/completions"
MODELS_URL = "http://localhost:1234/v1/models"
DEFAULT_MODEL = "google/gemma-4-31b"  # muss zu pipeline.py LLM_MODEL passen

# Mehrschrittige Rechnung mit kurzer Endantwort: im Thinking-Modus entsteht viel
# Reasoning, bei aus eine knappe Antwort -> die none/high-Luecke wird gross und eindeutig.
PROMPT = (
    "A train travels 17 km in hour 1, 23 km in hour 2, and 23 km in hour 3. "
    "Compute the average speed over the 3 hours, then state whether it is above "
    "or below 21 km/h. Answer in one short sentence."
)


def call(model, reasoning_effort, max_tokens, timeout):
    """Ein Request. Gibt ein ausgewertetes dict zurueck; Netz-/HTTP-Fehler werden
    als Feld zurueckgegeben, nie geworfen (ein Fall darf den Lauf nicht killen)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    try:
        r = requests.post(URL, json=body, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - bewusst alles abfangen
        return {"error": repr(e)}
    if not r.ok:
        return {"http": r.status_code, "text": r.text[:300]}
    d = r.json()
    choice = d["choices"][0]
    msg = choice["message"]
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    usage = d.get("usage", {})
    details = usage.get("completion_tokens_details") or {}
    return {
        "content": content,
        "reasoning_content": reasoning,
        "reasoning_chars": len(reasoning),
        "reasoning_tokens": details.get("reasoning_tokens"),  # kann None sein
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "has_think_tag": "<think>" in content,
    }


def thought(res):
    """'Hat gedacht?' robust ueber alle Report-Formen (Token-Zaehler, eigenes Feld, Tag)."""
    rt = res.get("reasoning_tokens") or 0
    return rt > 0 or bool(res.get("reasoning_content")) or res.get("has_think_tag")


def where_reasoning(res):
    """Welche Felder trugen das Reasoning? Steuert den llm_call-Logging-Fix."""
    parts = []
    if res.get("reasoning_content"):
        parts.append(f"reasoning_content ({res['reasoning_chars']} chars)")
    if res.get("reasoning_tokens"):
        parts.append(f"usage.completion_tokens_details.reasoning_tokens={res['reasoning_tokens']}")
    if res.get("has_think_tag"):
        parts.append("<think>-Tag im content")
    return ", ".join(parts) if parts else "(nirgends sichtbar)"


def show(step, effort, res):
    print(f"{step}  (reasoning_effort={effort!r})")
    if "error" in res:
        print(f"    REQUEST-FEHLER: {res['error']}")
        return
    if "http" in res:
        print(f"    HTTP {res['http']}: {res['text']}")
        return
    rt = res["reasoning_tokens"]
    rt_s = "n/a" if rt is None else rt
    print(f"    gedacht? {'JA ' if thought(res) else 'nein'}   "
          f"reasoning_tokens={rt_s}   completion_tokens={res['completion_tokens']}   "
          f"finish={res['finish_reason']}")
    print(f"    reasoning kommt in: {where_reasoning(res)}")
    print(f"    content: {res['content'][:140]!r}")
    if res["finish_reason"] == "length" and not res["content"]:
        print("    WARN: am Token-Limit abgeschnitten, keine Antwort -> --max-tokens erhoehen")


def core_ok(res):
    """Hilfsbedingung: echte Antwort vorhanden (kein Fehler/HTTP, content nicht leer)."""
    return "error" not in res and "http" not in res and bool(res.get("content"))


def main():
    ap = argparse.ArgumentParser(description="PASS/FAIL-Test: wirkt reasoning_effort none/high pro Call?")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    try:
        m = requests.get(MODELS_URL, timeout=8).json()
        print("geladen:", [x.get("id") for x in m.get("data", [])])
    except Exception as e:  # noqa: BLE001
        print("WARN /v1/models:", repr(e))
    print(f"modell unter test: {args.model}")
    print(f"prompt: {PROMPT}")
    print("=" * 78)

    # Die zwei Pipeline-Modi. Compute und Answer sind beide "high" -> ein high-Call genuegt.
    none_res = call(args.model, "none", args.max_tokens, args.timeout)
    show("SCHRITT 1 — Fact-Extraktion (soll NICHT denken)", "none", none_res)
    print("-" * 78)
    high_res = call(args.model, "high", args.max_tokens, args.timeout)
    show("SCHRITT 2 — Compute + Answer (sollen denken)", "high", high_res)
    print("-" * 78)

    # [D] Sanity: off/on sind laut Memory keine gueltigen Werte -> HTTP 400 erwartet.
    off_res = call(args.model, "off", args.max_tokens, args.timeout)
    on_res = call(args.model, "on", args.max_tokens, args.timeout)
    d_ok = off_res.get("http") == 400 and on_res.get("http") == 400
    print("WERTE-VALIDIERUNG (Sanity):")
    print(f"    reasoning_effort='off' -> {('HTTP 400' if off_res.get('http') == 400 else off_res)}")
    print(f"    reasoning_effort='on'  -> {('HTTP 400' if on_res.get('http') == 400 else on_res)}")
    print("=" * 78)

    # Urteil
    a_ok = core_ok(none_res) and not thought(none_res)
    b_ok = core_ok(high_res) and thought(high_res)
    c_ok = a_ok and b_ok  # gegensaetzliches Verhalten bei festem Toggle => Param schlaegt Toggle

    def line(tag, desc, ok):
        print(f"  [{tag}] {desc:.<52} {'PASS' if ok else 'FAIL'}")

    print("URTEIL")
    line("A", "none denkt NICHT + Antwort da", a_ok)
    line("B", "high denkt + Antwort da", b_ok)
    line("C", "pro Call steuerbar (none != high)", c_ok)
    line("D", "off/on -> HTTP 400 (Werte korrekt)", d_ok)
    print()

    if a_ok and b_ok and c_ok:
        print("  => GRUEN: reasoning_effort none/high wirkt pro Call und schlaegt den App-Toggle.")
        print(f"     Logging-Fix in llm_call muss erfassen: {where_reasoning(high_res)}")
        if not d_ok:
            print("  HINWEIS: off/on warfen KEIN 400 -> abweichend von der Memory, off/on waeren nutzbar.")
        sys.exit(0)
    else:
        print("  => ROT: Annahme NICHT bestaetigt. Vor dem Pipeline-Umbau klaeren (siehe Felder oben).")
        if not core_ok(none_res) or not core_ok(high_res):
            print("     Mind. ein Kern-Call hatte Fehler/HTTP/keinen content -> LM Studio + Modell laden, ")
            print("     ggf. --max-tokens erhoehen, dann erneut.")
        sys.exit(1)


if __name__ == "__main__":
    main()
