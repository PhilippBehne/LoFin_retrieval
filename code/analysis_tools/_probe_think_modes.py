"""Probe: Respektiert LM Studio den `think`-API-Parameter — und wer gewinnt gegen
den App-seitigen Thinking/Reasoning-Toggle?

Hintergrund: pipeline.py sendet `body["think"]=True/False` (llm_call, Zeile 482).
Ob das überhaupt wirkt, entscheidet LM Studio, nicht der Code. Dieses Skript schickt
denselben Prompt in mehreren Request-Varianten an den lokal geladenen Chat-Endpunkt
und zeigt pro Variante, ob das Modell GEDACHT hat (<think>-Block im content ODER ein
separates reasoning-Feld) und wie viele completion_tokens es dafür verbrannt hat.

Voller Wahrheitstest = App-Einstellung x API-Parameter. Dieses Skript fährt die
API-Achse. Die App-Achse stellst du selbst:
  1. LM Studio: Thinking/Reasoning-Toggle AUS -> dieses Skript laufen lassen.
  2. LM Studio: Toggle AN -> nochmal laufen lassen.
Vergleich der beiden Läufe zeigt, wer gewinnt:
  - "think=True" zeigt thinking=YES, obwohl App-Toggle AUS  -> API-Parameter ueberschreibt die App.
  - alle Varianten zeigen "no", sobald App-Toggle AUS        -> die App gewinnt, unser think-Param ist machtlos.

Voraussetzung: LM Studio laeuft, das Answer-Modell ist geladen.
    python code/analysis_tools/_probe_think_modes.py
    python code/analysis_tools/_probe_think_modes.py --model google/gemma-4-31b --timeout 90
"""

import argparse
import requests

URL = "http://localhost:1234/v1/chat/completions"
MODELS_URL = "http://localhost:1234/v1/models"
DEFAULT_MODEL = "google/gemma-4-31b"

# Reasoning-induzierend, aber kurze Endantwort (21) -> die completion_tokens-Luecke
# zwischen "denkt" und "denkt nicht" wird gross und gut sichtbar.
PROMPT = (
    "A train goes 17 km in hour 1, then 23 km in each of hours 2 and 3. "
    "What is the average speed over the 3 hours? Give just the number."
)

# label, zusaetzliche Body-Felder. Die ersten vier sind genau die gewuenschten Faelle;
# "alle AUS" als Gegenprobe zu "alle AN".
CASES = [
    ("nix gesetzt (reiner Default)", {}),
    ("think=True (sync on)",         {"think": True}),
    ("think=False (sync off)",       {"think": False}),
    ("ALLE Optionen AN",            {"think": True, "reasoning_effort": "high",
                                      "enable_thinking": True,
                                      "chat_template_kwargs": {"enable_thinking": True}}),
    ("ALLE Optionen AUS",           {"think": False, "reasoning_effort": "low",
                                      "enable_thinking": False,
                                      "chat_template_kwargs": {"enable_thinking": False}}),
]


def run_case(model, label, extra, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body.update(extra)
    try:
        r = requests.post(URL, json=body, timeout=timeout)
    except Exception as e:  # noqa: BLE001 - bewusst alles abfangen, ein Fall darf den Lauf nicht killen
        print(f"[{label}]\n    REQUEST-FEHLER: {e!r}")
        return
    if not r.ok:
        print(f"[{label}]\n    HTTP {r.status_code}: {r.text[:200]}")
        return
    data = r.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    ctoks = data.get("usage", {}).get("completion_tokens")
    has_tag = "<think>" in content
    thinks = bool(has_tag or reasoning)
    print(f"[{label}]")
    print(f"    THINKING? {'YES' if thinks else 'no '}   "
          f"(<think>-tag={has_tag}, reasoning-feld={'ja' if reasoning else 'nein'})   "
          f"completion_tokens={ctoks}   content_len={len(content)}")
    print(f"    content head: {content[:140]!r}")
    if reasoning:
        print(f"    reasoning head: {reasoning[:140]!r}")


def main():
    ap = argparse.ArgumentParser(description="Probe think/reasoning modes against LM Studio")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    try:
        m = requests.get(MODELS_URL, timeout=8).json()
        print("models gelistet:", [x.get("id") for x in m.get("data", [])])
    except Exception as e:  # noqa: BLE001
        print("WARN /v1/models:", repr(e))

    print(f"modell unter test: {args.model}")
    print(f"prompt: {PROMPT}")
    print("=" * 78)
    for label, extra in CASES:
        run_case(args.model, label, extra, args.max_tokens, args.timeout)
        print("-" * 78)
    print("Lesehilfe: 'completion_tokens' hoch (hunderte+) = gedacht; niedrig (~zig) = nicht.")
    print("Skript zweimal fahren (App-Toggle AUS, dann AN), um zu sehen wer gewinnt.")


if __name__ == "__main__":
    main()
