"""Probe: Hat gemma-4-31b eine reasoning_effort-INTENSITAETSSTUFE oder nur an/aus?

Schwesterskript zu _probe_think_modes.py. Jenes faehrt die App-vs-API-Achse
(wer gewinnt, App-Toggle oder API-Parameter). DIESES isoliert eine einzige
Achse: reasoning_effort, ein Wert pro Request, sonst nichts. So sieht man, ob
'low'/'medium'/'high' verschiedene Denkmengen erzeugen (echte Stufe) oder alle
gleich viel (= nur an/aus, Stufe ist Attrappe).

Lesehilfe: completion_tokens hoch (hunderte+) = gedacht; niedrig (~zig) = nicht.
Unterscheiden sich low/medium/high in den ctoks -> echte Stufe. Identisch -> nur an/aus.

    python code/analysis_tools/_probe_effort_levels.py --model google/gemma-4-31b
"""

import argparse
import requests

URL = "http://localhost:1234/v1/chat/completions"
PROMPT = (
    "A train goes 17 km in hour 1, then 23 km in each of hours 2 and 3. "
    "What is the average speed over the 3 hours? Give just the number."
)

# Jeder Fall setzt GENAU EIN Feld (ausser baseline) -> saubere Isolation.
CASES = [
    ("baseline (nichts gesetzt)", {}),
    ("reasoning_effort=off",      {"reasoning_effort": "off"}),
    ("reasoning_effort=none",     {"reasoning_effort": "none"}),
    ("reasoning_effort=minimal",  {"reasoning_effort": "minimal"}),
    ("reasoning_effort=low",      {"reasoning_effort": "low"}),
    ("reasoning_effort=medium",   {"reasoning_effort": "medium"}),
    ("reasoning_effort=high",     {"reasoning_effort": "high"}),
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
    except Exception as e:  # noqa: BLE001
        print(f"[{label}]  REQUEST-FEHLER: {e!r}")
        return
    if not r.ok:
        print(f"[{label}]  HTTP {r.status_code}: {r.text[:200]}")
        return
    d = r.json()
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    ct = d.get("usage", {}).get("completion_tokens")
    has_tag = "<think>" in content
    thinks = bool(has_tag or reasoning)
    print(f"[{label}]")
    print(f"    THINKING={'YES' if thinks else 'no '}  ctoks={ct}  "
          f"<think>-tag={has_tag}  reasoning-feld={'ja' if reasoning else 'nein'}  "
          f"content_len={len(content)}")
    print(f"    content head: {content[:110]!r}")
    if reasoning:
        print(f"    reasoning head: {reasoning[:110]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31b")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    print(f"modell unter test: {args.model}")
    print(f"prompt: {PROMPT}")
    print("=" * 78)
    for label, extra in CASES:
        run_case(args.model, label, extra, args.max_tokens, args.timeout)
        print("-" * 78)
    print("Lesehilfe: low/medium/high unterschiedliche ctoks = echte Stufe; gleich = nur an/aus.")


if __name__ == "__main__":
    main()
