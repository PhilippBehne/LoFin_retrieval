"""Offline self-test for the PoT arithmetic evaluator (safe_eval + _parse_formulas).
No LM Studio / no model load beyond importing pipeline. Touches no real data and no
test split. Run: .venv\\Scripts\\python.exe code\\analysis_tools\\_test_safe_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `code/` importable
from pipeline import safe_eval, _parse_formulas


def main():
    # ── arithmetic correctness (Plan asserts, verbatim) ──
    assert abs(safe_eval("(21526-10466)/10466*100") - 105.6755) < 1e-3   # openqa_289 Gold
    assert abs(safe_eval("(8000-7575)/abs(-7575)*100") - 5.6106) < 1e-3  # %-change, |old|
    assert safe_eval("(1.0731)**1") is not None                          # CAGR-Form evaluiert
    assert safe_eval("__import__('os').system('x')") is None             # kein fremder Call
    assert safe_eval("x + 1") is None                                    # keine Variablen
    assert safe_eval("1/0") is None                                      # div by zero

    # ── the four standard formulas from COMPUTE_PROMPT ──
    assert abs(safe_eval("(8000 - 7575) / abs(7575) * 100") - 5.6106) < 1e-3  # percentage change
    assert abs(safe_eval("(50 / 200) * 100") - 25.0) < 1e-9                   # margin / share
    assert abs(safe_eval("3846 - 1200") - 2646.0) < 1e-9                      # difference
    # CAGR over n: compare against native Python — Python keeps full precision
    assert abs(safe_eval("(8000 / 7575) ** (1/2) * 100 - 100")
               - ((8000 / 7575) ** (1 / 2) * 100 - 100)) < 1e-9
    assert abs(safe_eval("10 % 3") - 1.0) < 1e-9                              # modulo
    assert abs(safe_eval("-5 + 3") - (-2.0)) < 1e-9                           # unary minus

    # ── reject everything that is not bare arithmetic ──
    assert safe_eval("max(1, 2)") is None        # only abs() is whitelisted
    assert safe_eval("abs(-5, 3)") is None       # abs with the wrong arity
    assert safe_eval("os.system") is None        # attribute access
    assert safe_eval("[1, 2, 3]") is None        # list literal
    assert safe_eval("2 +") is None              # syntax error
    assert safe_eval("") is None                 # empty string
    assert safe_eval("True + 1") is None         # bool is an int subclass — reject
    assert safe_eval("False") is None
    assert safe_eval("abs(True)") is None

    # ── ** must not hang on an integer tower (exponent is capped) ──
    assert safe_eval("2 ** 10") == 1024.0        # ordinary power still works
    assert safe_eval("10 ** 10 ** 8") is None    # MUST return (not hang) — capped exponent
    assert safe_eval("10 ** 1000") is None       # overflow caught fast (exp within cap)
    assert safe_eval("10 ** 2000") is None       # exponent above cap → rejected

    # ── _parse_formulas tolerance (must never crash, returns {} on junk) ──
    assert _parse_formulas('{"a": "1+1"}') == {"a": "1+1"}
    assert _parse_formulas('```json\n{"a": "1+1"}\n```') == {"a": "1+1"}     # code fence
    assert _parse_formulas('Sure, here you go:\n{"a": "1+1"}\nDone.') == {"a": "1+1"}
    assert _parse_formulas("no json here") == {}
    assert _parse_formulas("") == {}
    assert _parse_formulas("{}") == {}           # bare {} → no calc needed, honoured
    assert _parse_formulas("[1, 2, 3]") == {}    # JSON array, not an object

    # balanced extraction: trailing prose with a stray brace must NOT drop the formulas
    assert _parse_formulas('{"revenue_pct": "(8000-7575)/abs(7575)*100"}\n'
                           'Note: use abs() for {old}.') == \
        {"revenue_pct": "(8000-7575)/abs(7575)*100"}
    assert _parse_formulas('```json\n{"a": "1+1"}\n```\nThanks {everyone}') == {"a": "1+1"}
    assert _parse_formulas('{"a": "1"}\n{"b": "2"}') == {"a": "1"}   # first object wins
    assert _parse_formulas('Here is {junk}: {"a": "1+1"}') == {"a": "1+1"}  # skip non-dict brace
    assert _parse_formulas('{"k": "a{b}c"}') == {"k": "a{b}c"}       # brace inside a string value

    # ── end-to-end: parse formulas then evaluate them ──
    formulas = _parse_formulas('{"pct": "(8000-7575)/abs(7575)*100", "diff": "8000-7575"}')
    computed = {k: round(safe_eval(str(v)), 6) for k, v in formulas.items()
                if safe_eval(str(v)) is not None}
    assert computed["diff"] == 425.0 and abs(computed["pct"] - 5.6106) < 1e-3, computed

    print("safe_eval OK")


if __name__ == "__main__":
    main()
