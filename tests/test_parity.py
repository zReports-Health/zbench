"""The Python grader must read every stored answer exactly as the reference JavaScript grader (5k) did: verdict, reason, the numbers
extracted, the caveat flags and the substitution labels. The fixture was produced by the reference implementation over 717 real
model answers (runs 5, 8, 9, 10 and 12 of zBench IVF plus the regression fixtures)."""
import json
from pathlib import Path

import pytest

from zbench.grader import auto_verdict, plain, signals, substitutions

FIXTURE = Path(__file__).parent / "fixtures" / "parity.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _norm_num(p):
    return int(p) if float(p).is_integer() else p


def _norm_subs(s):
    if s is None:
        return None
    return [{"value": _norm_num(x["value"]) if isinstance(x["value"], (int, float)) else x["value"], "rel": x["rel"], "year": x.get("year"),
             "measure": x.get("measure"), "band": x.get("band"), "attributed": bool(x.get("attributed")), "illustrative": bool(x.get("illustrative"))} for x in s]


@pytest.mark.parametrize("case", CASES, ids=[f"{c['src']}:{c['id']}:{i}" for i, c in enumerate(CASES)])
def test_parity(case):
    sig = signals(case["answer"])
    subs = substitutions(sig, case["item"], plain(case["answer"]))
    v = auto_verdict(case["item"], case["answer"], subs)
    e = case["expected"]
    assert v["verdict"] == e["verdict"]
    assert v["reason"] == e["reason"]
    assert [_norm_num(p) for p in sig["percentages"]] == [_norm_num(p) for p in e["percentages"]]
    assert sig["fractions"] == e["fractions"]
    for k in ("asserts_zero", "caveat_small_sample", "caveat_suppressed", "caveat_no_cycles", "caveat_no_rank"):
        assert bool(sig[k]) == bool(e[k]), k
    assert _norm_subs(subs) == _norm_subs(e["substitutions"])


def test_fixture_covers_every_verdict():
    verdicts = {c["expected"]["verdict"] for c in CASES}
    assert {"Pass", "Half", "Fail", "Abstain", "Review"} <= verdicts
    assert len(CASES) >= 700
