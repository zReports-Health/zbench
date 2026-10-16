"""Grade a file of model outputs against a zBench release, offline.

    python -m zbench.cli outputs.jsonl --public zbench-ivf-1.2-public.jsonl --neighbours zbench-ivf-1.2-neighbours.jsonl [--report report.json]

``outputs.jsonl``: one JSON object per line with ``id`` (the item id) and ``answer`` (the model's text); optional ``citations`` as a list
of ``{"url": ...}``. Prints the scorecard by layer and by probe, and writes every verdict with its reason when ``--report`` is given.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import Any

from zbench.data import DEFAULT_HF_REPO, DEFAULT_RELEASE, load_items_from_files, load_items_from_hub
from zbench.grader import GRADER_VERSION, grade


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="zbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", help="JSONL of {id, answer[, citations]}")
    ap.add_argument("--public", help="the release's public file, e.g. zbench-ivf-1.2-public.jsonl (default: download from the Hugging Face repo)")
    ap.add_argument("--neighbours", help="the release's neighbours file, e.g. zbench-ivf-1.2-neighbours.jsonl")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"Hugging Face dataset repo, used when --public is not given (default {DEFAULT_HF_REPO})")
    ap.add_argument("--release", default=DEFAULT_RELEASE, help=f"release id, the prefix of the file names on the hub (default {DEFAULT_RELEASE})")
    ap.add_argument("--report", help="write every verdict with its reason and signals to this JSON file")
    args = ap.parse_args(argv)

    items = load_items_from_files(args.public, args.neighbours) if args.public else load_items_from_hub(args.hf_repo, release=args.release)
    by_id = {it["id"]: it for it in items}
    rows: list[dict[str, Any]] = []
    missing = 0
    with open(args.outputs, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            it = by_id.get(o.get("id"))
            if not it:
                missing += 1
                continue
            g = grade(it, o.get("answer"), o.get("citations"))
            rows.append({"id": it["id"], "layer": it.get("layer"), "probe": it["meta"].get("probe"), "verdict": g["verdict"], "reason": g["reason"], "signals": g["signals"]})

    def tally(keyfn):
        t: dict[Any, collections.Counter] = collections.defaultdict(collections.Counter)
        for r in rows:
            t[keyfn(r)][r["verdict"]] += 1
        return t

    def line(name: str, c: collections.Counter) -> str:
        n = sum(c.values())
        exact = 100.0 * c["Pass"] / n if n else 0.0
        credit = 100.0 * (c["Pass"] + 0.5 * c["Half"]) / n if n else 0.0
        return f"{name:<58} n={n:<5} Pass={c['Pass']:<4} Half={c['Half']:<4} Fail={c['Fail']:<4} Abstain={c['Abstain']:<4} Review={c['Review']:<3} Error={c['Error']:<3} exact={exact:5.1f}% credit={credit:5.1f}%"

    total = collections.Counter(r["verdict"] for r in rows)
    print(f"zBench grader {GRADER_VERSION}: {len(rows)} answers graded" + (f", {missing} ids not in the release" if missing else ""))
    print(line("all", total))
    print("by layer")
    for layer, c in sorted(tally(lambda r: r["layer"]).items(), key=lambda kv: str(kv[0])):
        print("  " + line(f"layer {layer}", c))
    print("by probe")
    for probe, c in sorted(tally(lambda r: r["probe"]).items(), key=lambda kv: str(kv[0])):
        print("  " + line(str(probe), c))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"grader_version": GRADER_VERSION, "n": len(rows), "totals": dict(total), "answers": rows}, fh, indent=1, default=list)
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
