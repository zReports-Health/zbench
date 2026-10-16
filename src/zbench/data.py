"""Load a zBench release (public split + neighbouring cells) into the item shape the grader reads.

A release ships three files, each named with the release id: ``<release>-public.jsonl`` (one item per line: question, answer key,
rules, the clinic's own cells, provenance), ``<release>-neighbours.jsonl`` (per item, the neighbouring cells of the same clinic with
their substitution labels) and ``<release>-manifest.json`` (counts, dataset ids, grader version, SHA-256 per file). The files are
published on Hugging Face and, without registration, at https://api.zreports.org/lab/zbench; ``release_files`` gives the names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_HF_REPO = os.environ.get("ZBENCH_HF_REPO", "shawnzreports/zbench-ivf")
DEFAULT_RELEASE = os.environ.get("ZBENCH_RELEASE", "zbench-ivf-1.2")
RECORD_PREFIX = (
    "Answer from the CDC record supplied below. Give the figure exactly as CDC publishes it, with its qualifier "
    "(small sample, suppressed, no cycles) when the record carries one; do not substitute another year, age band or measure.\n\n"
)


def _read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def item_from_record(rec: dict[str, Any], neighbours: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A public.jsonl record (plus its neighbours) as the item the grader expects."""
    return {
        "id": rec["id"],
        "layer": rec.get("layer"),
        "band": rec.get("band"),
        "clinic": rec.get("clinic"),
        "year": rec.get("year"),
        "question": rec["question"],
        "answer_key": rec.get("answer_key") or "",
        "rubric": rec.get("rubric") or {},
        "rules": rec.get("rules") or {},
        "verify": rec.get("verify"),
        "meta": {
            "form": rec.get("form"),
            "forms": rec.get("forms"),
            "cells": rec.get("cells") or [],
            "neighbours": list(neighbours or []),
            "probe": rec.get("probe"),
            "template": rec.get("template"),
            "measure": rec.get("measure"),
            "type": rec.get("type"),
            "trap": rec.get("trap"),
        },
    }


def load_items_from_files(public_path: str | Path, neighbours_path: str | Path | None = None) -> list[dict[str, Any]]:
    nb: dict[str, list[dict[str, Any]]] = {}
    if neighbours_path and Path(neighbours_path).exists():
        for rec in _read_jsonl(neighbours_path):
            nb[rec["id"]] = rec.get("neighbours") or []
    return [item_from_record(rec, nb.get(rec["id"])) for rec in _read_jsonl(public_path)]


def release_files(release: str = DEFAULT_RELEASE) -> dict[str, str]:
    """The three file names of a release, as published on Hugging Face and as downloaded from the benchmark page."""
    return {"public": f"{release}-public.jsonl", "neighbours": f"{release}-neighbours.jsonl", "manifest": f"{release}-manifest.json"}


def load_items_from_hub(repo: str = DEFAULT_HF_REPO, revision: str | None = None, release: str = DEFAULT_RELEASE) -> list[dict[str, Any]]:
    """Download a release's public and neighbours files from a Hugging Face dataset repository and join them."""
    from huggingface_hub import hf_hub_download  # imported lazily: the grader itself has no network dependency

    names = release_files(release)
    public = hf_hub_download(repo, names["public"], repo_type="dataset", revision=revision)
    try:
        neighbours = hf_hub_download(repo, names["neighbours"], repo_type="dataset", revision=revision)
    except Exception:  # the neighbours file is optional: without it, substitution labels are not computed
        neighbours = None
    return load_items_from_files(public, neighbours)


def record_excerpt(item: dict[str, Any]) -> str:
    """The clinic's own cells, printed the way the record tools print them (the 'record' condition, inline)."""
    lines = []
    for c in item["meta"].get("cells") or []:
        where = ", ".join(str(x) for x in (c.get("y"), c.get("b"), c.get("m")) if x is not None)
        lines.append(f"- {where}: {c.get('v')}" + (f" (all-ages count {c['n']})" if c.get("n") not in (None, "") else ""))
    return "CDC record for this clinic (as published):\n" + "\n".join(lines) if lines else ""


def filter_items(items: Iterable[dict[str, Any]], layers: Iterable[int] | None = None, forms: Iterable[str] | None = None, probes: Iterable[str] | None = None) -> list[dict[str, Any]]:
    ls = {int(x) for x in layers} if layers else None
    fs = set(forms) if forms else None
    ps = set(probes) if probes else None
    out = []
    for it in items:
        if ls is not None and int(it.get("layer") or 0) not in ls:
            continue
        if fs is not None and (it["meta"].get("form") or "") not in fs:
            continue
        if ps is not None and (it["meta"].get("probe") or "") not in ps:
            continue
        out.append(it)
    return out
