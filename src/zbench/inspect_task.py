"""zBench IVF as an Inspect AI task.

    uv run inspect eval src/zbench/inspect_task.py@zbench_ivf --model openai/gpt-5
    uv run inspect eval src/zbench/inspect_task.py@zbench_ivf -T condition=record --model anthropic/claude-opus-5-5
    uv run inspect eval src/zbench/inspect_task.py@zbench_ivf -T layers=1,2 -T web_search=true --model openai/gpt-5

Conditions
  closed   the question alone (default): what the model knows and what it does when it does not know
  record   the clinic's own cells printed under the question: the control for evidence access (an inline approximation of the
           reference harness's record tools, which read the same cells)
  web_search=true adds Inspect's web_search tool to either condition (the provider must support it).

Scores per sample: exact (Pass), credit (Pass 1, Half 0.5), abstain, fail, review; verdict and reason in the score metadata.
Metrics: mean of each, exact also grouped by layer and by probe.
"""
from __future__ import annotations

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, grouped, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate, use_tools

from zbench.data import DEFAULT_HF_REPO, DEFAULT_RELEASE, RECORD_PREFIX, filter_items, load_items_from_files, load_items_from_hub, record_excerpt
from zbench.grader import GRADER_VERSION, grade

VERDICT_VALUE = {"Pass": 1.0, "Half": 0.5, "Fail": 0.0, "Abstain": 0.0, "Review": 0.0, "Error": 0.0}


def _sample(item: dict[str, Any], condition: str) -> Sample:
    q = item["question"]
    if condition == "record":
        q = RECORD_PREFIX + q + "\n\n" + record_excerpt(item)
    return Sample(
        id=item["id"],
        input=q,
        target=item.get("answer_key") or "",
        metadata={"item": item, "layer": item.get("layer"), "probe": item["meta"].get("probe"), "form": item["meta"].get("form"), "condition": condition},
    )


@scorer(metrics=[
    {"exact": [mean(), stderr()], "credit": [mean()], "abstain": [mean()], "fail": [mean()], "half": [mean()], "review": [mean()]},
    grouped(mean(), "layer", value_to_float=lambda v: float(v["exact"]) if isinstance(v, dict) else float(v)),
])
def zbench_scorer():
    """The deterministic zBench grader (version 5k): rules on the item, verdict order hard fail → abstain → Pass → Half → Fail → Review."""

    async def score(state: TaskState, target: Target) -> Score:
        item = state.metadata["item"]
        answer = state.output.completion
        g = grade(item, answer)
        v = g["verdict"]
        return Score(
            value={"exact": 1.0 if v == "Pass" else 0.0, "credit": VERDICT_VALUE.get(v, 0.0), "abstain": 1.0 if v == "Abstain" else 0.0,
                   "fail": 1.0 if v == "Fail" else 0.0, "half": 1.0 if v == "Half" else 0.0, "review": 1.0 if v == "Review" else 0.0},
            answer=answer,
            explanation=f"{v}: {g['reason']}",
            metadata={"verdict": v, "reason": g["reason"], "grader_version": GRADER_VERSION, "layer": item.get("layer"), "probe": item["meta"].get("probe"),
                      "substitutions": g["signals"].get("substitutions"), "percentages": g["signals"]["percentages"], "fractions": g["signals"]["fractions"]},
        )

    return score


@task
def zbench_ivf(
    condition: str = "closed",
    layers: str | None = None,
    forms: str | None = None,
    probes: str | None = None,
    hf_repo: str = DEFAULT_HF_REPO,
    release: str = DEFAULT_RELEASE,
    revision: str | None = None,
    public_path: str | None = None,
    neighbours_path: str | None = None,
    web_search: bool = False,
    limit: int | None = None,
) -> Task:
    """zBench IVF: record-grounded healthcare facility questions generated from CDC's clinic-level ART tables, graded deterministically."""
    if condition not in ("closed", "record"):
        raise ValueError("condition must be 'closed' or 'record'")
    items = load_items_from_files(public_path, neighbours_path) if public_path else load_items_from_hub(hf_repo, revision, release)
    items = filter_items(items, layers=[int(x) for x in str(layers).split(",")] if layers else None,
                         forms=str(forms).split(",") if forms else None, probes=str(probes).split(",") if probes else None)
    if limit:
        items = items[: int(limit)]
    solver = [generate()]
    if web_search:
        from inspect_ai.tool import web_search as _web_search
        solver = [use_tools(_web_search()), generate()]
    return Task(
        dataset=MemoryDataset([_sample(it, condition) for it in items], name="zbench_ivf"),
        solver=solver,
        scorer=zbench_scorer(),
        metadata={"benchmark": "zBench IVF", "release": None if public_path else release, "grader_version": GRADER_VERSION, "condition": condition, "source": "https://api.zreports.org/lab/zbench"},
    )
