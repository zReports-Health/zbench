"""The Inspect task loads a release, builds samples, and scores an answer with the grader (no model, no network)."""
import asyncio
from pathlib import Path

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from zbench.data import load_items_from_files
from zbench.inspect_task import zbench_ivf, zbench_scorer

SAMPLE = Path(__file__).parent / "sample"


def _state(sample, completion: str) -> TaskState:
    return TaskState(model="mockllm/model", sample_id=sample.id, epoch=0, input=sample.input, messages=[], output=ModelOutput.from_content("mockllm/model", completion), metadata=sample.metadata)


def test_loader_joins_neighbours():
    items = load_items_from_files(SAMPLE / "public.jsonl", SAMPLE / "neighbours.jsonl")
    assert len(items) == 12
    assert all(it["rules"].get("pass_all") for it in items)
    assert any(it["meta"]["neighbours"] for it in items)


def test_task_builds_and_scores():
    task = zbench_ivf(public_path=str(SAMPLE / "public.jsonl"), neighbours_path=str(SAMPLE / "neighbours.jsonl"), limit=5)
    samples = list(task.dataset)
    assert len(samples) == 5 and samples[0].metadata["item"]["rules"]
    scorer = zbench_scorer()
    s = samples[0]
    key = s.metadata["item"]["answer_key"]
    # the answer key restated is graded; an "I couldn't find it" is an abstention; an empty answer is an error
    scored = asyncio.run(scorer(_state(s, key), Target(key)))
    assert scored.metadata["verdict"] in ("Pass", "Half", "Fail", "Abstain", "Review")
    abst = asyncio.run(scorer(_state(s, "I couldn't find this cell on CDC's site and won't guess."), Target(key)))
    assert abst.metadata["verdict"] == "Abstain" and abst.value["abstain"] == 1.0 and abst.value["exact"] == 0.0
    err = asyncio.run(scorer(_state(s, ""), Target(key)))
    assert err.metadata["verdict"] == "Error"


def test_record_condition_prints_cells():
    task = zbench_ivf(public_path=str(SAMPLE / "public.jsonl"), neighbours_path=str(SAMPLE / "neighbours.jsonl"), condition="record", limit=1)
    s = list(task.dataset)[0]
    assert "CDC record for this clinic" in s.input and s.metadata["condition"] == "record"
