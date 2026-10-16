"""zBench IVF for HELM / MedHELM: a Scenario, a Metric and a run spec function.

Drop-in paths for a pull request to stanford-crfm/helm:
    helm/benchmark/scenarios/zbench_scenario.py     ← ZBenchIVFScenario (this file's scenario half)
    helm/benchmark/metrics/zbench_metrics.py        ← ZBenchMetric (this file's metric half) + zbench.grader vendored as zbench_grader.py
    helm/benchmark/run_specs/medhelm_run_specs.py   ← get_zbench_ivf_spec appended (see ``RUN_SPEC_SOURCE`` below)

Run (once the files are in place):
    helm-run --run-entries zbench_ivf:condition=closed --suite zbench --max-eval-instances 500 --models-to-run openai/gpt-5

MedHELM category: Administration and Workflow (facility records, reporting periods, measures), not clinical decision support.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from zbench.data import DEFAULT_HF_REPO, DEFAULT_RELEASE, RECORD_PREFIX, filter_items, load_items_from_files, load_items_from_hub, record_excerpt
from zbench.grader import GRADER_VERSION, grade

try:  # HELM is an optional dependency: the scenario and metric only import when HELM is installed
    from helm.benchmark.adaptation.adapter_spec import AdapterSpec
    from helm.benchmark.adaptation.request_state import RequestState
    from helm.benchmark.metrics.metric import Metric
    from helm.benchmark.metrics.metric_name import MetricName
    from helm.benchmark.metrics.metric_service import MetricService
    from helm.benchmark.metrics.statistic import Stat
    from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT, Input, Instance, Output, Reference, Scenario
    HELM_AVAILABLE = True
except Exception:  # pragma: no cover
    HELM_AVAILABLE = False


def _load(condition: str, layers: Optional[str], hf_repo: str, release: str, public_path: Optional[str], neighbours_path: Optional[str]) -> List[Dict[str, Any]]:
    items = load_items_from_files(public_path, neighbours_path) if public_path else load_items_from_hub(hf_repo, release=release)
    return filter_items(items, layers=[int(x) for x in str(layers).split(",")] if layers else None)


def instance_input(item: Dict[str, Any], condition: str) -> str:
    q = item["question"]
    return RECORD_PREFIX + q + "\n\n" + record_excerpt(item) if condition == "record" else q


if HELM_AVAILABLE:

    class ZBenchIVFScenario(Scenario):
        """zBench IVF: questions generated from CDC's clinic-level ART tables; every answer key is a CDC row."""

        name = "zbench_ivf"
        description = "Record-grounded healthcare facility questions (CDC ART clinic tables 2020 to 2022) with deterministic, row-provenanced grading."
        tags = ["healthcare", "facility_records", "question_answering", "administration_and_workflow"]

        def __init__(self, condition: str = "closed", layers: Optional[str] = None, hf_repo: str = DEFAULT_HF_REPO,
                     release: str = DEFAULT_RELEASE, public_path: Optional[str] = None, neighbours_path: Optional[str] = None):
            super().__init__()
            if condition not in ("closed", "record"):
                raise ValueError("condition must be 'closed' or 'record'")
            self.condition, self.layers, self.hf_repo, self.release = condition, layers, hf_repo, release
            self.public_path = public_path or os.environ.get("ZBENCH_PUBLIC_PATH")
            self.neighbours_path = neighbours_path or os.environ.get("ZBENCH_NEIGHBOURS_PATH")

        def get_instances(self, output_path: str) -> List[Instance]:
            out: List[Instance] = []
            for it in _load(self.condition, self.layers, self.hf_repo, self.release, self.public_path, self.neighbours_path):
                out.append(Instance(
                    input=Input(text=instance_input(it, self.condition)),
                    references=[Reference(Output(text=it.get("answer_key") or ""), tags=[CORRECT_TAG])],
                    split=TEST_SPLIT,
                    id=it["id"],
                    extra_data={"item": it, "layer": it.get("layer"), "probe": it["meta"].get("probe"), "condition": self.condition},
                ))
            return out

    class ZBenchMetric(Metric):
        """The deterministic zBench grader as HELM stats: zbench_exact, zbench_credit, zbench_abstain, zbench_fail, zbench_half, zbench_review."""

        def evaluate_generation(self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService, eval_cache_path: str) -> List[Stat]:
            item = (request_state.instance.extra_data or {}).get("item")
            completions = request_state.result.completions if request_state.result else []
            text = completions[0].text if completions else ""
            g = grade(item, text) if item else {"verdict": "Error", "reason": "no item on the instance"}
            v = g["verdict"]
            return [
                Stat(MetricName("zbench_exact")).add(1.0 if v == "Pass" else 0.0),
                Stat(MetricName("zbench_credit")).add({"Pass": 1.0, "Half": 0.5}.get(v, 0.0)),
                Stat(MetricName("zbench_abstain")).add(1.0 if v == "Abstain" else 0.0),
                Stat(MetricName("zbench_fail")).add(1.0 if v == "Fail" else 0.0),
                Stat(MetricName("zbench_half")).add(1.0 if v == "Half" else 0.0),
                Stat(MetricName("zbench_review")).add(1.0 if v == "Review" else 0.0),
            ]


RUN_SPEC_SOURCE = '''
@run_spec_function("zbench_ivf")
def get_zbench_ivf_spec(condition: str = "closed", layers: Optional[str] = None) -> RunSpec:
    """zBench IVF (zReports HealthCare Data Lab): record-grounded facility questions from CDC's ART clinic tables, graded deterministically."""
    scenario_spec = ScenarioSpec(
        class_name="helm.benchmark.scenarios.zbench_scenario.ZBenchIVFScenario",
        args={"condition": condition, **({"layers": layers} if layers else {})},
    )
    adapter_spec = get_generation_adapter_spec(
        instructions="Answer the question about the healthcare facility record. Give the figure exactly as the source publishes it, with any qualifier the source attaches (small sample, suppressed, no cycles); if the source publishes no figure for the cell, say exactly what it does publish.",
        input_noun=None,
        newline_after_input_noun=False,
        output_noun="Answer",
        max_tokens=600,
        max_train_instances=0,
        stop_sequences=[],
    )
    metric_specs = [MetricSpec(class_name="helm.benchmark.metrics.zbench_metrics.ZBenchMetric", args={})] + get_basic_metric_specs([])
    return RunSpec(
        name=f"zbench_ivf:condition={condition}" + (f",layers={layers}" if layers else ""),
        scenario_spec=scenario_spec,
        adapter_spec=adapter_spec,
        metric_specs=metric_specs,
        groups=["zbench_ivf"],
    )
'''

__all__ = ["HELM_AVAILABLE", "RUN_SPEC_SOURCE", "instance_input", "GRADER_VERSION"] + (["ZBenchIVFScenario", "ZBenchMetric"] if HELM_AVAILABLE else [])
