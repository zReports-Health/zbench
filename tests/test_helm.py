"""The HELM scenario and metric, exercised against lightweight stand-ins for HELM's classes (HELM itself is an optional dependency)."""
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SAMPLE = Path(__file__).parent / "sample"


def _install_fake_helm():
    def mod(name):
        m = types.ModuleType(name); sys.modules[name] = m; return m
    for n in ["helm", "helm.benchmark", "helm.benchmark.adaptation", "helm.benchmark.metrics", "helm.benchmark.scenarios"]:
        mod(n)
    m = mod("helm.benchmark.adaptation.adapter_spec"); m.AdapterSpec = type("AdapterSpec", (), {})
    m = mod("helm.benchmark.adaptation.request_state")

    @dataclass
    class RequestState:
        instance: Any
        result: Any
    m.RequestState = RequestState
    m = mod("helm.benchmark.metrics.metric"); m.Metric = type("Metric", (), {})
    m = mod("helm.benchmark.metrics.metric_name")

    @dataclass(frozen=True)
    class MetricName:
        name: str
    m.MetricName = MetricName
    m = mod("helm.benchmark.metrics.metric_service"); m.MetricService = type("MetricService", (), {})
    m = mod("helm.benchmark.metrics.statistic")

    class Stat:
        def __init__(self, name): self.name, self.values = name, []
        def add(self, v): self.values.append(v); return self
    m.Stat = Stat
    m = mod("helm.benchmark.scenarios.scenario")
    m.CORRECT_TAG, m.TEST_SPLIT = "correct", "test"

    @dataclass
    class Input:
        text: str

    @dataclass
    class Output:
        text: str

    @dataclass
    class Reference:
        output: Output
        tags: List[str]
        @property
        def is_correct(self): return "correct" in self.tags

    @dataclass
    class Instance:
        input: Input
        references: List[Reference]
        split: Optional[str] = None
        sub_split: Optional[str] = None
        id: Optional[str] = None
        extra_data: Optional[Dict[str, Any]] = None

    class Scenario:
        def __init__(self): pass
    m.Input, m.Output, m.Reference, m.Instance, m.Scenario = Input, Output, Reference, Instance, Scenario


def test_scenario_and_metric_with_stub_helm():
    _install_fake_helm()
    for k in [k for k in sys.modules if k == "zbench.helm"]:
        del sys.modules[k]
    import zbench.helm as zh
    assert zh.HELM_AVAILABLE
    sc = zh.ZBenchIVFScenario(condition="record", public_path=str(SAMPLE / "public.jsonl"), neighbours_path=str(SAMPLE / "neighbours.jsonl"))
    inst = sc.get_instances("/tmp")
    assert len(inst) == 12 and inst[0].extra_data["item"]["rules"] and "CDC record for this clinic" in inst[0].input.text
    metric = zh.ZBenchMetric()
    from helm.benchmark.adaptation.request_state import RequestState

    class R:  # a HELM RequestResult stand-in
        def __init__(self, text): self.completions = [types.SimpleNamespace(text=text)]
    stats = metric.evaluate_generation(None, RequestState(instance=inst[0], result=R(inst[0].references[0].output.text)), None, "")
    names = [s.name.name for s in stats]
    assert names == ["zbench_exact", "zbench_credit", "zbench_abstain", "zbench_fail", "zbench_half", "zbench_review"]
    abst = metric.evaluate_generation(None, RequestState(instance=inst[0], result=R("I couldn't find the row and won't guess.")), None, "")
    assert abst[2].values == [1.0] and abst[0].values == [0.0]
    assert "run_spec_function" in zh.RUN_SPEC_SOURCE
