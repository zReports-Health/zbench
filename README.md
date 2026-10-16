# zbench

Evaluation code for **zBench**, the record-grounded healthcare facility benchmarks of the zReports HealthCare Data Lab. This repository holds the deterministic grader (a Python port of the reference grader, version 5k, checked against it answer by answer), the data loader for a release, an [Inspect AI](https://inspect.aisi.org.uk) task, and a HELM / MedHELM scenario.

Benchmark page, results, method and keyless downloads: https://api.zreports.org/lab/zbench
Dataset on Hugging Face: https://huggingface.co/datasets/shawnzreports/zbench-ivf (release `zbench-ivf-1.2`: public split, neighbouring cells, manifest).
Paper: *zBench IVF: A Record-Grounded Benchmark for Healthcare Facility Reasoning in Language Models* (WP-2026-07).

## What zBench IVF is

5,144 tasks generated from CDC's clinic level ART success rate tables (reporting years 2020 to 2022): 4,139 public with answer keys, 1,005 held out. Every answer key is a cell of the public record with a row locator that re-derives it from the published file. Seven layers: retrieval, encoding (fractions, suppressed counts, zeros, "no cycles"), evidence, reconciliation against the national row, temporal, no-rank comparison, graph joins. Grading is deterministic: hard fail (a figure the record cannot support) → clarifying question or abstention → Pass → Half → Fail → Review, with every wrong figure labelled by the neighbouring cell it came from.

Results so far (exact answers, layers 1 to 6, browsing → record): GPT-5 25% → 84%, Gemini 3.1 Pro Preview 17% → 92% of completed, Claude Opus 5.5 3% → 77% (173 abstentions while browsing).

## Install

```
git clone https://github.com/zReports-Health/zbench
cd zbench
uv sync            # or: pip install -e .
```

Python 3.10+. Dependencies: `inspect_ai`, `regex`, `huggingface_hub`. HELM is optional (`pip install -e ".[helm]"`).

## Run with Inspect

```
uv run inspect eval src/zbench/inspect_task.py@zbench_ivf --model openai/gpt-5
uv run inspect eval src/zbench/inspect_task.py@zbench_ivf -T condition=record --model anthropic/claude-opus-5-5
uv run inspect eval src/zbench/inspect_task.py@zbench_ivf -T layers=1,2,4 -T web_search=true --model google/gemini-3.1-pro-preview
```

Task arguments: `condition` (`closed`, the question alone; `record`, the clinic's own cells printed under the question, the control for evidence access), `layers`, `forms`, `probes` (comma separated filters), `web_search` (adds Inspect's `web_search` tool), `hf_repo` / `release` / `revision`, or `public_path` / `neighbours_path` for local files, `limit`.

Scores per sample: `exact` (Pass), `credit` (Pass 1, Half 0.5), `abstain`, `fail`, `half`, `review`; the verdict, its reason and the substitution labels sit in the score metadata. Metrics: the mean of each, `exact` grouped by layer.

The reference harness's `record` condition gives the model three tools over the same tables; the Inspect task prints the same cells inline. Numbers from the two are comparable, not identical.

## Grade your own outputs, offline

```
python -m zbench.cli outputs.jsonl --public zbench-ivf-1.2-public.jsonl --neighbours zbench-ivf-1.2-neighbours.jsonl --report report.json
```

`outputs.jsonl` holds one `{"id": ..., "answer": ...}` per line. The scorecard prints by layer and by probe. For a graded run on the held out split (questions sent to you, keys never leave the server) write to shawn@zreports.org.

## HELM / MedHELM

`src/zbench/helm.py` holds `ZBenchIVFScenario`, `ZBenchMetric` and the run spec source for `zbench_ivf`. For a pull request to stanford-crfm/helm the three parts drop into `helm/benchmark/scenarios/zbench_scenario.py`, `helm/benchmark/metrics/zbench_metrics.py` (with `grader.py` vendored beside it) and `helm/benchmark/run_specs/medhelm_run_specs.py`. MedHELM category: Administration and Workflow.

## The grader is the reference grader

`tests/fixtures/parity.json` carries 717 real model answers from the zBench IVF runs (runs 5, 8, 9, 10, 12 and the regression fixtures) with the reference grader's verdict, reason, extracted numbers, caveat flags and substitution labels. `pytest` checks the Python port reproduces every one of them.

```
pytest -q        # 722 passed
```

Rules live on the items (regular expressions the generator writes from the record's cells); the grader applies them in a fixed order. A grader change is versioned; results are reported with the version (`zbench.GRADER_VERSION`).

## Data files

Each release ships three files named with its id (`zbench-ivf-1.2-public.jsonl`, and so on); `zbench.release_files(release)` gives the names.

- `public.jsonl`: id, layer, question, answer key, rubric, rules, probe, the clinic's own cells, the row locator (ClinicId, dataset, filter, question, breakout).
- `neighbours.jsonl`: for each public item, the neighbouring cells with their substitution labels.
- `manifest.json`: counts by layer, form, template and probe; CDC dataset ids; grader version; license; SHA-256 per file.

The loader joins the first two by id. Without the neighbours file the grader still runs, but substitution labels are not computed.

## License

Code: Apache-2.0. The zBench IVF item set is released for model evaluation under the zReports HealthCare Data Lab terms; redistribution of the item set, use as training data, or publication of held out results requires a written license from zReports. Source data: CDC, National ART Surveillance System, data.cdc.gov; CDC publishes its clinic tables with a notice that they are not to be used to rank clinics, and no item asks for a ranking.

## Citation

Redd, S. (2026). zBench IVF: A Record-Grounded Benchmark for Healthcare Facility Reasoning in Language Models. zReports HealthCare Data Lab, WP-2026-07.
