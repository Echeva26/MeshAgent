# Benchmarking Open-Source Models

This repository now includes a new benchmark wrapper that runs the existing tests repeatedly across multiple OpenAI-compatible open-source model endpoints. The old tests are not modified. The benchmark runner invokes them as subprocesses, captures their output, parses the visible pass/fail/accuracy lines, and stores everything in an organized results directory.

## New Files

- `scripts/benchmark_opensource_models.py` - benchmark runner.
- `scripts/benchmark_models.example.json` - example model matrix.
- `BENCHMARKING_OPEN_SOURCE_MODELS.md` - this usage guide.

## What the Runner Executes

The runner wraps these existing tests and app entrypoints:

- `scripts/test_embeddings.py`
- `scripts/test_vllm.py`
- `scripts/test_retrieval.py --app app-malt`
- `scripts/test_retrieval.py --app app-CRG`
- `scripts/test_retrieval.py --app app-traffic-analysis`
- `app-malt/full_cot_with_tools.py`
- `app-CRG/full_cot_with_tools.py`
- `app-traffic-analysis/full_cot_with_tools.py`

Each wrapped test is executed in a separate subprocess with model-specific environment variables:

- `LLM_MODEL`
- `LLM_BASE_URL`
- `LLM_API_KEY`

## Requirements

The runner expects each model to already be served through an OpenAI-compatible API, for example with vLLM. It does not start or stop vLLM servers because model loading depends on local GPU availability and deployment preferences.

Example server command:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

For multiple models, either run one server per model on different ports or run the benchmark once per served model.

## Running a Benchmark

Using explicit model arguments:

```bash
python3 scripts/benchmark_opensource_models.py \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --continue-on-failure
```

Using a model matrix:

```bash
python3 scripts/benchmark_opensource_models.py \
  --config scripts/benchmark_models.example.json \
  --continue-on-failure
```

Running only selected wrappers:

```bash
python3 scripts/benchmark_opensource_models.py \
  --config scripts/benchmark_models.example.json \
  --tests smoke_vllm app_malt_full app_traffic_analysis_full \
  --continue-on-failure
```

For long app runs, adjust the per-test timeout:

```bash
python3 scripts/benchmark_opensource_models.py \
  --config scripts/benchmark_models.example.json \
  --timeout-seconds 7200 \
  --continue-on-failure
```

Checking the benchmark matrix without executing tests:

```bash
python3 scripts/benchmark_opensource_models.py \
  --config scripts/benchmark_models.example.json \
  --dry-run
```

## Result Layout

Each run is stored under:

```text
benchmark_results/YYYYMMDD_HHMMSS/
```

The run directory contains:

- `manifest.json` - models, selected tests, timeout, and run metadata.
- `summary.json` - full structured result records.
- `summary.csv` - compact table for quick comparison.
- one folder per model.
- one folder per test inside each model folder.

Each test folder contains:

- `stdout.txt`
- `stderr.txt`
- `result.json`

## Parsed Metrics

The runner parses these values from the existing test output:

- `query_count`
- `accuracy_values`
- `accuracy_mean`
- `accuracy_count`
- `accuracy_skipped_count`
- `pass_count`
- `fail_count`
- `unsupported_count`
- process `returncode`
- `duration_seconds`
- timeout status

This keeps the old tests untouched while still producing benchmark-friendly summaries.

## Notes

Embedding and retrieval smoke tests do not depend on `LLM_MODEL`. If you are comparing many models and want to avoid repeating those checks, use:

```bash
python3 scripts/benchmark_opensource_models.py \
  --config scripts/benchmark_models.example.json \
  --skip-non-llm-tests-after-first-model \
  --continue-on-failure
```

The app entrypoints still write their legacy JSONL logs to their original locations. The benchmark runner additionally captures stdout and stderr per model/test so the benchmark output remains organized even though the old scripts keep their historical logging behavior.
