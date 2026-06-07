import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark_results"


@dataclass(frozen=True)
class BenchmarkTest:
    name: str
    command: list[str]
    cwd: Path
    requires_llm: bool = True


BENCHMARK_TESTS = {
    "smoke_embeddings": BenchmarkTest(
        name="smoke_embeddings",
        command=[sys.executable, str(REPO_ROOT / "scripts" / "test_embeddings.py")],
        cwd=REPO_ROOT,
        requires_llm=False,
    ),
    "smoke_vllm": BenchmarkTest(
        name="smoke_vllm",
        command=[sys.executable, str(REPO_ROOT / "scripts" / "test_vllm.py")],
        cwd=REPO_ROOT,
    ),
    "retrieval_app_malt": BenchmarkTest(
        name="retrieval_app_malt",
        command=[
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_retrieval.py"),
            "--app",
            "app-malt",
        ],
        cwd=REPO_ROOT,
        requires_llm=False,
    ),
    "retrieval_app_crg": BenchmarkTest(
        name="retrieval_app_crg",
        command=[
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_retrieval.py"),
            "--app",
            "app-CRG",
        ],
        cwd=REPO_ROOT,
        requires_llm=False,
    ),
    "retrieval_app_traffic_analysis": BenchmarkTest(
        name="retrieval_app_traffic_analysis",
        command=[
            sys.executable,
            str(REPO_ROOT / "scripts" / "test_retrieval.py"),
            "--app",
            "app-traffic-analysis",
        ],
        cwd=REPO_ROOT,
        requires_llm=False,
    ),
    "app_malt_full": BenchmarkTest(
        name="app_malt_full",
        command=[sys.executable, "full_cot_with_tools.py"],
        cwd=REPO_ROOT / "app-malt",
    ),
    "app_crg_full": BenchmarkTest(
        name="app_crg_full",
        command=[sys.executable, "full_cot_with_tools.py"],
        cwd=REPO_ROOT / "app-CRG",
    ),
    "app_traffic_analysis_full": BenchmarkTest(
        name="app_traffic_analysis_full",
        command=[sys.executable, "full_cot_with_tools.py"],
        cwd=REPO_ROOT / "app-traffic-analysis",
    ),
}

DEFAULT_TEST_ORDER = [
    "smoke_embeddings",
    "smoke_vllm",
    "retrieval_app_malt",
    "retrieval_app_crg",
    "retrieval_app_traffic_analysis",
    "app_malt_full",
    "app_crg_full",
    "app_traffic_analysis_full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing MeshAgent tests repeatedly across open-source LLM "
            "models and store organized benchmark results."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file with a `models` list. See benchmark_models.example.json.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Model id to benchmark. Can be repeated. Ignored when --config is used. "
            "Uses the current LLM_BASE_URL and LLM_API_KEY."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        help="OpenAI-compatible base URL used with --model entries.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", "EMPTY"),
        help="API key used with --model entries.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where benchmark runs are written.",
    )
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=DEFAULT_TEST_ORDER + ["all"],
        default=["all"],
        help="Subset of new benchmark wrappers to run. Defaults to all.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Timeout per test invocation.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running remaining tests after a failure.",
    )
    parser.add_argument(
        "--skip-non-llm-tests-after-first-model",
        action="store_true",
        help=(
            "Run embedding and retrieval smoke tests only for the first model, "
            "because they do not depend on LLM_MODEL."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the model/test matrix without executing any subprocesses.",
    )
    parser.add_argument(
        "--serve-models-sequentially",
        action="store_true",
        help=(
            "Start one vLLM server per model, run that model's tests, stop it, "
            "then continue with the next model. Useful for one-GPU benchmarks."
        ),
    )
    parser.add_argument(
        "--serve-host",
        default="0.0.0.0",
        help="Host passed to vLLM when --serve-models-sequentially is used.",
    )
    parser.add_argument(
        "--serve-client-host",
        default="localhost",
        help="Host used by tests to call the sequential vLLM server.",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8000,
        help="Port used for each sequential vLLM server.",
    )
    parser.add_argument(
        "--serve-start-timeout-seconds",
        type=int,
        default=1200,
        help="Maximum time to wait for each vLLM server to become ready.",
    )
    parser.add_argument(
        "--vllm-python",
        default=sys.executable,
        help="Python executable used to start vLLM in sequential serving mode.",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default="",
        help=(
            "Extra arguments appended to `python -m vllm.entrypoints.openai.api_server`. "
            "Pass them as one quoted string."
        ),
    )
    return parser.parse_args()


def _slug(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return clean.strip("._") or "unnamed"


def load_models(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.config:
        with args.config.open("r", encoding="utf-8") as file:
            data = json.load(file)
        models = data.get("models")
        if not isinstance(models, list) or not models:
            raise ValueError("Config file must contain a non-empty `models` list.")
        normalized = []
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                raise ValueError(f"Model entry {index} must be an object.")
            model_id = model.get("llm_model") or model.get("model")
            if not model_id:
                raise ValueError(f"Model entry {index} is missing `llm_model`.")
            normalized.append(
                {
                    "name": str(model.get("name") or model_id),
                    "llm_model": str(model_id),
                    "llm_base_url": str(
                        model.get("llm_base_url")
                        or model.get("base_url")
                        or args.base_url
                    ),
                    "llm_api_key": str(
                        model.get("llm_api_key")
                        or model.get("api_key")
                        or args.api_key
                    ),
                }
            )
        return normalized

    if not args.model:
        raise ValueError("Provide at least one --model or a --config file.")

    return [
        {
            "name": model,
            "llm_model": model,
            "llm_base_url": args.base_url,
            "llm_api_key": args.api_key,
        }
        for model in args.model
    ]


def selected_tests(args: argparse.Namespace) -> list[BenchmarkTest]:
    names = DEFAULT_TEST_ORDER if "all" in args.tests else args.tests
    return [BENCHMARK_TESTS[name] for name in names]


def parse_metrics(output: str) -> dict[str, Any]:
    accuracies = []
    skipped_accuracy = 0
    for raw_value in re.findall(r"Testing accuracy:\s*([^\n\r]+)", output):
        value = raw_value.strip()
        if value.startswith("skipped"):
            skipped_accuracy += 1
            continue
        try:
            accuracies.append(float(value))
        except ValueError:
            pass

    return {
        "accuracy_values": accuracies,
        "accuracy_mean": sum(accuracies) / len(accuracies) if accuracies else None,
        "accuracy_count": len(accuracies),
        "accuracy_skipped_count": skipped_accuracy,
        "pass_count": len(re.findall(r"\bPass the test!\b", output)),
        "fail_count": len(re.findall(r"\bFail the test\b", output)),
        "unsupported_count": len(re.findall(r"Un-support ground truth|Skip: no golden answer", output)),
        "query_count": len(re.findall(r"^Query:\s", output, flags=re.MULTILINE)),
    }


def sequential_base_url(args: argparse.Namespace) -> str:
    return f"http://{args.serve_client_host}:{args.serve_port}/v1"


def fetch_served_model_ids(base_url: str) -> list[str]:
    models_url = f"{base_url.rstrip('/')}/models"
    with urllib.request.urlopen(models_url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = payload.get("data", [])
    if not isinstance(data, list):
        return []
    return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]


def wait_for_server(
    base_url: str,
    process: subprocess.Popen,
    timeout_seconds: int,
    expected_model: str,
) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    last_error = None

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming ready with code {process.returncode}.")
        try:
            served_model_ids = fetch_served_model_ids(base_url)
            if expected_model in served_model_ids:
                return served_model_ids
            last_error = (
                f"{base_url.rstrip('/')}/models responded, but served "
                f"{served_model_ids}; waiting for {expected_model}."
            )
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(5)

    raise TimeoutError(
        f"Timed out waiting for {base_url.rstrip('/')}/models to serve "
        f"{expected_model}. Last error: {last_error}"
    )


def wait_for_server_shutdown(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            fetch_served_model_ids(base_url)
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError):
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {base_url} to stop responding.")


def ensure_server_not_running(base_url: str) -> None:
    try:
        served_model_ids = fetch_served_model_ids(base_url)
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError):
        return
    raise RuntimeError(
        f"{base_url} is already serving models {served_model_ids}. "
        "Stop the existing server before using --serve-models-sequentially."
    )


def start_vllm_server(
    model: dict[str, str],
    model_dir: Path,
    args: argparse.Namespace,
) -> tuple[subprocess.Popen, Any, Any]:
    server_dir = model_dir / "vllm_server"
    server_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = (server_dir / "stdout.txt").open("w", encoding="utf-8")
    stderr_file = (server_dir / "stderr.txt").open("w", encoding="utf-8")

    command = [
        args.vllm_python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model["llm_model"],
        "--host",
        args.serve_host,
        "--port",
        str(args.serve_port),
        *shlex.split(args.vllm_extra_args),
    ]
    ensure_server_not_running(sequential_base_url(args))
    metadata = {
        "model_name": model["name"],
        "llm_model": model["llm_model"],
        "command": command,
        "base_url": sequential_base_url(args),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_path": str(server_dir / "stdout.txt"),
        "stderr_path": str(server_dir / "stderr.txt"),
    }
    (server_dir / "server.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        start_new_session=True,
    )
    served_model_ids = wait_for_server(
        sequential_base_url(args),
        process,
        args.serve_start_timeout_seconds,
        model["llm_model"],
    )
    metadata["ready_at"] = datetime.now(timezone.utc).isoformat()
    metadata["served_model_ids"] = served_model_ids
    (server_dir / "server.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return process, stdout_file, stderr_file


def stop_vllm_server(
    process: subprocess.Popen,
    stdout_file: Any,
    stderr_file: Any,
    base_url: str,
) -> None:
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
        wait_for_server_shutdown(base_url)
    finally:
        stdout_file.close()
        stderr_file.close()


def run_test(
    model: dict[str, str],
    test: BenchmarkTest,
    model_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    test_dir = model_dir / test.name
    test_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LLM_MODEL"] = model["llm_model"]
    env["LLM_BASE_URL"] = model["llm_base_url"]
    env["LLM_API_KEY"] = model["llm_api_key"]

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    timed_out = False
    try:
        completed = subprocess.run(
            test.command,
            cwd=test.cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

    duration_seconds = round(time.monotonic() - started, 3)
    ended_at = datetime.now(timezone.utc).isoformat()
    combined_output = f"{stdout}\n{stderr}"
    metrics = parse_metrics(combined_output)

    (test_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (test_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

    result = {
        "model_name": model["name"],
        "llm_model": model["llm_model"],
        "llm_base_url": model["llm_base_url"],
        "test": test.name,
        "command": test.command,
        "cwd": str(test.cwd),
        "requires_llm": test.requires_llm,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "success": returncode == 0 and not timed_out,
        "metrics": metrics,
        "stdout_path": str(test_dir / "stdout.txt"),
        "stderr_path": str(test_dir / "stderr.txt"),
    }
    (test_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = run_dir / "summary.csv"
    fields = [
        "model_name",
        "llm_model",
        "test",
        "success",
        "returncode",
        "timed_out",
        "duration_seconds",
        "query_count",
        "accuracy_count",
        "accuracy_mean",
        "accuracy_skipped_count",
        "pass_count",
        "fail_count",
        "unsupported_count",
        "stdout_path",
        "stderr_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            writer.writerow(
                {
                    "model_name": result["model_name"],
                    "llm_model": result["llm_model"],
                    "test": result["test"],
                    "success": result["success"],
                    "returncode": result["returncode"],
                    "timed_out": result["timed_out"],
                    "duration_seconds": result["duration_seconds"],
                    "query_count": metrics["query_count"],
                    "accuracy_count": metrics["accuracy_count"],
                    "accuracy_mean": metrics["accuracy_mean"],
                    "accuracy_skipped_count": metrics["accuracy_skipped_count"],
                    "pass_count": metrics["pass_count"],
                    "fail_count": metrics["fail_count"],
                    "unsupported_count": metrics["unsupported_count"],
                    "stdout_path": result["stdout_path"],
                    "stderr_path": result["stderr_path"],
                }
            )


def main() -> int:
    args = parse_args()
    models = load_models(args)
    tests = selected_tests(args)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "tests": [test.name for test in tests],
        "timeout_seconds": args.timeout_seconds,
        "serve_models_sequentially": args.serve_models_sequentially,
        "sequential_base_url": sequential_base_url(args) if args.serve_models_sequentially else None,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"Dry run manifest written to: {run_dir / 'manifest.json'}")
        for model in models:
            print(f"== Model: {model['name']} ({model['llm_model']}) ==")
            if args.serve_models_sequentially:
                print(
                    "Sequential server: "
                    f"{args.vllm_python} -m vllm.entrypoints.openai.api_server "
                    f"--model {model['llm_model']} --host {args.serve_host} "
                    f"--port {args.serve_port} {args.vllm_extra_args}".strip()
                )
            for test in tests:
                print(f"- {test.name}: cwd={test.cwd} command={' '.join(test.command)}")
        return 0

    results = []
    for model_index, model in enumerate(models):
        model_dir = run_dir / _slug(model["name"])
        model_dir.mkdir(parents=True, exist_ok=True)
        active_model = dict(model)
        server_process = None
        server_stdout = None
        server_stderr = None
        print(f"== Model: {model['name']} ({model['llm_model']}) ==")

        if args.serve_models_sequentially:
            active_model["llm_base_url"] = sequential_base_url(args)
            active_model["llm_api_key"] = model.get("llm_api_key") or args.api_key
            print(f"Starting vLLM on {active_model['llm_base_url']}...")
            try:
                server_process, server_stdout, server_stderr = start_vllm_server(
                    active_model,
                    model_dir,
                    args,
                )
                print("vLLM is ready.")
            except Exception as exc:
                error_path = model_dir / "vllm_server_start_error.txt"
                error_path.write_text(str(exc), encoding="utf-8")
                print(f"Failed to start vLLM for {model['name']}: {exc}")
                if not args.continue_on_failure:
                    return 1
                continue

        try:
            for test in tests:
                if (
                    args.skip_non_llm_tests_after_first_model
                    and model_index > 0
                    and not test.requires_llm
                ):
                    print(f"Skipping {test.name}: does not depend on LLM_MODEL.")
                    continue

                print(f"Running {test.name}...")
                result = run_test(active_model, test, model_dir, args.timeout_seconds)
                results.append(result)
                write_summary(run_dir, results)

                status = "ok" if result["success"] else "failed"
                print(
                    f"Finished {test.name}: {status}, "
                    f"returncode={result['returncode']}, "
                    f"duration={result['duration_seconds']}s"
                )

                if not result["success"] and not args.continue_on_failure:
                    print("Stopping after failure. Use --continue-on-failure to keep going.")
                    return 1
        finally:
            if server_process is not None and server_stdout is not None and server_stderr is not None:
                print(f"Stopping vLLM for {model['name']}...")
                stop_vllm_server(
                    server_process,
                    server_stdout,
                    server_stderr,
                    sequential_base_url(args),
                )

    write_summary(run_dir, results)
    print(f"Benchmark results written to: {run_dir}")
    print(f"Summary CSV: {run_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
