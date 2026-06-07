import argparse
import ast
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIRS = {
    "app-malt": REPO_ROOT / "app-malt",
    "app-CRG": REPO_ROOT / "app-CRG",
    "app-traffic-analysis": REPO_ROOT / "app-traffic-analysis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an app's legacy userQuery prompt list one prompt at a time, "
            "continuing after prompt-level failures."
        )
    )
    parser.add_argument("--app", choices=sorted(APP_DIRS), required=True)
    parser.add_argument("--script", default="full_cot_with_tools.py")
    parser.add_argument("--prompt-limit", type=int)
    return parser.parse_args()


def extract_prompt_list(script_path: Path) -> list[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "main":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id == "prompt_list":
                    prompts = ast.literal_eval(child.value)
                    if not isinstance(prompts, list) or not all(
                        isinstance(prompt, str) for prompt in prompts
                    ):
                        raise ValueError("main.prompt_list must be a list of strings.")
                    return prompts
    raise ValueError(f"Could not find prompt_list in {script_path}.")


def import_app_module(script_path: Path, app_dir: Path):
    os.chdir(app_dir)
    for path in (str(app_dir), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    spec = importlib.util.spec_from_file_location("meshagent_legacy_app", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    app_dir = APP_DIRS[args.app]
    script_path = app_dir / args.script
    prompts = extract_prompt_list(script_path)
    if args.prompt_limit is not None:
        prompts = prompts[: args.prompt_limit]

    failures = 0
    completed = 0

    print(
        "RESILIENT_APP_START "
        + json.dumps({"app": args.app, "script": args.script, "prompt_count": len(prompts)})
    )

    if not prompts:
        print(
            "RESILIENT_APP_SUMMARY "
            + json.dumps(
                {
                    "app": args.app,
                    "prompt_count": 0,
                    "completed_without_exception": 0,
                    "prompt_exceptions": 0,
                }
            )
        )
        return 0

    module = import_app_module(script_path, app_dir)

    for index, prompt in enumerate(prompts, start=1):
        print(
            "RESILIENT_PROMPT_START "
            + json.dumps({"index": index, "total": len(prompts), "query": prompt})
        )
        try:
            module.userQuery([prompt])
        except BaseException as exc:
            failures += 1
            print("Fail the test, prompt raised an exception.")
            print(
                "RESILIENT_PROMPT_EXCEPTION "
                + json.dumps(
                    {
                        "index": index,
                        "query": prompt,
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    }
                )
            )
            traceback.print_exc()
        else:
            completed += 1
        finally:
            print(
                "RESILIENT_PROMPT_END "
                + json.dumps({"index": index, "failures": failures, "completed": completed})
            )

    print(
        "RESILIENT_APP_SUMMARY "
        + json.dumps(
            {
                "app": args.app,
                "prompt_count": len(prompts),
                "completed_without_exception": completed,
                "prompt_exceptions": failures,
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
