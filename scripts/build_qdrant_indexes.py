import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.rag_indexer import SUPPORTED_APPS, index_apps


def parse_args():
    parser = argparse.ArgumentParser(description="Build local Qdrant RAG indexes.")
    parser.add_argument(
        "--app",
        choices=[*SUPPORTED_APPS, "all"],
        default="all",
        help="App to index. Defaults to all apps.",
    )
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Upsert into existing collections without deleting them first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apps = None if args.app == "all" else [args.app]
    result = index_apps(apps, recreate=not args.no_recreate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
