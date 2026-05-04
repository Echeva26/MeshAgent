import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.rag_indexer import SUPPORTED_APPS
from common.vector_store import search_constraints, search_tools


def parse_args():
    parser = argparse.ArgumentParser(description="Test local Qdrant retrieval.")
    parser.add_argument("--app", choices=SUPPORTED_APPS, default="app-malt")
    parser.add_argument(
        "--query",
        default="Find packet switches and ports in the graph.",
        help="Query to retrieve constraints and tools for.",
    )
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def _preview(result: dict) -> str:
    text = result.get("constraint") or result.get("description") or result.get("tool") or ""
    return str(text).replace("\n", " ")[:180]


def main() -> None:
    args = parse_args()
    constraints = search_constraints(args.app, args.query, args.limit)
    tools = search_tools(args.app, args.query, args.limit)

    print("constraints:")
    for result in constraints:
        print(f"- score={result.get('@search.score'):.4f} {_preview(result)}")

    print("tools:")
    for result in tools:
        print(f"- score={result.get('@search.score'):.4f} {_preview(result)}")


if __name__ == "__main__":
    main()
