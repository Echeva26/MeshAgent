from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from common.embeddings_provider import embed_documents, embed_text


def main() -> None:
    single = embed_text("How many packet switches are in this graph?")
    batch = embed_documents(["constraint text", "tool description"])
    print(f"single_dimension={len(single)}")
    print(f"batch_count={len(batch)}")
    print(f"first_values={[round(value, 6) for value in single[:5]]}")


if __name__ == "__main__":
    main()
