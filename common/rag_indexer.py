import json
import uuid
from pathlib import Path
from typing import Any

from common.config import REPO_ROOT, collection_name
from common.embeddings_provider import embed_documents
from common.vector_store import ensure_collection, get_qdrant_client


SUPPORTED_APPS = ("app-malt", "app-CRG", "app-traffic-analysis")


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def _point_id(app_name: str, kind: str, item: dict[str, Any], index: int) -> str:
    raw_id = str(item.get("id", index))
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"meshagent:{app_name}:{kind}:{raw_id}"))


def _index_items(
    app_name: str,
    kind: str,
    items: list[dict[str, Any]],
    text_field: str,
    recreate: bool,
) -> int:
    from qdrant_client.models import PointStruct

    if not items:
        return 0

    texts = [str(item.get(text_field, "")) for item in items]
    vectors = embed_documents(texts)
    collection = collection_name(app_name, kind)
    client = get_qdrant_client()
    ensure_collection(client, collection, len(vectors[0]), recreate=recreate)

    points = []
    for index, (item, vector) in enumerate(zip(items, vectors)):
        payload = dict(item)
        payload["text"] = texts[index]
        payload["app"] = app_name
        payload["kind"] = kind
        points.append(
            PointStruct(
                id=_point_id(app_name, kind, item, index),
                vector=vector,
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def index_app(app_name: str, recreate: bool = True) -> dict[str, int]:
    app_dir = REPO_ROOT / app_name
    data_dir = app_dir / "data"
    constraints = _load_json(data_dir / "rag_constraints.json")
    tools = _load_json(data_dir / "rag_tools.json")

    return {
        "constraints": _index_items(app_name, "constraints", constraints, "constraint", recreate),
        "tools": _index_items(app_name, "tools", tools, "description", recreate),
    }


def index_apps(app_names: list[str] | None = None, recreate: bool = True) -> dict[str, dict[str, int]]:
    targets = app_names or list(SUPPORTED_APPS)
    unknown = sorted(set(targets) - set(SUPPORTED_APPS))
    if unknown:
        raise ValueError(f"Unsupported apps: {', '.join(unknown)}")
    return {app_name: index_app(app_name, recreate=recreate) for app_name in targets}
