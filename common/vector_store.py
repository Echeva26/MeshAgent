from typing import Any
from functools import lru_cache

from common.config import collection_name, get_config
from common.embeddings_provider import embed_text


@lru_cache(maxsize=1)
def get_qdrant_client():
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for local RAG. "
            "Install dependencies from requirements.txt."
        ) from exc

    config = get_config()
    if config.qdrant_url in {":memory:", "memory", "in-memory"}:
        return QdrantClient(location=":memory:")
    return QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)


def ensure_collection(client: Any, name: str, vector_size: int, recreate: bool = False) -> None:
    from qdrant_client.models import Distance, VectorParams

    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _search(client: Any, collection: str, vector: list[float], limit: int):
    try:
        return client.search(collection_name=collection, query_vector=vector, limit=limit)
    except AttributeError:
        response = client.query_points(collection_name=collection, query=vector, limit=limit)
        return response.points


def search_collection(collection: str, query: str, limit: int) -> list[dict[str, Any]]:
    client = get_qdrant_client()
    vector = embed_text(query)
    points = _search(client, collection, vector, limit)
    results = []
    for point in points:
        payload = dict(point.payload or {})
        payload["@search.score"] = float(point.score)
        results.append(payload)
    return results


def search_constraints(app_name: str, query: str, limit: int) -> list[dict[str, Any]]:
    return search_collection(collection_name(app_name, "constraints"), query, limit)


def search_tools(app_name: str, query: str, limit: int) -> list[dict[str, Any]]:
    return search_collection(collection_name(app_name, "tools"), query, limit)
