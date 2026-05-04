from typing import Any
from functools import lru_cache

from common.config import collection_name, get_config
from common.embeddings_provider import embed_text


@lru_cache(maxsize=16)
def _cached_qdrant_client(mode: str, local_path: str, url: str, api_key: str):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for local RAG. "
            "Install dependencies from requirements.txt."
        ) from exc

    if mode == "memory":
        return QdrantClient(location=":memory:")
    if mode == "local":
        return QdrantClient(path=local_path)
    return QdrantClient(url=url, api_key=api_key or None)


def get_qdrant_client():
    config = get_config()
    return _cached_qdrant_client(
        config.qdrant_mode,
        str(config.qdrant_local_path),
        config.qdrant_url,
        config.qdrant_api_key or "",
    )


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
