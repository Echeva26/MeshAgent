from functools import lru_cache
from typing import Iterable

from common.config import get_config


@lru_cache(maxsize=1)
def _get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install dependencies from requirements.txt."
        ) from exc

    return SentenceTransformer(get_config().embedding_model)


def embed_text(text: str) -> list[float]:
    if not isinstance(text, str):
        raise TypeError("embed_text expects a single string.")
    vector = _get_model().encode(text, normalize_embeddings=True)
    return vector.astype(float).tolist()


def embed_documents(texts: Iterable[str]) -> list[list[float]]:
    texts_list = list(texts)
    if not all(isinstance(text, str) for text in texts_list):
        raise TypeError("embed_documents expects an iterable of strings.")
    if not texts_list:
        return []
    vectors = _get_model().encode(texts_list, normalize_embeddings=True)
    return [vector.astype(float).tolist() for vector in vectors]
