import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _slug(value: str) -> str:
    return value.lower().replace("-", "_").replace("/", "_")


@dataclass(frozen=True)
class LocalConfig:
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_temperature: float
    llm_max_tokens: int
    embedding_model: str
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_constraint_collection: str
    qdrant_tool_collection: str


def get_config() -> LocalConfig:
    _load_env()
    return LocalConfig(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct"),
        llm_api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.0),
        llm_max_tokens=_int_env("LLM_MAX_TOKENS", 4000),
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        qdrant_constraint_collection=os.getenv("QDRANT_CONSTRAINT_COLLECTION", "meshagent_constraints"),
        qdrant_tool_collection=os.getenv("QDRANT_TOOL_COLLECTION", "meshagent_tools"),
    )


def collection_name(app_name: str, kind: str) -> str:
    config = get_config()
    normalized_kind = kind.lower()
    app_env = _slug(app_name).upper()
    kind_env = "CONSTRAINT" if normalized_kind.startswith("constraint") else "TOOL"
    override = os.getenv(f"QDRANT_{app_env}_{kind_env}_COLLECTION")
    if override:
        return override

    base = (
        config.qdrant_constraint_collection
        if kind_env == "CONSTRAINT"
        else config.qdrant_tool_collection
    )
    return f"{base}_{_slug(app_name)}"
