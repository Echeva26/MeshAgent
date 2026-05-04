from functools import lru_cache

from common.config import get_config


@lru_cache(maxsize=1)
def get_llm():
    config = get_config()
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for local vLLM integration. "
            "Install dependencies from requirements.txt."
        ) from exc

    return ChatOpenAI(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens,
    )
