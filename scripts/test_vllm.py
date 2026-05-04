from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from common.config import get_config


def main() -> None:
    config = get_config()
    client = OpenAI(base_url=config.llm_base_url, api_key=config.llm_api_key)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": "You are a concise coding assistant."},
            {"role": "user", "content": "Reply with exactly: vLLM OK"},
        ],
        temperature=0,
        max_tokens=16,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
