import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5-nano")


def get_chat_model(
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL
):
    """Return a LangChain chat model based on provider."""

    provider = provider.lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4000,
        )

    elif provider == "anthropic" or provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4000,
        )

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=4000,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def validate_api_key(provider: str = DEFAULT_PROVIDER) -> bool:
    try:
        get_chat_model(provider=provider)
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False