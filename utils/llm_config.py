import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

def get_chat_model(model: str = DEFAULT_MODEL) -> ChatOpenAI:
    """Get LangChain ChatOpenAI model instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=4000
    )


def validate_api_key() -> bool:
    """Validate that the API key is set and working."""
    try:
        get_chat_model()
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False
