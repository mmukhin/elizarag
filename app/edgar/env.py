"""Load settings (OPENAI_API_KEY, LLM_PROVIDER, ...) from the project's .env file."""

from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env() -> None:
    """Read .env into os.environ; variables already set in the shell win. No-op if the file is missing."""
    load_dotenv(ENV_FILE, override=False)
