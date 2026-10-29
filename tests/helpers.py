"""Paths and live-backend probes shared by unit, integration and eval tests."""

from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "edgar_corpus"
FIXTURES = ROOT / "tests" / "fixtures"

# the SEC corpus is not in the repo; put the *_full.txt filings in data/edgar_corpus/ to run these tests
needs_corpus = pytest.mark.skipif(
    not any(CORPUS.glob("*_full.txt")), reason=f"SEC filings corpus not found in {CORPUS}"
)


def ollama_has(model: str) -> bool:
    """True if a local Ollama is up and has `model` (a bare name like `embeddinggemma` matches any tag)."""
    try:
        models = httpx.get("http://localhost:11434/api/tags", timeout=2).json()["models"]
    except Exception:
        return False
    return any(m["name"] == model or m["name"].split(":")[0] == model for m in models)
