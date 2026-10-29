"""Eval fixtures (live LLM, live index) and the per-run report in tests/evals/results/."""

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pytest

from app.edgar.env import load_env
from app.edgar.hybrid import HybridRetriever
from app.edgar.llm import OllamaChat, get_chat_model
from app.edgar.prompts import prompt_version
from app.edgar.vectorstore import get_client, get_embeddings, load_store
from tests.evals.harness import RESULTS
from tests.helpers import ollama_has

load_env()  # evals may use OpenAI (LLM_PROVIDER=openai) with the key from .env
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RUN = {"model": None, "prompt_version": prompt_version(), "collection": None}


@pytest.fixture(scope="session")
def llm():
    """The model under test: $LLM_PROVIDER / $LLM_MODEL, default local gemma."""
    llm = get_chat_model()
    if isinstance(llm, OllamaChat) and not ollama_has(llm.model):
        pytest.skip(f"Ollama with {llm.model} not available")
    RUN["model"] = llm.model
    return llm


@pytest.fixture(scope="session")
def retriever():
    """Hybrid retriever over $EVAL_COLLECTION (default `10k`) on the local Chroma server."""
    name = os.environ.get("EVAL_COLLECTION", "10k")
    try:
        retriever = HybridRetriever(load_store(name, get_embeddings(), get_client()))
    except Exception as exc:  # server down, collection missing
        pytest.skip(f"Chroma collection {name!r} not available ({exc})")
    RUN["collection"] = name
    return retriever


def _summary() -> dict:
    """Pass rate per suite (all checks passed) and per suite.check."""
    rates = defaultdict(lambda: [0, 0])
    for row in RESULTS:
        outcomes = {row["suite"]: row["passed"]} | {f"{row['suite']}.{n}": c["passed"] for n, c in row["checks"].items()}
        for key, passed in outcomes.items():
            rates[key][0] += passed
            rates[key][1] += 1
    return {key: f"{p}/{n} ({p / n:.0%})" for key, (p, n) in sorted(rates.items())}


def pytest_sessionfinish(session, exitstatus):
    if not RESULTS:
        return
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{stamp}-{RUN['prompt_version']}.json"
    path.write_text(json.dumps(RUN | {"summary": _summary(), "cases": RESULTS}, indent=2))
    RUN["report"] = str(path)


def pytest_terminal_summary(terminalreporter):
    if not RESULTS:
        return
    terminalreporter.section("eval pass rates")
    for key, rate in _summary().items():
        terminalreporter.write_line(f"{key:<45} {rate}")
    if RUN.get("report"):
        terminalreporter.write_line(f"report: {RUN['report']}")
