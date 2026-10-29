import pytest

from app.edgar.llm import OllamaChat
from app.edgar.qa import answer_question
from tests.fakes import hits_for, sample_chunks
from tests.helpers import ollama_has


@pytest.mark.slow
@pytest.mark.skipif(not ollama_has("gemma4:e2b"), reason="Ollama with gemma4:e2b not available")
def test_real_gemma_returns_valid_grounded_structured_answer():
    hits = hits_for(*[c for c in sample_chunks() if c.metadata["ticker"] in {"AAPL", "TSLA"}][:6])
    result = answer_question("What supply chain or tariff risks do Apple and Tesla face?", hits, OllamaChat())
    assert result.status in ("answered", "partial") and result.answer.strip()
    assert result.claims, "expected at least one claim"
    valid = {s.chunk_id for s in result.sources}
    cited = {c for claim in result.claims for c in claim.chunk_ids}
    assert cited and cited <= valid
    assert not any("invalid passage ids" in w for w in result.warnings)
