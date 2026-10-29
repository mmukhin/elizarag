import pytest

from app.edgar.hybrid import Hit
from app.edgar.prompts import PROMPT_DIR, load_prompt
from app.edgar.qa import (
    ModelAnswer,
    ModelClaim,
    QAResult,
    answer_question,
    build_context,
    resolve_claims,
    select_hits,
    validate,
)
from tests.fakes import ScriptedChat, answer_json, chunk, hits_for, sample_chunks


def sized(text, ticker, tokens, index=0):
    doc = chunk(text, ticker, index=index)
    doc.metadata["token_count"] = tokens
    return Hit(doc, 1.0)


# --- prompts ---------------------------------------------------------------


def test_prompt_files_live_in_prompts_dir():
    assert PROMPT_DIR.name == "prompts"
    assert {p.name for p in PROMPT_DIR.glob("*.md")} >= {"qa_system.md", "qa_user.md"}


def test_system_prompt_enforces_grounding_and_schema_fields():
    system = load_prompt("qa_system")
    for needle in ("ONLY", "passage", "status", "off_topic", "not_in_passages", "claims", "passages", "caveats"):
        assert needle in system


def test_user_prompt_substitution_and_dollar_signs_in_values():
    user = load_prompt("qa_user", question="What is $5 billion?", context="[1] revenue was $96.9B")
    assert "What is $5 billion?" in user and "[1] revenue was $96.9B" in user
    assert "$question" not in user and "$context" not in user


def test_prompt_missing_value_raises():
    with pytest.raises(KeyError):
        load_prompt("qa_user", question="only one")


# --- context building -------------------------------------------------------


def test_build_context_numbers_passages_and_returns_sources():
    hits = hits_for(*sample_chunks()[:2])
    context, sources = build_context(hits)
    assert context.startswith("[1] AAPL · 10-K · FY2025 · filed 2025-12-01\n" + hits[0].doc.page_content)
    assert "\n\n[2] AAPL" in context
    assert [s.ref for s in sources] == [1, 2]
    assert sources[0].chunk_id == "AAPL_10-K_2025.txt#0" and sources[1].chunk_id == "AAPL_10-K_2025.txt#1"
    assert "text" not in sources[0].model_dump()  # content stays local, looked up by chunk_id


def test_select_hits_keeps_everything_when_it_fits():
    hits = [sized("a", "AAPL", 100), sized("b", "TSLA", 100)]
    assert select_hits(hits, 10_000) == hits


def test_select_hits_takes_each_companys_best_passage_before_second_ones():
    # budget fits 3 passages (3 * (800 + 30) = 2490); order is AAPL, AAPL, AAPL, TSLA, JPM
    hits = [
        sized("a1", "AAPL", 800, 0), sized("a2", "AAPL", 800, 1), sized("a3", "AAPL", 800, 2),
        sized("t1", "TSLA", 800, 0), sized("j1", "JPM", 800, 0),
    ]  # fmt: skip
    kept = select_hits(hits, 2500)
    assert [h.doc.page_content for h in kept] == ["a1", "t1", "j1"]  # original order, one per company


def test_select_hits_always_keeps_one_even_over_budget():
    assert len(select_hits([sized("big", "AAPL", 5000)], 100)) == 1


def test_select_hits_counts_tokens_when_metadata_missing():
    hits = hits_for(chunk("word " * 400, "AAPL"), chunk("word " * 400, "TSLA"))
    assert len(select_hits(hits, 500)) == 1


# --- validation -------------------------------------------------------------


def test_validate_clean_answer_has_no_warnings():
    answer = ModelAnswer.model_validate_json(answer_json())
    assert validate(answer, n_sources=2) == []


def test_validate_removes_invalid_passage_ids():
    answer = ModelAnswer(status="answered", answer="x [1]", claims=[ModelClaim(text="p", passages=[1, 7])])
    warnings = validate(answer, n_sources=2)
    assert answer.claims[0].passages == [1]
    assert any("invalid passage ids [7]" in w for w in warnings)


def test_validate_flags_uncited_claim_unknown_text_citation_and_no_citations():
    answer = ModelAnswer(status="answered", answer="Claim [9].", claims=[ModelClaim(text="floating claim")])
    warnings = validate(answer, n_sources=2)
    assert any("uncited claim" in w for w in warnings)
    assert any("unknown passages [9]" in w for w in warnings)
    assert any("no citations" in w for w in warnings)


def test_validate_partial_answer_must_still_cite():
    answer = ModelAnswer(status="partial", answer="Only Apple is covered.", caveats=["no Tesla passages"])
    assert "answer has no citations" in validate(answer, n_sources=2)


def test_validate_not_in_passages_needs_no_citations():
    answer = ModelAnswer(status="not_in_passages", answer="The passages do not say.", caveats=["no data"])
    assert validate(answer, n_sources=2) == []


def test_validate_not_in_passages_with_claims_is_flagged():
    answer = ModelAnswer(status="not_in_passages", answer="No data.", claims=[ModelClaim(text="but here's one")])
    assert "not_in_passages answer still makes claims" in validate(answer, n_sources=2)


def test_validate_off_topic_needs_no_citations():
    answer = ModelAnswer(status="off_topic", answer="I only answer questions about SEC filings.")
    assert validate(answer, n_sources=2) == []


def test_validate_off_topic_with_claims_or_citations_is_flagged():
    answer = ModelAnswer(status="off_topic", answer="Mix flour and sugar [1].", claims=[ModelClaim(text="x", passages=[1])])
    assert "off-topic answer should not make claims or cite passages" in validate(answer, n_sources=2)


# --- answer_question --------------------------------------------------------


def test_answer_question_returns_structured_result_and_sends_grounded_prompt():
    hits = hits_for(*sample_chunks()[:3])
    llm = ScriptedChat(answer_json())
    result = answer_question("What tariff risks does Apple face?", hits, llm)

    assert isinstance(result, QAResult) and result.model == "scripted" and result.status == "answered"
    assert result.claims[0].chunk_ids == ["AAPL_10-K_2025.txt#0"] and result.claims[0].tickers == ["AAPL"]
    assert result.warnings == []
    assert [s.ref for s in result.sources] == [1, 2, 3]

    call = llm.calls[0]
    assert call["schema"] is ModelAnswer
    assert "What tariff risks does Apple face?" in call["user"]
    assert all(h.doc.page_content in call["user"] for h in hits)
    assert "ONLY" in call["system"]


def test_answer_question_result_round_trips_through_json():
    result = answer_question("q", hits_for(*sample_chunks()[:2]), ScriptedChat(answer_json()))
    assert QAResult.model_validate_json(result.model_dump_json()) == result


def test_answer_question_without_hits_does_not_call_model():
    llm = ScriptedChat(answer_json())
    result = answer_question("anything", [], llm)
    assert llm.calls == [] and result.status == "not_in_passages" and result.sources == []
    assert result.warnings == ["no passages to answer from"]


def test_answer_question_reports_dropped_passages_and_renumbers():
    hits = [sized("a1", "AAPL", 800, 0), sized("a2", "AAPL", 800, 1), sized("t1", "TSLA", 800, 0)]
    llm = ScriptedChat(answer_json())
    result = answer_question("q", hits, llm, max_context_tokens=1700)
    assert [s.chunk_id for s in result.sources] == ["AAPL_10-K_2025.txt#0", "TSLA_10-K_2025.txt#0"]  # a2 dropped
    assert [s.ref for s in result.sources] == [1, 2]
    assert any("1 lower-ranked passages dropped" in w for w in result.warnings)
    assert "a2" not in llm.calls[0]["user"]


def test_answer_question_surfaces_hallucinated_citation():
    llm = ScriptedChat(
        answer_json(answer="Tariffs [5].", claims=[{"text": "p", "passages": [5]}])
    )
    result = answer_question("q", hits_for(*sample_chunks()[:2]), llm)
    assert result.claims[0].chunk_ids == [] and result.claims[0].tickers == []
    assert any("invalid passage ids [5]" in w for w in result.warnings)


def test_resolve_claims_maps_passages_to_chunk_ids_and_tickers():
    _, sources = build_context(hits_for(*sample_chunks()[:6]))  # 1-3 AAPL, 4-6 TSLA
    answer = ModelAnswer(
        status="answered",
        answer="x",
        claims=[
            ModelClaim(text="single company", passages=[4, 5]),
            ModelClaim(text="spans two companies", passages=[1, 4, 4]),
            ModelClaim(text="no citation"),
        ],
    )
    claims = resolve_claims(answer, sources)
    assert [c.tickers for c in claims] == [["TSLA"], ["AAPL", "TSLA"], []]
    assert claims[0].chunk_ids == ["TSLA_10-K_2025.txt#0", "TSLA_10-K_2025.txt#1"]
    assert claims[1].chunk_ids == ["AAPL_10-K_2025.txt#0", "TSLA_10-K_2025.txt#0"]  # duplicates collapsed


def test_answer_question_off_topic_passes_through_without_claims():
    llm = ScriptedChat(answer_json(status="off_topic", answer="I only answer questions about SEC filings.", claims=[]))
    result = answer_question("chocolate chip cookie recipe", hits_for(*sample_chunks()[:2]), llm)
    assert result.status == "off_topic" and result.claims == [] and result.warnings == []


def test_answer_question_records_filters():
    result = answer_question("q", hits_for(*sample_chunks()[:2]), ScriptedChat(answer_json()), filters={"tickers": ["AAPL"]})
    assert result.filters == {"tickers": ["AAPL"]}
