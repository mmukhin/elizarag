from streamlit.testing.v1 import AppTest

from app.ui import EXAMPLE_QUESTIONS


def script():
    from app.ui import render
    from tests.fakes import ScriptedChat, answer_json, fake_retriever

    render(fake_retriever(), ScriptedChat(answer_json()))


def script_llm_down():
    import httpx

    from app.edgar.llm import ChatModel
    from app.ui import render
    from tests.fakes import fake_retriever

    class Down(ChatModel):
        def _complete(self, system, user, schema):
            raise httpx.ConnectError("ollama is not running")

    render(fake_retriever(), Down())


def script_tight_budget():
    from app.ui import render
    from tests.fakes import ScriptedChat, answer_json, fake_retriever

    render(fake_retriever(), ScriptedChat(answer_json()), context_budget=100)


def script_backend_down():
    from app.ui import render
    from tests.fakes import BackendDownEmbeddings, ScriptedChat, answer_json, fake_retriever

    render(fake_retriever(BackendDownEmbeddings()), ScriptedChat(answer_json()))


def run(fn=script) -> AppTest:
    at = AppTest.from_function(fn, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def ask(at: AppTest, query: str) -> AppTest:
    at.text_input[0].input(query).run()
    assert not at.exception
    return at


def passages(at: AppTest) -> list:
    """Passage expanders only (labelled '[n] TICKER ...'), not the JSON expander."""
    return [e for e in at.expander if e.label.startswith("[")]


def requests(at: AppTest) -> list[str]:
    """Retrieval request JSON blocks (Chroma, BM25, per search), not the answer's structured output."""
    return [j.value for j in at.json if '"question"' not in j.value]


def subheaders(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader]


def labels(at: AppTest) -> list[str]:
    return [e.label.split("] ", 1)[1] for e in passages(at)]


def test_initial_page_has_no_results():
    at = run()
    assert at.title[0].value == "SEC filings Q&A"
    assert "12 chunks" in at.caption[0].value and "4 companies" in at.caption[0].value
    assert len(passages(at)) == 0 and len(at.warning) == 0 and len(at.error) == 0


def test_example_question_button_fills_the_question_and_searches():
    at = run()
    examples = next(e for e in at.expander if e.label == "Example questions")
    assert [b.label for b in examples.button] == EXAMPLE_QUESTIONS
    examples.button[0].click().run()
    assert at.text_input[0].value == EXAMPLE_QUESTIONS[0]
    assert any(c.value == "Filters from the question: tickers AAPL, TSLA, JPM" for c in at.caption)
    assert {label.split(" ")[0] for label in labels(at)} == {"AAPL", "TSLA", "JPM"}


def test_query_shows_ranked_passages_with_rank_info():
    at = ask(run(), "supply chain tariffs")
    assert len(passages(at)) == 5  # default k
    assert "5 passages sent to the LLM" in subheaders(at)
    assert all("·" in label and "FY" in label for label in labels(at))
    caption = passages(at)[0].caption[0].value
    assert "score" in caption and ("vector #" in caption or "keyword #" in caption)
    assert "tariffs" in passages(at)[0].text[0].value.lower()


def test_company_filter_limits_results():
    at = run()
    at.sidebar.multiselect[0].select("TSLA")
    ask(at, "tariffs")
    assert labels(at) and all(label.startswith("TSLA") for label in labels(at))


def test_filters_detected_from_question_limit_and_balance_results():
    at = run()
    at.sidebar.slider[0].set_value(2)
    ask(at, "What tariff risks do Tesla and Apple face?")
    assert any(c.value == "Filters from the question: tickers TSLA, AAPL" for c in at.caption)
    assert [label.split(" ")[0] for label in labels(at)] == ["TSLA", "TSLA", "AAPL", "AAPL"]  # question order
    payload = next(e for e in at.expander if e.label == "Structured output (JSON)").json[0].value
    assert '"filters": {"tickers": ["TSLA", "AAPL"]}' in payload


def test_sidebar_choice_overrides_detected_filter_and_detection_can_be_off():
    at = run()
    at.sidebar.multiselect[0].select("JPM")
    ask(at, "Apple tariffs and interest income")
    assert labels(at) and all(label.startswith("JPM") for label in labels(at))
    at.sidebar.checkbox[1].uncheck()  # "Detect filters from the question"
    at.sidebar.multiselect[0].unselect("JPM").run()
    assert not any(c.value.startswith("Filters from the question") for c in at.caption)
    assert len({label.split(" ")[0] for label in labels(at)}) > 1


def test_quarter_filter_from_sidebar():
    at = run()
    at.sidebar.multiselect[2].select(2)  # quarters appear because the fake chunks carry them
    ask(at, "net interest income")
    assert len(labels(at)) == 1 and labels(at)[0].startswith("JPM · 10-Q")


def test_balanced_comparison_returns_each_selected_company():
    at = run()
    at.sidebar.multiselect[0].select("AAPL").select("TSLA").select("JPM")
    at.sidebar.slider[0].set_value(2)
    ask(at, "what are the main risks")
    assert [label.split(" ")[0] for label in labels(at)] == ["AAPL", "AAPL", "TSLA", "TSLA", "JPM", "JPM"]


def test_balance_checkbox_disabled_until_two_companies_selected():
    at = run()
    assert at.sidebar.checkbox[0].disabled
    at.sidebar.multiselect[0].select("AAPL").select("TSLA").run()
    assert not at.sidebar.checkbox[0].disabled


def test_year_and_filing_type_filters():
    at = run()
    at.sidebar.selectbox[0].select("10-Q")
    ask(at, "net interest income")
    assert labels(at) == ["JPM · 10-Q FY2025 · chunk 1/10"]

    at.sidebar.selectbox[0].select("Any")
    at.sidebar.multiselect[1].select(2024)
    ask(at, "revenue")
    assert labels(at) and all("FY2024" in label for label in labels(at))


def test_keyword_mode_exact_figure():
    at = run()
    at.sidebar.radio[0].set_value("keyword")
    ask(at, "23.5 billion")
    assert len(passages(at)) == 1 and labels(at)[0].startswith("JPM")
    assert "keyword #1" in passages(at)[0].caption[0].value


def test_no_results_warning():
    at = run()
    at.sidebar.radio[0].set_value("keyword")
    ask(at, "qqqqzzzz")
    assert len(passages(at)) == 0
    assert "No matching passages" in at.warning[0].value


def test_backend_down_shows_error_and_keyword_fallback_works():
    at = run(script_backend_down)
    ask(at, "tariffs")
    assert "Embedding backend unreachable" in at.error[0].value
    assert len(passages(at)) == 0

    at.sidebar.radio[0].set_value("keyword").run()
    assert not at.exception and len(at.error) == 0
    assert len(passages(at)) > 0


def test_answer_is_generated_with_key_points_citations_and_json():
    at = ask(run(), "tariffs on imports from China")
    assert at.subheader[0].value == "Answer"
    assert "[1]" in at.markdown[0].value
    assert any("AAPL" in m.value and "Tariffs raise component costs" in m.value and "[1]" in m.value for m in at.markdown)
    json_expander = next(e for e in at.expander if e.label == "Structured output (JSON)")
    payload = json_expander.json[0].value
    assert '"question"' in payload and '"sources"' in payload and '"claims"' in payload
    assert '"chunk_id"' in payload and "component costs and hurt gross margin" not in payload  # no passage text


def test_answer_shows_tokens_cost_and_latency():
    at = ask(run(), "tariffs on imports from China")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Tokens sent"] == "1,200" and metrics["Tokens received"] == "80"
    assert metrics["Cost"] == "unknown" and metrics["LLM latency"].endswith(" s")
    assert any("LLM_PRICE_INPUT" in c.value for c in at.caption)  # how to price an unknown model


def test_retrieval_and_passage_sections_can_be_hidden():
    at = ask(run(), "tariffs")
    assert "Retrieval" in subheaders(at) and passages(at)
    at.toggle(key="show-retrieval").set_value(False).run()
    assert "Retrieval" not in subheaders(at) and passages(at)
    at.toggle(key="show-passages").set_value(False).run()
    assert not passages(at) and not any("candidates not used" in h for h in subheaders(at))
    assert subheaders(at)[0] == "Answer"  # the answer stays


def test_answer_sees_exactly_the_passages_shown():
    at = ask(run(), "tariffs")
    shown = len(passages(at))
    assert f"{shown} passages sent to the LLM" in subheaders(at)
    assert "Structured output (JSON)" in [e.label for e in at.expander]


def test_generation_can_be_turned_off():
    at = run()
    at.sidebar.checkbox[2].uncheck()
    ask(at, "tariffs")
    assert subheaders(at)[:2] == ["Retrieval", "5 passages"]
    assert not any("not sent" in s for s in subheaders(at))
    assert not any(e.label == "Structured output (JSON)" for e in at.expander)


def test_llm_failure_shows_error_but_keeps_passages():
    at = ask(run(script_llm_down), "tariffs")
    assert "Answer generation failed" in at.error[0].value and "ConnectError" in at.error[0].value
    assert len(passages(at)) == 5


def test_retrieval_shows_chroma_and_bm25_requests_and_top_k():
    at = run()
    at.sidebar.slider[0].set_value(3)
    at.sidebar.selectbox[0].select("10-K")
    ask(at, "tariffs on imports")
    assert subheaders(at)[:2] == ["Answer", "Retrieval"]
    assert any("top-k 3" in m.value and "Search 1/1" in m.value for m in at.markdown)
    chroma, bm25 = requests(at)
    assert '"n_results": 50' in chroma and '"where": {' in chroma and '"filing_type": {"$in": ["10-K"]}' in chroma
    assert "embed(" in chroma and '"collection"' in chroma
    assert '"tariffs"' in bm25 and '"imports"' in bm25 and '"corpus_chunks": 12' in bm25


def test_balanced_search_shows_one_query_per_company():
    at = run()
    at.sidebar.multiselect[0].select("AAPL").select("TSLA")
    ask(at, "risks")
    searches = [m.value for m in at.markdown if m.value.startswith("**Search ")]
    assert len(searches) == 2 and all("/2**" in s for s in searches)
    assert '"AAPL"' in requests(at)[0] and '"TSLA"' in requests(at)[2]


def test_keyword_mode_shows_chroma_not_run():
    at = run()
    at.sidebar.radio[0].set_value("keyword")
    ask(at, "tariffs")
    assert any(c.value == "not run in keyword mode" for c in at.caption)


def test_candidates_below_top_k_are_listed_as_not_used():
    at = run()
    at.sidebar.slider[0].set_value(2)
    ask(at, "revenue tariffs")
    unused = next(s for s in subheaders(at) if s.endswith("candidates not used (ranked below top-k)"))
    n = int(unused.split()[0])
    assert n == 10  # all 12 chunks are vector candidates, 2 made top-k
    table = at.dataframe[-1].value
    assert len(table) == n and {"search", "vector #", "cosine", "bm25 #", "rrf", "tokens", "text"} <= set(table.columns)
    assert table["vector #"].min() > 0 and not set(table["passage"]) & set(labels(at))


def test_tight_budget_splits_sent_and_not_sent():
    at = ask(run(script_tight_budget), "tariffs")
    sent = len(passages(at))
    assert sent >= 1 and f"{sent} passages sent to the LLM" in subheaders(at)
    not_sent = next(s for s in subheaders(at) if "not sent (over the 100-token budget)" in s)
    assert int(not_sent.split()[0]) == 5 - sent
    assert len(at.dataframe[0].value) == 5 - sent
    payload = next(e for e in at.expander if e.label == "Structured output (JSON)").json[0].value
    assert payload.count('"chunk_id"') == sent  # the LLM saw exactly the sent passages
