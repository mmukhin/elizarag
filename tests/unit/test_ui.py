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


def apply(at: AppTest) -> AppTest:
    """Submit the sidebar settings form (values set since the last run take effect)."""
    at.button(key="apply").click().run()
    assert not at.exception
    return at


def ask(at: AppTest, query: str) -> AppTest:
    """Submit pending sidebar settings and ask `query` in the same run."""
    at.button(key="apply").click()
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


def sections(at: AppTest) -> list[str]:
    """Collapsible section headers below the answer (not the passages or the JSON)."""
    return [e.label for e in at.expander if not e.label.startswith("[") and e.label not in ("Example questions", "Structured output (JSON)")]


def section(at: AppTest, prefix: str):
    return next(e for e in at.expander if e.label.startswith(prefix) or prefix in e.label)


def labels(at: AppTest) -> list[str]:
    return [e.label.split("] ", 1)[1] for e in passages(at)]


def test_initial_page_has_no_results():
    at = run()
    assert at.title[0].value == "SCiFi Assistant"
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
    assert len(passages(at)) == 5  # default k for an unknown model
    assert any(s.startswith("5 passages sent to the LLM") for s in sections(at))
    assert all("·" in label and "FY" in label for label in labels(at))
    caption = passages(at)[0].caption[0].value
    assert "score" in caption and ("vector #" in caption or "keyword #" in caption)
    text = passages(at)[0].markdown[0].value
    assert text.startswith('<div class="passage" id="passage-1">') and "tariffs" in text.lower()


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
    next(c for c in at.sidebar.checkbox if c.label == "Detect filters from the question").uncheck()
    at.sidebar.multiselect[0].unselect("JPM")
    apply(at)
    assert not any(c.value.startswith("Filters from the question") for c in at.caption)
    assert len({label.split(" ")[0] for label in labels(at)}) > 1


def test_sidebar_company_that_contradicts_the_question_is_flagged():
    at = run()
    at.sidebar.multiselect[0].select("JPM")
    ask(at, "How has NVIDIA's revenue changed?")
    assert "The question implies tickers NVDA, but the sidebar filter (JPM) is used" in at.warning[0].value
    assert labels(at) and all(label.startswith("JPM") for label in labels(at))


def test_company_named_but_not_indexed_is_reported_not_silently_dropped():
    at = ask(run(), "How has Boeing's revenue changed?")  # the fake index has AAPL, JPM, NVDA, TSLA
    assert any("BA is not in this collection" in w.value for w in at.warning)
    assert not passages(at)  # no other company's passages stand in for Boeing


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


def test_search_settings_apply_only_when_submitted():
    at = ask(run(), "tariffs")
    assert len({label.split(" ")[0] for label in labels(at)}) > 1
    at.sidebar.multiselect[0].select("TSLA")
    at.text_input[0].input("tariffs and costs").run()  # a new question alone keeps the saved search settings
    assert len({label.split(" ")[0] for label in labels(at)}) > 1
    at.sidebar.multiselect[0].select("TSLA")
    apply(at)  # "Apply and re-run" re-runs the current question with the new settings
    assert {label.split(" ")[0] for label in labels(at)} == {"TSLA"}


def script_pick_model():
    from app.ui import llm_settings
    from tests.fakes import fake_retriever

    import streamlit as st

    with st.sidebar:
        llm_settings(fake_retriever(), None, None)


def test_one_model_picker_sets_window_k_and_budget_for_the_model(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    at = run(script_pick_model)
    picker = at.selectbox(key="llm")
    assert picker.options == ["ollama: gemma4:e2b", "openai: gpt-6-luna", "openai: gpt-4o-mini"]
    assert (at.slider[0].value, at.slider[1].value) == (5, 8_000)
    assert any("16,384-token window" in c.value for c in at.caption)

    picker.set_value(("openai", "gpt-6-luna")).run()
    assert not at.exception
    assert (at.slider[0].value, at.slider[1].value, at.slider[1].max) == (8, 24_000, 64_000)
    assert any("gpt-6-luna" in c.value and "1,100,000-token window" in c.value and "4,000 reserved" in c.value for c in at.caption)


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

    at.sidebar.radio[0].set_value("keyword")
    apply(at)
    assert len(at.error) == 0
    assert len(passages(at)) > 0


def test_answer_is_generated_with_key_points_citations_and_json():
    at = ask(run(), "tariffs on imports from China")
    assert at.subheader[0].value == "Answer"
    link = '<a class="cite" href="#passage-1">[1]</a>'
    assert link in at.markdown[0].value
    assert any("AAPL" in m.value and "Tariffs raise component costs" in m.value and link in m.value for m in at.markdown)
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


def test_sections_below_the_answer_are_collapsible():
    at = ask(run(), "tariffs")
    labels_ = sections(at)
    assert labels_[0] == "Retrieval · 1 search" and labels_[1].startswith("5 passages sent to the LLM")
    assert labels_[2].endswith("top-k passages not sent (over the 8,000-token budget)")
    assert labels_[3].endswith("candidates not used (ranked below top-k)")
    expanded = {e.label: e.proto.expanded for e in at.expander}
    assert not expanded["Retrieval · 1 search"] and expanded[labels_[1]]  # passages open, trace collapsed
    assert all(p.label in [e.label for e in section(at, "5 passages sent").expander] for p in passages(at))  # nested


def test_cited_passages_are_open_and_linked_targets():
    at = ask(run(), "tariffs")  # the scripted answer cites [1]
    opened = [e.proto.expanded for e in passages(at)]
    assert opened == [True, False, False, False, False]
    assert all(f'id="passage-{i}"' in p.markdown[0].value for i, p in enumerate(passages(at), 1))
    assert any(":target" in m.value for m in at.markdown)  # highlight style for the jumped-to passage


def test_context_budget_and_k_default_from_the_model_profile():
    at = run()
    k, budget = at.sidebar.slider[0], at.sidebar.slider[1]
    assert (k.value, budget.value, budget.max) == (5, 8_000, 16_384 - 1_024 - 1_000)
    assert any("16,384-token window" in c.value and "1,024 reserved" in c.value for c in at.sidebar.caption)


def test_answer_sees_exactly_the_passages_shown():
    at = ask(run(), "tariffs")
    shown = len(passages(at))
    assert any(s.startswith(f"{shown} passages sent to the LLM") for s in sections(at))
    assert "Structured output (JSON)" in [e.label for e in at.expander]


def test_generation_can_be_turned_off():
    at = run()
    at.checkbox(key="generate").uncheck()
    ask(at, "tariffs")
    assert sections(at)[:2] == ["Retrieval · 1 search", "5 passages"]
    assert not any("not sent" in s for s in sections(at))
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
    assert subheaders(at)[0] == "Answer" and sections(at)[0] == "Retrieval · 1 search"
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
    unused = next(s for s in sections(at) if s.endswith("candidates not used (ranked below top-k)"))
    n = int(unused.split()[0])
    assert n == 10  # all 12 chunks are vector candidates, 2 made top-k
    table = at.dataframe[-1].value
    assert len(table) == n and {"search", "vector #", "cosine", "bm25 #", "rrf", "tokens", "text"} <= set(table.columns)
    assert table["vector #"].min() > 0 and not set(table["passage"]) & set(labels(at))


def test_tight_budget_splits_sent_and_not_sent():
    at = ask(run(script_tight_budget), "tariffs")
    sent = len(passages(at))
    assert sent >= 1 and any(s.startswith(f"{sent} passages sent to the LLM") for s in sections(at))
    not_sent = next(s for s in sections(at) if "not sent (over the 100-token budget)" in s)
    assert int(not_sent.split()[0]) == 5 - sent
    assert len(at.dataframe[0].value) == 5 - sent
    payload = next(e for e in at.expander if e.label == "Structured output (JSON)").json[0].value
    assert payload.count('"chunk_id"') == sent  # the LLM saw exactly the sent passages
