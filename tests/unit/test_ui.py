import html

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


def script_pick_model():
    import streamlit as st

    from app.ui import llm_settings
    from tests.fakes import fake_retriever

    with st.sidebar:
        llm_settings(fake_retriever(), None, None)


def run(fn=script) -> AppTest:
    at = AppTest.from_function(fn, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def ask(at: AppTest, question: str) -> AppTest:
    """Submit pending sidebar settings and send `question` in the same run."""
    at.button(key="apply").click()
    at.chat_input[0].set_value(question).run()
    assert not at.exception
    return at


def apply(at: AppTest) -> AppTest:
    """Press "Apply and re-run": re-asks the last question with the settings set since the last run."""
    at.button(key="apply").click().run()
    assert not at.exception
    return at


def answer(at: AppTest):
    """The latest assistant message."""
    return [m for m in at.chat_message if m.name == "assistant"][-1]


def user_message(at: AppTest):
    return [m for m in at.chat_message if m.name == "user"][-1]


def passages(at: AppTest) -> list:
    """Passage expanders of the latest answer (labelled '[n] TICKER ...')."""
    return [e for e in answer(at).expander if e.label.startswith("[")]


def labels(at: AppTest) -> list[str]:
    return [e.label.split("] ", 1)[1] for e in passages(at)]


def sections(at: AppTest) -> list[str]:
    """Collapsible sections of the latest answer (not single passages or the JSON)."""
    return [e.label for e in answer(at).expander if not e.label.startswith("[") and e.label != "Structured output (JSON)"]


def answer_text(at: AppTest) -> str:
    return next(m.value for m in answer(at).markdown if 'class="answer"' in m.value)


def payload(at: AppTest) -> str:
    return next(e for e in answer(at).expander if e.label == "Structured output (JSON)").json[0].value


def requests(at: AppTest) -> list[str]:
    """Retrieval request JSON blocks (Chroma, BM25, per search) of the latest answer."""
    return [j.value for j in answer(at).json if '"question"' not in j.value]


def companies(at: AppTest) -> set[str]:
    return {label.split(" ")[0] for label in labels(at)}


# --- page and conversation ------------------------------------------------------------


def test_welcome_screen_with_example_prompts_and_pinned_chat_input():
    at = run()
    assert at.title[0].value == "SCiFi Assistant"
    assert "12 chunks" in at.caption[0].value and "4 companies" in at.caption[0].value
    assert any("What do you want to know" in m.value for m in at.markdown)
    assert [b.label for b in at.button if b.key and b.key.startswith("example-")] == EXAMPLE_QUESTIONS
    assert len(at.chat_input) == 1 and not at.chat_message


def test_example_prompt_asks_it():
    at = run()
    at.button(key="example-0").click().run()
    assert html.escape(EXAMPLE_QUESTIONS[0]) in user_message(at).markdown[0].value
    detected = user_message(at).caption[0].value
    assert "AAPL" in detected and "TSLA" in detected and "JPM" in detected
    assert companies(at) == {"AAPL", "TSLA", "JPM"}


def test_conversation_keeps_earlier_answers_with_sources_collapsed():
    at = ask(run(), "tariffs")
    ask(at, "net interest income")
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    first, latest = [m for m in at.chat_message if m.name == "assistant"]
    first_sources = next(e for e in first.expander if "passages sent" in e.label)
    latest_sources = next(e for e in latest.expander if "passages sent" in e.label)
    assert not first_sources.proto.expanded and not latest_sources.proto.expanded  # sources start collapsed
    assert 'href="#passage-0-1"' in next(m.value for m in first.markdown if 'class="answer"' in m.value)
    assert 'href="#passage-1-1"' in answer_text(at)  # anchors are unique per answer


def test_example_prompts_stay_available_after_asking():
    at = ask(run(), "tariffs")
    assert [b.label for b in at.button if b.key and b.key.startswith("example-")] == EXAMPLE_QUESTIONS
    assert not any("What do you want to know" in m.value for m in at.markdown)  # the welcome text is gone
    at.button(key="example-1").click().run()
    assert html.escape(EXAMPLE_QUESTIONS[1]) in user_message(at).markdown[0].value
    assert sum(m.name == "assistant" for m in at.chat_message) == 2


def test_new_chat_clears_the_conversation():
    at = ask(run(), "tariffs")
    at.button(key="new-chat").click().run()
    assert not at.chat_message and any("What do you want to know" in m.value for m in at.markdown)


def test_regenerate_replaces_the_last_answer():
    at = ask(run(), "tariffs")
    at.button(key="regenerate").click().run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]  # replaced, not appended
    assert passages(at)


# --- the answer ------------------------------------------------------------------------


def test_answer_with_status_badge_linked_citations_key_points_and_json():
    at = ask(run(), "tariffs on imports from China")
    msg = answer(at)
    assert any(m.value.startswith(":green-badge[") and "Answered" in m.value for m in msg.markdown)
    link = '<a class="cite" href="#passage-0-1">[1]</a>'
    assert link in answer_text(at)
    assert any("AAPL" in m.value and "Tariffs raise component costs" in m.value and link in m.value for m in msg.markdown)
    data = payload(at)
    assert '"question"' in data and '"sources"' in data and '"claims"' in data
    assert '"chunk_id"' in data and "component costs and hurt gross margin" not in data  # no passage text


def test_answer_shows_tokens_cost_and_latency_in_one_line():
    at = ask(run(), "tariffs on imports from China")
    line = next(c.value for c in answer(at).caption if "tokens sent" in c.value)
    assert "1,200 tokens sent" in line and "80 received" in line and "cost unknown" in line and "LLM " in line
    assert "LLM_PRICE_INPUT" in line  # how to price an unknown model


def test_assistant_steps_are_listed_in_its_status_box():
    at = ask(run(), "What tariff risks do Tesla and Apple face?")
    steps = [m.value for m in answer(at).markdown if m.value.startswith(":material/")]
    assert any("Filters:" in s and "TSLA" in s and "AAPL" in s for s in steps)
    assert any("search" in s for s in steps) and any("Sending" in s for s in steps)
    assert any("Answer checked" in s for s in steps)


def test_future_actions_open_a_coming_soon_dialog():
    from app.ui import COMING_SOON

    at = ask(run(), "tariffs")
    for name, (_, title, description) in COMING_SOON.items():
        button = at.button(key=f"{name}-0")
        assert button.help == title
        button.click().run()
        assert not at.exception
        texts = [m.value for m in at.markdown]
        assert any(title in t for t in texts) and description in texts
        assert any("Not available in this prototype yet" in c.value for c in at.caption)


def test_feedback_control_under_the_answer():
    at = ask(run(), "tariffs")
    assert "feedback-0" in at.session_state and at.session_state["feedback-0"] is None  # thumbs shown, none chosen


def test_cited_passages_are_open_and_linked_targets():
    at = ask(run(), "tariffs")  # the scripted answer cites [1]
    assert [e.proto.expanded for e in passages(at)] == [True, False, False, False, False]
    assert all(f'id="passage-0-{i}"' in p.markdown[0].value for i, p in enumerate(passages(at), 1))
    assert any(":target" in m.value for m in at.markdown)  # highlight style for the jumped-to passage


def test_sections_below_the_answer_follow_the_pipeline_and_start_collapsed():
    at = ask(run(), "tariffs")
    names = sections(at)
    assert names[0] == "Retrieval · 1 search"
    assert names[1].endswith("candidates not used (ranked below top-k)")
    assert names[2].endswith("top-k passages not sent (over the 8,000-token budget)")
    assert names[3].startswith("5 passages sent to the LLM")
    details = [e.label for e in answer(at).expander if not e.label.startswith("[")]
    assert details[-1] == "Structured output (JSON)"  # what the model returned comes last
    expanded = {e.label: e.proto.expanded for e in answer(at).expander}
    assert not any(expanded[name] for name in names)  # every section starts collapsed


def test_query_shows_ranked_passages_with_rank_info():
    at = ask(run(), "supply chain tariffs")
    assert len(passages(at)) == 5  # default k for an unknown model
    assert all("·" in label and "FY" in label for label in labels(at))
    caption = passages(at)[0].caption[0].value
    assert "score" in caption and ("vector #" in caption or "keyword #" in caption)
    text = passages(at)[0].markdown[0].value
    assert text.startswith('<div class="passage" id="passage-0-1">') and "tariffs" in text.lower()


def test_generation_can_be_turned_off():
    at = run()
    at.checkbox(key="generate").uncheck()
    ask(at, "tariffs")
    assert "5 passages" in sections(at) and not any("not sent" in s for s in sections(at))
    assert not any(e.label == "Structured output (JSON)" for e in answer(at).expander)
    assert any("generation is off" in c.value for c in answer(at).caption)


def test_llm_failure_shows_error_but_keeps_passages():
    at = ask(run(script_llm_down), "tariffs")
    error = answer(at).error[0].value
    assert "Answer generation failed" in error and "ConnectError" in error
    assert len(passages(at)) == 5


def test_tight_budget_splits_sent_and_not_sent():
    at = ask(run(script_tight_budget), "tariffs")
    sent = len(passages(at))
    assert sent >= 1 and any(s.startswith(f"{sent} passages sent to the LLM") for s in sections(at))
    not_sent = next(s for s in sections(at) if "not sent (over the 100-token budget)" in s)
    assert int(not_sent.split()[0]) == 5 - sent
    not_sent_table = next(e for e in answer(at).expander if e.label == not_sent).dataframe[0].value
    assert len(not_sent_table) == 5 - sent
    assert payload(at).count('"chunk_id"') == sent  # the LLM saw exactly the sent passages


# --- filters ---------------------------------------------------------------------------------


def test_company_filter_limits_results():
    at = run()
    at.sidebar.multiselect[0].select("TSLA")
    ask(at, "tariffs")
    assert companies(at) == {"TSLA"}


def test_filters_detected_from_question_limit_and_balance_results():
    at = run()
    at.sidebar.slider[0].set_value(2)
    ask(at, "What tariff risks do Tesla and Apple face?")
    assert [label.split(" ")[0] for label in labels(at)] == ["TSLA", "TSLA", "AAPL", "AAPL"]  # question order
    assert '"filters": {"tickers": ["TSLA", "AAPL"]}' in payload(at)


def test_sidebar_choice_overrides_detected_filter_and_detection_can_be_off():
    at = run()
    at.sidebar.multiselect[0].select("JPM")
    ask(at, "Apple tariffs and interest income")
    assert companies(at) == {"JPM"}
    next(c for c in at.sidebar.checkbox if c.label == "Detect filters from the question").uncheck()
    at.sidebar.multiselect[0].unselect("JPM")
    apply(at)
    assert not user_message(at).caption and len(companies(at)) > 1


def test_sidebar_company_that_contradicts_the_question_is_flagged():
    at = run()
    at.sidebar.multiselect[0].select("JPM")
    ask(at, "How has NVIDIA's revenue changed?")
    assert "The question implies tickers NVDA, but the sidebar filter (JPM) is used" in answer(at).warning[0].value
    assert companies(at) == {"JPM"}


def test_company_named_but_not_indexed_is_reported_not_silently_dropped():
    at = ask(run(), "How has Boeing's revenue changed?")  # the fake index has AAPL, JPM, NVDA, TSLA
    assert any("BA is not in this collection" in w.value for w in answer(at).warning)
    assert not passages(at)


def test_search_settings_apply_only_when_submitted():
    at = ask(run(), "tariffs")
    assert len(companies(at)) > 1
    at.sidebar.multiselect[0].select("TSLA")
    at.chat_input[0].set_value("tariffs and costs").run()  # a new question alone keeps the saved search settings
    assert len(companies(at)) > 1
    at.sidebar.multiselect[0].select("TSLA")
    apply(at)  # "Apply and re-run" re-asks the last question with the new settings
    assert companies(at) == {"TSLA"}
    assert sum(m.name == "assistant" for m in at.chat_message) == 2  # replaced, not appended


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


def test_year_and_filing_type_filters():
    at = run()
    at.sidebar.selectbox[0].select("10-Q")
    ask(at, "net interest income")
    assert labels(at) == ["JPM · 10-Q FY2025 · chunk 1/10"]

    at.sidebar.selectbox[0].select("Any")
    at.sidebar.multiselect[1].select(2024)
    ask(at, "revenue")
    assert labels(at) and all("FY2024" in label for label in labels(at))


# --- search modes and failures --------------------------------------------------------------


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
    assert not passages(at)
    assert "No matching passages" in answer(at).warning[0].value


def test_backend_down_shows_error_and_keyword_fallback_works():
    at = ask(run(script_backend_down), "tariffs")
    assert "Embedding backend unreachable" in answer(at).error[0].value
    assert not passages(at)
    at.sidebar.radio[0].set_value("keyword")
    apply(at)
    assert not answer(at).error and passages(at)


# --- retrieval details ---------------------------------------------------------------------------


def test_retrieval_shows_chroma_and_bm25_requests_and_top_k():
    at = run()
    at.sidebar.slider[0].set_value(3)
    at.sidebar.selectbox[0].select("10-K")
    ask(at, "tariffs on imports")
    assert any("top-k 3" in m.value and "Search 1/1" in m.value for m in answer(at).markdown)
    chroma, bm25 = requests(at)
    assert '"n_results": 50' in chroma and '"where": {' in chroma and '"filing_type": {"$in": ["10-K"]}' in chroma
    assert "embed(" in chroma and '"collection"' in chroma
    assert '"tariffs"' in bm25 and '"imports"' in bm25 and '"corpus_chunks": 12' in bm25


def test_balanced_search_shows_one_query_per_company():
    at = run()
    at.sidebar.multiselect[0].select("AAPL").select("TSLA")
    ask(at, "risks")
    searches = [m.value for m in answer(at).markdown if m.value.startswith("**Search ")]
    assert len(searches) == 2 and all("/2**" in s for s in searches)
    assert '"AAPL"' in requests(at)[0] and '"TSLA"' in requests(at)[2]


def test_keyword_mode_shows_chroma_not_run():
    at = run()
    at.sidebar.radio[0].set_value("keyword")
    ask(at, "tariffs")
    assert any(c.value == "not run in keyword mode" for c in answer(at).caption)


def test_candidates_below_top_k_are_listed_as_not_used():
    at = run()
    at.sidebar.slider[0].set_value(2)
    ask(at, "revenue tariffs")
    unused = next(s for s in sections(at) if s.endswith("candidates not used (ranked below top-k)"))
    n = int(unused.split()[0])
    assert n == 10  # all 12 chunks are vector candidates, 2 made top-k
    table = answer(at).dataframe[-1].value
    assert len(table) == n and {"search", "vector #", "cosine", "bm25 #", "rrf", "tokens", "text"} <= set(table.columns)
    assert table["vector #"].min() > 0 and not set(table["passage"]) & set(labels(at))


# --- model and context settings ----------------------------------------------------------------------


def test_context_budget_and_k_default_from_the_model_profile():
    at = run()
    k, budget = at.sidebar.slider[0], at.sidebar.slider[1]
    assert (k.value, budget.value, budget.max) == (5, 8_000, 16_384 - 1_024 - 1_000)
    assert any("16,384-token window" in c.value and "1,024 reserved" in c.value for c in at.sidebar.caption)


def test_model_picker_defaults_to_gpt_6_luna_with_an_openai_key(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    at = run(script_pick_model)
    picker = at.selectbox(key="llm")
    assert picker.options == ["ollama: gemma4:e2b", "openai: gpt-6-luna", "openai: gpt-4o-mini"]
    assert picker.value == ("openai", "gpt-6-luna")
    assert (at.slider[0].value, at.slider[1].value, at.slider[1].max) == (8, 24_000, 64_000)
    assert any("gpt-6-luna" in c.value and "1,100,000-token window" in c.value and "4,000 reserved" in c.value for c in at.caption)


def test_model_picker_falls_back_to_local_gemma_without_a_key_and_follows_the_choice(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    at = run(script_pick_model)
    picker = at.selectbox(key="llm")
    assert picker.value == ("ollama", "gemma4:e2b")
    assert (at.slider[0].value, at.slider[1].value) == (5, 8_000)
    assert any("16,384-token window" in c.value for c in at.caption)

    picker.set_value(("openai", "gpt-6-luna")).run()
    assert not at.exception
    assert (at.slider[0].value, at.slider[1].value) == (8, 24_000)
