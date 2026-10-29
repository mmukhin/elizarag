import json

import httpx
import pytest

from app.edgar.llm import (
    DEFAULT_PROFILE,
    MAX_USEFUL_BUDGET,
    PROMPT_OVERHEAD,
    ChatModel,
    OllamaChat,
    OpenAIChat,
    StructuredOutputError,
    context_profile,
    get_chat_model,
)
from app.edgar.qa import ModelAnswer, answer_question
from tests.fakes import ScriptedChat, answer_json, hits_for, sample_chunks


def ollama_with(handler) -> OllamaChat:
    llm = OllamaChat(model="gemma4:e2b", num_ctx=4096)
    llm.client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return llm


# --- ChatModel.generate ------------------------------------------------------


def test_generate_parses_valid_json():
    result = ScriptedChat(answer_json()).generate("s", "u", ModelAnswer)
    assert isinstance(result, ModelAnswer) and result.claims[0].passages == [1]


def test_generate_retries_once_on_invalid_output():
    llm = ScriptedChat("not json", answer_json())
    assert llm.generate("s", "u", ModelAnswer).answer.startswith("Apple")
    assert len(llm.calls) == 2


def test_generate_retries_on_schema_mismatch():
    llm = ScriptedChat(json.dumps({"wrong": "shape"}), answer_json())
    assert llm.generate("s", "u", ModelAnswer)
    assert len(llm.calls) == 2


def test_generate_raises_after_max_attempts():
    llm = ScriptedChat("garbage")
    with pytest.raises(StructuredOutputError, match="no valid ModelAnswer after 2 attempts"):
        llm.generate("s", "u", ModelAnswer)
    assert len(llm.calls) == 2


def test_generate_accepts_already_parsed_instance():
    instance = ModelAnswer(status="answered", answer="done [1]")
    assert ScriptedChat(instance).generate("s", "u", ModelAnswer) is instance


# --- Ollama -------------------------------------------------------------------


def test_ollama_request_carries_schema_context_window_and_no_thinking():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": answer_json()}})

    result = ollama_with(handler).generate("SYSTEM", "USER", ModelAnswer)

    body = seen["body"]
    assert seen["url"] == "http://x/api/chat"
    assert body["model"] == "gemma4:e2b" and body["stream"] is False and body["think"] is False
    assert body["format"] == ModelAnswer.model_json_schema()
    assert body["options"] == {"temperature": 0.0, "num_ctx": 4096, "num_predict": 1024}
    assert body["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert result.status == "answered" and result.claims[0].passages == [1]


def test_ollama_http_error_propagates():
    llm = ollama_with(lambda request: httpx.Response(500, text="model not found"))
    with pytest.raises(httpx.HTTPStatusError):
        llm.generate("s", "u", ModelAnswer)


def test_ollama_end_to_end_through_answer_question():
    llm = ollama_with(lambda r: httpx.Response(200, json={"message": {"content": answer_json()}}))
    result = answer_question("tariffs?", hits_for(*sample_chunks()[:2]), llm)
    assert result.model == "gemma4:e2b" and result.warnings == []


# --- OpenAI ---------------------------------------------------------------------


class FakeRaw:
    content = answer_json()
    usage_metadata = {"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300}


class FakeStructured:
    def __init__(self, parent, schema):
        self.parent, self.schema = parent, schema

    def invoke(self, messages):
        self.parent.messages = messages
        return {"raw": FakeRaw(), "parsed": self.schema.model_validate_json(answer_json()), "parsing_error": None}


class FakeOpenAIClient:
    def with_structured_output(self, schema, include_raw=False):
        self.schema, self.include_raw = schema, include_raw
        return FakeStructured(self, schema)


def test_openai_uses_structured_output_and_returns_schema_instance():
    client = FakeOpenAIClient()
    result = OpenAIChat(model="gpt-test", client=client).generate("SYSTEM", "USER", ModelAnswer)
    assert client.schema is ModelAnswer and client.include_raw
    assert client.messages == [("system", "SYSTEM"), ("user", "USER")]
    assert isinstance(result, ModelAnswer)


def test_openai_end_to_end_through_answer_question():
    result = answer_question("q", hits_for(*sample_chunks()[:2]), OpenAIChat(model="gpt-test", client=FakeOpenAIClient()))
    assert result.model == "gpt-test" and result.claims


# --- provider selection ---------------------------------------------------------


def test_get_chat_model_defaults_to_local_gemma(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    llm = get_chat_model()
    assert isinstance(llm, OllamaChat) and llm.model == "gemma4:e2b"


def test_get_chat_model_env_and_args(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    llm = get_chat_model()
    assert isinstance(llm, ChatModel) and isinstance(llm, OpenAIChat) and llm.model == "gpt-6-luna"

    monkeypatch.setenv("LLM_MODEL", "gpt-4.1")
    assert get_chat_model().model == "gpt-4.1"
    assert get_chat_model("ollama", "tev1:0.8b").model == "tev1:0.8b"  # args beat env provider; explicit model wins


def test_get_chat_model_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        get_chat_model("anthropic-nope")



# --- usage: tokens, latency, cost --------------------------------------------------


def test_openai_usage_and_cost_from_price_table(monkeypatch):
    monkeypatch.delenv("LLM_PRICE_INPUT", raising=False)
    llm = OpenAIChat(model="gpt-6-luna", client=FakeOpenAIClient())
    llm.generate("s", "u", ModelAnswer)
    usage = llm.usage
    assert (usage.model, usage.calls, usage.input_tokens, usage.output_tokens) == ("gpt-6-luna", 1, 2000, 300)
    assert usage.cost_usd == pytest.approx((2000 * 0.10 + 300 * 0.50) / 1e6) and usage.seconds >= 0


def test_unknown_model_has_no_cost_unless_priced_by_env(monkeypatch):
    monkeypatch.delenv("LLM_PRICE_INPUT", raising=False)
    llm = OpenAIChat(model="gpt-test", client=FakeOpenAIClient())
    llm.generate("s", "u", ModelAnswer)
    assert llm.usage.cost_usd is None
    monkeypatch.setenv("LLM_PRICE_INPUT", "1.0")
    monkeypatch.setenv("LLM_PRICE_OUTPUT", "2.0")
    llm.generate("s", "u", ModelAnswer)
    assert llm.usage.cost_usd == pytest.approx((2000 * 1.0 + 300 * 2.0) / 1e6)


def test_ollama_reports_token_counts_and_is_free():
    body = {"message": {"content": answer_json()}, "prompt_eval_count": 1500, "eval_count": 120}
    llm = ollama_with(lambda r: httpx.Response(200, json=body))
    llm.generate("s", "u", ModelAnswer)
    assert (llm.usage.input_tokens, llm.usage.output_tokens, llm.usage.cost_usd) == (1500, 120, 0.0)


def test_usage_sums_retries():
    llm = ScriptedChat("not json", answer_json())
    llm.generate("s", "u", ModelAnswer)
    assert (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) == (2, 2400, 160)


def test_answer_question_attaches_usage_only_when_model_called():
    hits = hits_for(*sample_chunks()[:2])
    assert answer_question("q", hits, ScriptedChat(answer_json())).usage.input_tokens == 1200
    assert answer_question("q", [], ScriptedChat(answer_json())).usage is None


def test_openai_does_not_send_temperature_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert OpenAIChat().client.temperature is None  # gpt-6-luna rejects temperature=0
    assert OpenAIChat(temperature=0.2).client.temperature == 0.2


# --- context profiles ---------------------------------------------------------------


def test_context_profiles_fit_the_window_and_unknown_models_are_conservative():
    gemma, luna = context_profile("gemma4:e2b"), context_profile("gpt-6-luna")
    assert gemma.budget + gemma.output_reserve + PROMPT_OVERHEAD <= gemma.context_window
    assert gemma.max_budget == 16_384 - 1_024 - PROMPT_OVERHEAD
    assert luna.budget > gemma.budget and luna.max_budget == MAX_USEFUL_BUDGET  # 1.1M window, capped
    assert context_profile("some-new-model") == DEFAULT_PROFILE


def test_ollama_window_and_output_cap_come_from_the_profile():
    llm = OllamaChat(model="gemma4:e2b")
    assert (llm.num_ctx, llm.num_predict) == (16_384, 1_024)
    assert OllamaChat(model="gemma4:e2b", num_ctx=8_192).num_ctx == 8_192
