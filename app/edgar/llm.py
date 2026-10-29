"""Chat models that return validated structured output. Providers: Ollama (default) and OpenAI."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.edgar.vectorstore import OLLAMA_URL

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {"ollama": "gemma4:e2b", "openai": "gpt-6-luna"}
PROVIDERS = tuple(DEFAULT_MODELS)
MODEL_CHOICES = [("ollama", "gemma4:e2b"), ("openai", "gpt-6-luna"), ("openai", "gpt-4o-mini")]  # the UI's model picker
# $ per 1M (input, output) tokens, from third-party listings (Oct 2026); check https://developers.openai.com/api/docs/pricing.
# Override with LLM_PRICE_INPUT / LLM_PRICE_OUTPUT. Local Ollama models cost 0.
PRICES = {"gpt-6-luna": (0.10, 0.50), "gpt-4o-mini": (0.15, 0.60)}


PROMPT_OVERHEAD = 1_000  # system prompt + question + passage headers, rounded up
MAX_USEFUL_BUDGET = 64_000  # beyond this, more passages mostly add noise, latency and cost


@dataclass(frozen=True)
class ContextProfile:
    """How much retrieved text to send a model, and how much room to leave for its answer."""

    context_window: int  # tokens the model accepts (for Ollama: the num_ctx we request)
    budget: int  # default tokens of passages per question
    k: int  # default passages retrieved (per company when balanced)
    output_reserve: int  # tokens kept for the answer (also caps Ollama's output)

    @property
    def max_budget(self) -> int:
        """Largest passage budget that still fits the window, capped where extra context stops helping."""
        return max(1_000, min(self.context_window - self.output_reserve - PROMPT_OVERHEAD, MAX_USEFUL_BUDGET))


# gemma4:e2b is a 2B model: long prompts dilute it and Ollama's memory grows with num_ctx, so keep it small.
# gpt-6-luna takes 1.1M tokens, but ~24k (3 companies x 8 passages x ~800 tokens) covers comparisons for ~$0.003.
PROFILES = {
    "gemma4:e2b": ContextProfile(context_window=16_384, budget=8_000, k=5, output_reserve=1_024),
    "gpt-6-luna": ContextProfile(context_window=1_100_000, budget=24_000, k=8, output_reserve=4_000),
    "gpt-4o-mini": ContextProfile(context_window=128_000, budget=16_000, k=6, output_reserve=2_000),
}
DEFAULT_PROFILE = ContextProfile(context_window=16_384, budget=8_000, k=5, output_reserve=1_024)  # unknown models


def context_profile(model: str) -> ContextProfile:
    return PROFILES.get(model, DEFAULT_PROFILE)


class Usage(BaseModel):
    """Tokens, latency and cost of one `generate` call, summed over retries."""

    model: str = ""
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = None  # None when the model's price is unknown


def price(model: str, local: bool = False) -> tuple[float, float] | None:
    """($ per 1M input tokens, $ per 1M output tokens) for `model`, or None if unknown."""
    if os.environ.get("LLM_PRICE_INPUT") and os.environ.get("LLM_PRICE_OUTPUT"):
        return float(os.environ["LLM_PRICE_INPUT"]), float(os.environ["LLM_PRICE_OUTPUT"])
    return (0.0, 0.0) if local else PRICES.get(model)


class StructuredOutputError(RuntimeError):
    """The model kept returning output that does not match the schema."""


class ChatModel(ABC):
    """`generate(system, user, schema)` returns an instance of `schema`, retrying on bad output.

    After each call, `usage` holds its tokens, latency and cost.
    """

    max_attempts = 2
    local = False  # runs on this machine, so it costs nothing
    usage = Usage()

    @abstractmethod
    def _complete(self, system: str, user: str, schema: type[T]) -> T | str:
        """One model call. Return a parsed `schema` instance or raw JSON text to validate; report tokens via `_count`."""

    def _count(self, input_tokens: int, output_tokens: int) -> None:
        self.usage.input_tokens += input_tokens or 0
        self.usage.output_tokens += output_tokens or 0

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        model = getattr(self, "model", type(self).__name__)
        self.usage = Usage(model=model)
        try:
            return self._generate(system, user, schema)
        finally:
            rates = price(model, self.local)
            if rates is not None:
                self.usage.cost_usd = (self.usage.input_tokens * rates[0] + self.usage.output_tokens * rates[1]) / 1e6

    def _generate(self, system: str, user: str, schema: type[T]) -> T:
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            started = time.monotonic()
            result = self._complete(system, user, schema)
            self.usage.calls += 1
            self.usage.seconds += time.monotonic() - started
            if isinstance(result, schema):
                return result
            try:
                return schema.model_validate_json(result)
            except ValidationError as exc:
                last_error = exc
        raise StructuredOutputError(f"no valid {schema.__name__} after {self.max_attempts} attempts: {last_error}")


class OllamaChat(ChatModel):
    """Local model through Ollama's /api/chat, constrained by the schema via `format`."""

    local = True

    def __init__(
        self,
        model: str = DEFAULT_MODELS["ollama"],
        base_url: str = OLLAMA_URL,
        num_ctx: int | None = None,
        think: bool = False,
        temperature: float = 0.0,
        timeout: float = 600.0,
    ):
        profile = context_profile(model)
        self.model = model
        self.num_ctx = num_ctx or profile.context_window  # Ollama's default (4k) would silently truncate passages
        self.num_predict = profile.output_reserve
        self.think = think
        self.temperature = temperature
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def _complete(self, system: str, user: str, schema: type[T]) -> str:
        response = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "think": self.think,
                "format": schema.model_json_schema(),
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.num_predict},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        response.raise_for_status()
        body = response.json()
        self._count(body.get("prompt_eval_count", 0), body.get("eval_count", 0))
        return body["message"]["content"]


class OpenAIChat(ChatModel):
    """OpenAI through LangChain structured output (needs OPENAI_API_KEY)."""

    def __init__(self, model: str = DEFAULT_MODELS["openai"], temperature: float | None = None, client=None):
        if client is None:
            from langchain_openai import ChatOpenAI

            # newer models (gpt-6-luna) reject any temperature but the default, so only send one if asked
            client = ChatOpenAI(model=model, **({} if temperature is None else {"temperature": temperature}))
        self.model = model
        self.client = client

    def _complete(self, system: str, user: str, schema: type[T]) -> T | str:
        runnable = self.client.with_structured_output(schema, include_raw=True)
        out = runnable.invoke([("system", system), ("user", user)])
        usage = getattr(out["raw"], "usage_metadata", None) or {}
        self._count(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
        return out["parsed"] if out["parsed"] is not None else str(out["raw"].content)


def get_chat_model(provider: str | None = None, model: str | None = None) -> ChatModel:
    """Pick the backend from args or LLM_PROVIDER / LLM_MODEL / OLLAMA_URL (default: local Ollama)."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "ollama")).lower()
    model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider)
    if provider == "ollama":
        return OllamaChat(model=model, base_url=os.environ.get("OLLAMA_URL", OLLAMA_URL))
    if provider == "openai":
        return OpenAIChat(model=model)
    raise ValueError(f"unknown LLM provider {provider!r} (expected one of {PROVIDERS})")


def model_choices() -> list[tuple[str, str]]:
    """(provider, model) pairs to offer, plus $LLM_PROVIDER / $LLM_MODEL if set to something not listed."""
    choices = list(MODEL_CHOICES)
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    env = (provider, os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider, ""))
    if env[1] and env not in choices:
        choices.append(env)
    return choices


def default_choice(choices: list[tuple[str, str]]) -> int:
    """Index of the UI's default model: $LLM_PROVIDER / $LLM_MODEL if set, else gpt-6-luna when an OpenAI key is
    available, else local gemma (so the app still answers offline)."""
    if os.environ.get("LLM_PROVIDER") or os.environ.get("LLM_MODEL"):
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        wanted = (provider, os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider, ""))
    else:
        wanted = ("openai", DEFAULT_MODELS["openai"]) if os.environ.get("OPENAI_API_KEY") else ("ollama", DEFAULT_MODELS["ollama"])
    return choices.index(wanted) if wanted in choices else 0
