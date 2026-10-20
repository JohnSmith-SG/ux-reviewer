"""Общение с нейросетью: OpenAI-совместимый API (по умолчанию ProxyAPI), ответ строго в JSON.

Ключ — из PROXYAPI_KEY (или OPENAI_API_KEY), либо из файла по пути в PROXYAPI_KEY_FILE.
Локально переменные можно положить в .env (см. .env.example), в коде ключа нет.
Сбои API и «битый» ответ модели — до 3 попыток с паузой (tenacity).
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (before_sleep_log, retry, retry_if_exception, stop_after_attempt,
                      wait_exponential)

load_dotenv()
log = logging.getLogger("ux.llm")

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openai.api.proxyapi.ru/v1")
MODEL = os.environ.get("UX_MODEL", "openai/gpt-4o")
TIMEOUT = 90
ATTEMPTS = 3


class BadAnswer(ValueError):
    """Модель вернула не тот JSON — повторяем запрос."""


class LLMError(RuntimeError):
    """Нет ключа или нейросеть не ответила после всех попыток."""


@dataclass
class Usage:
    model: str
    tokens_in: int
    tokens_out: int


def _api_key() -> str:
    key = os.environ.get("PROXYAPI_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key and os.environ.get("PROXYAPI_KEY_FILE"):
        key = Path(os.environ["PROXYAPI_KEY_FILE"]).read_text(encoding="utf-8").strip()
    if not key:
        raise LLMError("Нет ключа API — задайте PROXYAPI_KEY (или PROXYAPI_KEY_FILE)")
    return key


_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        # max_retries=0: повторами управляет tenacity, чтобы их было ровно до ATTEMPTS
        _client = OpenAI(api_key=_api_key(), base_url=BASE_URL, timeout=TIMEOUT, max_retries=0)
    return _client


def _retryable(e: BaseException) -> bool:
    if isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError, BadAnswer)):
        return True
    # 5xx — временная беда на стороне поставщика; 4xx (ключ, баланс, запрос) повтор не исправит
    return isinstance(e, APIStatusError) and e.status_code >= 500


@retry(stop=stop_after_attempt(ATTEMPTS), wait=wait_exponential(multiplier=2, min=2, max=10),
       retry=retry_if_exception(_retryable), before_sleep=before_sleep_log(log, logging.WARNING),
       reraise=True)
def ask_json(system: str, user: str, schema: dict[str, Any], *, name: str = "answer",
             check: Callable[[dict[str, Any]], None] | None = None,
             model: str = MODEL) -> tuple[dict[str, Any], Usage]:
    """Один запрос: system — роль, user — данные. Модель обязана ответить по JSON-схеме
    (structured outputs); check() дополнительно проверяет смысл и бросает BadAnswer → повтор."""
    r = client().chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": name, "strict": True, "schema": schema}},
    )
    text = r.choices[0].message.content or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise BadAnswer(f"ответ не JSON: {e}") from e
    if not isinstance(data, dict):
        raise BadAnswer("ответ не объект JSON")
    if check:
        check(data)
    usage = Usage(model, r.usage.prompt_tokens if r.usage else 0,
                  r.usage.completion_tokens if r.usage else 0)
    return data, usage
