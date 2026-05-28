"""Async LLM client for OpenAI-compatible chat-completions endpoints.

Features:
- Retries with exponential backoff on 5xx / network errors.
- Optional JSON mode (`response_format`) with graceful fallback if the
  server rejects the field (some OpenAI-compatible proxies don't support it).
- Reads `reasoning_content` as a fallback when `content` is empty
  (some proxies/models return their final answer there).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    api_key: str
    endpoint: str = "http://localhost:20128/v1/chat/completions"
    model: str = "openrouter/owl-alpha"
    timeout_seconds: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        # Expand ${ENV_VAR} placeholders for api_key.
        api_key = str(data.get("api_key", ""))
        if api_key.startswith("${") and api_key.endswith("}"):
            api_key = os.environ.get(api_key[2:-1], "")
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")
        return cls(
            api_key=api_key,
            endpoint=str(data.get("endpoint", os.environ.get("LLM_ENDPOINT", cls.endpoint))),
            model=str(data.get("model", os.environ.get("LLM_MODEL", cls.model))),
            timeout_seconds=float(data.get("timeout_seconds", cls.timeout_seconds)),
            max_retries=int(data.get("max_retries", cls.max_retries)),
        )


@dataclass
class LLMCallStats:
    attempts: int = 0
    last_error: Optional[str] = None
    json_mode_supported: Optional[bool] = None


class LLMClient:
    """Thin async wrapper over an OpenAI-compatible /v1/chat/completions API."""

    # Module-level memoization across instances to avoid re-discovering
    # JSON-mode support after the first failed attempt.
    _json_mode_cache: Dict[str, bool] = {}

    def __init__(self, config: LLMConfig, http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._client = http_client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self.stats = LLMCallStats()

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 800,
        json_mode: bool = False,
    ) -> str:
        """Single completion call.

        Returns the assistant text (or the contents of `reasoning_content` if
        the proxy put its final answer there).
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return await self._post_with_retries(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    async def _post_with_retries(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        endpoint = self.config.endpoint
        json_mode_supported = self._json_mode_cache.get(endpoint, True)
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            self.stats.attempts = attempt
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode and json_mode_supported:
                payload["response_format"] = {"type": "json_object"}

            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }

            try:
                resp = await self._client.post(endpoint, json=payload, headers=headers)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_exc = exc
                logger.warning("LLM transport error (attempt %d/%d): %s", attempt, self.config.max_retries, exc)
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code == 400 and json_mode and json_mode_supported:
                # Some proxies reject `response_format`. Disable and retry once
                # within the same attempt budget.
                logger.info("Endpoint %s rejected json_mode; disabling and retrying.", endpoint)
                json_mode_supported = False
                self._json_mode_cache[endpoint] = False
                continue

            if resp.status_code == 429:
                # Rate-limited. Honor Retry-After if present; otherwise use a
                # doubled exponential backoff.
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                wait = retry_after if retry_after is not None else _backoff(attempt) * 2
                last_exc = httpx.HTTPStatusError(
                    "rate limited (429)", request=resp.request, response=resp,
                )
                logger.warning(
                    "LLM 429 rate-limited (attempt %d/%d); sleeping %.2fs",
                    attempt, self.config.max_retries, wait,
                )
                await asyncio.sleep(wait)
                continue

            if 500 <= resp.status_code < 600:
                last_exc = httpx.HTTPStatusError(
                    f"server {resp.status_code}", request=resp.request, response=resp
                )
                logger.warning(
                    "LLM 5xx (attempt %d/%d): status=%d body=%s",
                    attempt, self.config.max_retries, resp.status_code, resp.text[:200],
                )
                await asyncio.sleep(_backoff(attempt))
                continue

            if resp.status_code >= 400:
                # 4xx other than 400/429 — don't retry, surface error.
                resp.raise_for_status()

            self._json_mode_cache.setdefault(endpoint, json_mode_supported)
            self.stats.json_mode_supported = json_mode_supported
            content = _extract_content(resp.json())
            if not content:
                # The provider returned a 200 but no usable content.
                # Treat as a transient error and retry rather than letting
                # the caller deal with an empty string (which usually leads
                # to confusing downstream validation failures).
                last_exc = RuntimeError("empty content from provider")
                logger.warning(
                    "LLM returned empty content (attempt %d/%d); retrying",
                    attempt, self.config.max_retries,
                )
                await asyncio.sleep(_backoff(attempt))
                continue
            return content

        # Exhausted retries.
        self.stats.last_error = str(last_exc) if last_exc else "unknown"
        raise RuntimeError(
            f"LLM call failed after {self.config.max_retries} attempts: {self.stats.last_error}"
        )


def _extract_content(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = msg.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def _backoff(attempt: int) -> float:
    """Exponential backoff: 0.5s, 1.0s, 2.0s, capped at 8s."""
    return min(0.5 * (2 ** (attempt - 1)), 8.0)


def _parse_retry_after(header_value: Optional[str]) -> Optional[float]:
    """Parse a `Retry-After` header value as seconds.

    Per RFC 7231 the value can be either an integer number of seconds or an
    HTTP-date. We only support the integer form here — that's what all the
    OpenAI-compatible providers actually emit. Capped at 60s to avoid a
    runaway sleep on a misconfigured proxy.
    """
    if not header_value:
        return None
    try:
        seconds = float(header_value.strip())
    except ValueError:
        return None
    if seconds < 0:
        return None
    return min(seconds, 60.0)
