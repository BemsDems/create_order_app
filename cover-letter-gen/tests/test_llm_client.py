"""Tests for the LLM client wrapper (v4).

Covers the retry / fallback behaviors that previously had no tests:
- 429 rate-limit triggers backoff and retry (not surfaced as error).
- Empty `content` + empty `reasoning_content` triggers retry, not silent ""
  passthrough.
- `Retry-After` header is honored.
- 400 with json_mode disables json_mode and retries.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from src.llm_client import LLMClient, LLMConfig, _parse_retry_after


def _ok_response(text: str) -> Dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def _empty_response() -> Dict[str, Any]:
    return {"choices": [{"message": {"content": ""}}]}


def _mock_transport(
    sequence: List[Callable[[httpx.Request], httpx.Response]],
) -> httpx.MockTransport:
    """Build an httpx MockTransport that pops a handler per request."""
    seq = list(sequence)

    def handler(request: httpx.Request) -> httpx.Response:
        if not seq:
            raise AssertionError("Unexpected extra request")
        return seq.pop(0)(request)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_retry_on_429_with_retry_after_header():
    """A 429 with Retry-After: 0 must trigger retry, not raise."""
    calls: List[httpx.Request] = []

    def first(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})

    def second(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200, json=_ok_response("hello"))

    async with httpx.AsyncClient(transport=_mock_transport([first, second])) as http:
        client = LLMClient(LLMConfig(api_key="k", endpoint="http://x/v1", max_retries=3), http)
        result = await client.generate("sys", "user", json_mode=False)

    assert result == "hello"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_429_eventually_raises_after_max_retries():
    """If 429 persists for `max_retries`, the client raises RuntimeError."""
    def always_429(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})

    async with httpx.AsyncClient(
        transport=_mock_transport([always_429, always_429])
    ) as http:
        client = LLMClient(LLMConfig(api_key="k", endpoint="http://x/v1", max_retries=2), http)
        with pytest.raises(RuntimeError, match="LLM call failed after 2 attempts"):
            await client.generate("sys", "user")


@pytest.mark.asyncio
async def test_empty_content_triggers_retry():
    """A 200 with empty content+reasoning must retry, not return ''."""
    def first(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_empty_response())

    def second(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("real answer"))

    async with httpx.AsyncClient(transport=_mock_transport([first, second])) as http:
        client = LLMClient(LLMConfig(api_key="k", endpoint="http://x/v1", max_retries=3), http)
        result = await client.generate("sys", "user")

    assert result == "real answer"


@pytest.mark.asyncio
async def test_empty_content_eventually_raises_after_max_retries():
    """Empty content for all attempts → RuntimeError, not silent empty string."""
    def empty(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_empty_response())

    async with httpx.AsyncClient(transport=_mock_transport([empty, empty])) as http:
        # Reset module-level cache so this test is independent.
        LLMClient._json_mode_cache.pop("http://x-empty/v1", None)
        client = LLMClient(
            LLMConfig(api_key="k", endpoint="http://x-empty/v1", max_retries=2),
            http,
        )
        with pytest.raises(RuntimeError, match="empty content"):
            await client.generate("sys", "user")


@pytest.mark.asyncio
async def test_reasoning_content_used_when_content_empty():
    """A non-empty `reasoning_content` is the fallback for empty `content`."""
    def with_reasoning(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": "", "reasoning_content": "fallback answer"},
            }],
        })

    async with httpx.AsyncClient(transport=_mock_transport([with_reasoning])) as http:
        client = LLMClient(LLMConfig(api_key="k", endpoint="http://x/v1"), http)
        result = await client.generate("sys", "user")

    assert result == "fallback answer"


@pytest.mark.asyncio
async def test_json_mode_400_disables_and_retries():
    """A 400 with json_mode enabled must disable json_mode and retry."""
    captured_payloads: List[Dict[str, Any]] = []

    def first(req: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(req.content.decode()))
        return httpx.Response(400, json={"error": "json_mode not supported"})

    def second(req: httpx.Request) -> httpx.Response:
        captured_payloads.append(json.loads(req.content.decode()))
        return httpx.Response(200, json=_ok_response('{"ok":true}'))

    # Reset cache for isolation.
    LLMClient._json_mode_cache.pop("http://x-json/v1", None)
    async with httpx.AsyncClient(transport=_mock_transport([first, second])) as http:
        client = LLMClient(
            LLMConfig(api_key="k", endpoint="http://x-json/v1", max_retries=3),
            http,
        )
        result = await client.generate("sys", "user", json_mode=True)

    assert result == '{"ok":true}'
    assert "response_format" in captured_payloads[0]
    assert "response_format" not in captured_payloads[1]


def test_parse_retry_after_integer():
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after("  10 ") == 10.0
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_invalid():
    assert _parse_retry_after("") is None
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("Mon, 01 Jan 2024 00:00:00 GMT") is None
    assert _parse_retry_after("-1") is None


def test_parse_retry_after_capped():
    """A misconfigured proxy could send 999999 — we cap at 60s."""
    assert _parse_retry_after("99999") == 60.0


@pytest.mark.asyncio
async def test_5xx_triggers_retry():
    """A transient 5xx must trigger retry, then succeed on the second try."""
    def first(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    def second(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("recovered"))

    async with httpx.AsyncClient(transport=_mock_transport([first, second])) as http:
        client = LLMClient(LLMConfig(api_key="k", endpoint="http://x/v1", max_retries=3), http)
        result = await client.generate("sys", "user")

    assert result == "recovered"
