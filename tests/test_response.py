"""Tests for LLMResponse lazy parsing and AgentResult."""

import asyncio

import pytest
from pydantic import BaseModel

from llm_interaction import LLMResponse


class TestLLMResponse:
    def test_text(self):
        resp = LLMResponse(text="hello world", response_id="r1")
        assert resp.text == "hello world"

    def test_json(self):
        resp = LLMResponse(
            text='```json\n{"topics": ["ai", "ml"]}\n```', response_id="r1"
        )
        assert resp.json() == {"topics": ["ai", "ml"]}

    def test_yaml(self):
        resp = LLMResponse(text="```yaml\ntopics:\n- ai\n- ml\n```", response_id="r1")
        result = resp.yaml()
        assert result == {"topics": ["ai", "ml"]}

    def test_scratchpad_json(self):
        text = (
            "I analyzed the data carefully.\n\n"
            '```json\n{"count": 5}\n```'
        )
        resp = LLMResponse(text=text, response_id="r1")
        scratchpad, data = resp.scratchpad_json()
        assert "analyzed" in scratchpad
        assert data == {"count": 5}

    def test_scratchpad_yaml(self):
        text = "My reasoning:\n\n```yaml\ncount: 5\n```"
        resp = LLMResponse(text=text, response_id="r1")
        scratchpad, data = resp.scratchpad_yaml()
        assert "reasoning" in scratchpad
        assert data == {"count": 5}

    def test_repr(self):
        resp = LLMResponse(text="short", response_id="r1")
        assert "short" in repr(resp)


# ---------------------------------------------------------------------------
# LLMResponse.parse()
# ---------------------------------------------------------------------------


class TopicResult(BaseModel):
    topics: list[str]
    confidence: float


class TestLLMResponseParse:
    @pytest.mark.asyncio
    async def test_parse_valid(self):
        text = '```json\n{"topics": ["ai"], "confidence": 0.95}\n```'
        resp = LLMResponse(text=text, response_id="r1")
        result = await resp.parse(TopicResult, max_retries=0)
        assert isinstance(result, TopicResult)
        assert result.topics == ["ai"]
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_parse_invalid_no_client_raises(self):
        text = '```json\n{"topics": "not_a_list"}\n```'
        resp = LLMResponse(text=text, response_id="r1", _client=None)
        with pytest.raises(RuntimeError, match="no client"):
            await resp.parse(TopicResult, max_retries=1)

    @pytest.mark.asyncio
    async def test_parse_invalid_no_retries_raises(self):
        text = '```json\n{"topics": "not_a_list"}\n```'
        resp = LLMResponse(text=text, response_id="r1")
        with pytest.raises(Exception):
            await resp.parse(TopicResult, max_retries=0)
