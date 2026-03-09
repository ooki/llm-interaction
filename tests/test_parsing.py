"""Tests for parsing helpers: JSON, YAML, scratchpad, fenced block extraction."""

import pytest

from llm_interaction.parsing import (
    _extract_fenced_block,
    _parse_json,
    _parse_yaml,
    _split_scratchpad,
)


class TestParsingHelpers:
    def test_extract_fenced_json(self):
        content = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        assert _extract_fenced_block(content, "json") == '{"key": "value"}'

    def test_extract_fenced_yaml(self):
        content = "Preamble\n```yaml\nkey: value\n```"
        assert _extract_fenced_block(content, "yaml") == "key: value"

    def test_extract_no_fence(self):
        content = '{"key": "value"}'
        assert _extract_fenced_block(content, "json") == '{"key": "value"}'

    def test_extract_last_block(self):
        content = '```json\n{"a": 1}\n```\nMiddle\n```json\n{"b": 2}\n```'
        assert _extract_fenced_block(content, "json") == '{"b": 2}'

    def test_parse_json_valid(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_parse_json_with_repair(self):
        result = _parse_json('{"a": 1,}')
        assert result == {"a": 1}

    def test_parse_json_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_json("")

    def test_parse_yaml_valid(self):
        assert _parse_yaml("key: value\ncount: 3") == {
            "key": "value",
            "count": 3,
        }

    def test_parse_yaml_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _parse_yaml("")

    def test_split_scratchpad_json(self):
        content = 'Let me think...\nHere is my analysis.\n```json\n{"result": "ok"}\n```'
        scratchpad, block = _split_scratchpad(content, "json")
        assert "Let me think" in scratchpad
        assert block == '{"result": "ok"}'

    def test_split_scratchpad_no_block(self):
        content = "Just some text, no code block."
        scratchpad, block = _split_scratchpad(content, "json")
        assert scratchpad == ""
        assert block == "Just some text, no code block."
