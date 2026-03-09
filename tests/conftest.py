"""Shared fixtures for llm_interaction tests."""

import pytest
from unittest.mock import patch

from llm_interaction import LLMInteraction


@pytest.fixture
def llm(tmp_path):
    """Create an LLMInteraction with mocked env vars and temp templates."""
    (tmp_path / "test_system.jinja").write_text("System: {{ topic }}")
    (tmp_path / "test_user.jinja").write_text("User: {{ question }}")

    with patch.dict(
        "os.environ",
        {
            "LLM_INTERACTION_API_KEY": "test-key",
            "LLM_INTERACTION_ENDPOINT": "https://test.openai.azure.com",
            "LLM_INTERACTION_MODEL": "gpt-4o",
        },
    ):
        yield LLMInteraction(prompt_dir=tmp_path)
