"""Tests for OpenRouter backend initialization."""

from unittest.mock import patch

import pytest


class TestLLMInteractionOpenRouterInit:
    def test_openrouter_init_with_api_key_kwarg(self, tmp_path):
        """OpenRouter backend initializes with api_key kwarg."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "openai/gpt-4"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="openrouter",
                    api_key="sk-or-test-key",
                    model="openai/gpt-4",
                )

        assert llm.model == "openai/gpt-4"
        mock_aoai.assert_called_once_with(
            api_key="sk-or-test-key",
            base_url="https://openrouter.ai/api/v1",
        )

    def test_openrouter_init_with_env_var(self, tmp_path):
        """OpenRouter backend reads API key from LLM_INTERACTION_API_KEY."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "sk-or-env-key",
            "LLM_INTERACTION_MODEL": "anthropic/claude-3-opus",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="openrouter",
                )

        assert llm.model == "anthropic/claude-3-opus"
        mock_aoai.assert_called_once_with(
            api_key="sk-or-env-key",
            base_url="https://openrouter.ai/api/v1",
        )

    def test_openrouter_missing_api_key_raises(self, tmp_path):
        """Missing API key raises ValueError with clear message."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "openai/gpt-4"}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="openrouter",
                )

    def test_unknown_backend_raises(self, tmp_path):
        """Unknown backend string raises ValueError."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with pytest.raises(ValueError, match="Unknown backend.*openrouter"):
                LLMInteraction(prompt_dir=tmp_path, backend="gcp")
