"""Tests for DeepSeek backend initialization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction.backend import ChatCompletionsBackend


class TestLLMInteractionDeepseekInit:
    def test_deepseek_init_with_api_key_kwarg(self, tmp_path):
        """DeepSeek backend initializes with api_key kwarg."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "deepseek-chat"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="deepseek",
                    api_key="sk-ds-test-key",
                    model="deepseek-chat",
                )

        assert llm.model == "deepseek-chat"
        assert llm._backend == "deepseek"
        assert isinstance(llm._backend_impl, ChatCompletionsBackend)
        mock_aoai.assert_called_once_with(
            api_key="sk-ds-test-key",
            base_url="https://api.deepseek.com",
        )

    def test_deepseek_init_with_env_var(self, tmp_path):
        """DeepSeek backend reads API key from LLM_INTERACTION_API_KEY."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "sk-ds-env-key",
            "LLM_INTERACTION_MODEL": "deepseek-chat",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="deepseek",
                )

        assert llm.model == "deepseek-chat"
        mock_aoai.assert_called_once_with(
            api_key="sk-ds-env-key",
            base_url="https://api.deepseek.com",
        )

    def test_deepseek_missing_api_key_raises(self, tmp_path):
        """Missing API key raises ValueError with clear message."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "deepseek-chat"}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="deepseek",
                )

    def test_deepseek_via_env_backend(self, tmp_path):
        """LLM_INTERACTION_BACKEND=deepseek selects the DeepSeek backend."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_BACKEND": "deepseek",
            "LLM_INTERACTION_API_KEY": "sk-ds-key",
            "LLM_INTERACTION_MODEL": "deepseek-chat",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI"):
                llm = LLMInteraction(prompt_dir=tmp_path)

        assert llm._backend == "deepseek"
        assert isinstance(llm._backend_impl, ChatCompletionsBackend)


class TestDeepseekQuery:
    @pytest.mark.asyncio
    async def test_query_uses_chat_completions(self, tmp_path):
        """Query through DeepSeek backend uses Chat Completions API."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "sk-ds-key",
            "LLM_INTERACTION_MODEL": "deepseek-chat",
        }
        with patch.dict("os.environ", env, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path, backend="deepseek")

        # Mock the client's chat completions
        mock_response = MagicMock()
        mock_response.id = "resp-ds"
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="Hello from DeepSeek!", tool_calls=None)
            )
        ]
        llm._client = MagicMock()
        llm._client.chat = MagicMock()
        llm._client.chat.completions = MagicMock()
        llm._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await llm.query(system="Be helpful", user="Hi")

        assert result.text == "Hello from DeepSeek!"
        llm._client.chat.completions.create.assert_called_once()

        # Verify it used Chat Completions format (messages, not input)
        call_kwargs = llm._client.chat.completions.create.call_args[1]
        assert "messages" in call_kwargs
        assert "input" not in call_kwargs
