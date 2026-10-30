"""Tests for DeepSeek backend initialization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction.backend import StatelessResponsesBackend


class TestLLMInteractionDeepseekInit:
    def test_deepseek_init_with_api_key_kwarg(self, tmp_path):
        """DeepSeek backend initializes with api_key kwarg."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "deepseek-flash"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="deepseek",
                    api_key="sk-ds-test-key",
                    model="deepseek-flash",
                )

        assert llm.model == "deepseek-flash"
        assert llm._backend == "deepseek"
        assert isinstance(llm._backend_impl, StatelessResponsesBackend)
        mock_aoai.assert_called_once_with(
            api_key="sk-ds-test-key",
            base_url="https://api.deepseek.com",
        )

    def test_deepseek_init_with_env_var(self, tmp_path):
        """DeepSeek backend reads API key from LLM_INTERACTION_API_KEY."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "sk-ds-env-key",
            "LLM_INTERACTION_MODEL": "deepseek-flash",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="deepseek",
                )

        assert llm.model == "deepseek-flash"
        mock_aoai.assert_called_once_with(
            api_key="sk-ds-env-key",
            base_url="https://api.deepseek.com",
        )

    def test_deepseek_missing_api_key_raises(self, tmp_path):
        """Missing API key raises ValueError with clear message."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "deepseek-flash"}, clear=True):
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
            "LLM_INTERACTION_MODEL": "deepseek-flash",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI"):
                llm = LLMInteraction(prompt_dir=tmp_path)

        assert llm._backend == "deepseek"
        assert isinstance(llm._backend_impl, StatelessResponsesBackend)


class TestDeepseekQuery:
    @pytest.mark.asyncio
    async def test_query_uses_responses_api(self, tmp_path):
        """Query through DeepSeek backend uses the Responses API without server state."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "sk-ds-key",
            "LLM_INTERACTION_MODEL": "deepseek-flash",
        }
        with patch.dict("os.environ", env, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path, backend="deepseek")

        mock_response = SimpleNamespace(
            id="resp-ds",
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    content=[SimpleNamespace(type="output_text", text="Hello from DeepSeek!")],
                )
            ],
        )
        llm._client = MagicMock()
        llm._client.responses.create = AsyncMock(return_value=mock_response)

        result = await llm.query(system="Be helpful", user="Hi")

        assert result.text == "Hello from DeepSeek!"
        llm._client.responses.create.assert_called_once()

        call_kwargs = llm._client.responses.create.call_args[1]
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs
        assert "previous_response_id" not in call_kwargs
