"""Tests for local backend initialization (llama-server)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction.backend import ResponsesBackend


class TestLLMInteractionLocalInit:
    def test_local_init_with_endpoint_kwarg(self, tmp_path):
        """Local backend initializes with endpoint kwarg."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "my-model"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="local",
                    endpoint="http://localhost:9090/v1",
                    model="my-model",
                )

        assert llm.model == "my-model"
        assert llm._backend == "local"
        assert isinstance(llm._backend_impl, ResponsesBackend)
        mock_aoai.assert_called_once_with(
            api_key="no-key",
            base_url="http://localhost:9090/v1",
        )

    def test_local_init_default_endpoint(self, tmp_path):
        """Local backend defaults to http://localhost:8080/v1."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "my-model"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="local",
                )

        mock_aoai.assert_called_once_with(
            api_key="no-key",
            base_url="http://localhost:8080/v1",
        )

    def test_local_init_with_env_endpoint(self, tmp_path):
        """Local backend reads endpoint from LLM_INTERACTION_ENDPOINT."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_ENDPOINT": "http://gpu-server:8080/v1",
            "LLM_INTERACTION_MODEL": "llama3",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="local",
                )

        mock_aoai.assert_called_once_with(
            api_key="no-key",
            base_url="http://gpu-server:8080/v1",
        )

    def test_local_no_api_key_needed(self, tmp_path):
        """Local backend does not require an API key."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI"):
                # Should NOT raise — no API key needed
                llm = LLMInteraction(prompt_dir=tmp_path, backend="local")

        assert llm._backend == "local"

    def test_local_via_env_backend(self, tmp_path):
        """LLM_INTERACTION_BACKEND=local selects the local backend."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_BACKEND": "local",
            "LLM_INTERACTION_MODEL": "my-model",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI"):
                llm = LLMInteraction(prompt_dir=tmp_path)

        assert llm._backend == "local"
        assert isinstance(llm._backend_impl, ResponsesBackend)


class TestBackendEnvVar:
    def test_env_backend_defaults_to_azure(self, tmp_path):
        """Without LLM_INTERACTION_BACKEND, defaults to azure."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_API_KEY": "k",
            "LLM_INTERACTION_ENDPOINT": "https://test.openai.azure.com",
            "LLM_INTERACTION_MODEL": "gpt-4o",
        }
        with patch.dict("os.environ", env, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path)

        assert llm._backend == "azure"

    def test_env_backend_overrides_default(self, tmp_path):
        """LLM_INTERACTION_BACKEND env var overrides the default."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_BACKEND": "local",
            "LLM_INTERACTION_MODEL": "m",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch("llm_interaction.client.AsyncOpenAI"):
                llm = LLMInteraction(prompt_dir=tmp_path)

        assert llm._backend == "local"

    def test_kwarg_backend_overrides_env(self, tmp_path):
        """Explicit backend= kwarg overrides LLM_INTERACTION_BACKEND env var."""
        from llm_interaction import LLMInteraction

        env = {
            "LLM_INTERACTION_BACKEND": "local",
            "LLM_INTERACTION_API_KEY": "k",
            "LLM_INTERACTION_ENDPOINT": "https://test.openai.azure.com",
            "LLM_INTERACTION_MODEL": "m",
        }
        with patch.dict("os.environ", env, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path, backend="azure")

        assert llm._backend == "azure"
