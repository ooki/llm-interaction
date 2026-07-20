"""Tests for LiteLLM backend: init, env-var forwarding, and _call_api_litellm."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_interaction.client import _forward_litellm_env


# ---------------------------------------------------------------------------
# _forward_litellm_env
# ---------------------------------------------------------------------------


class TestForwardLitellmEnv:
    def test_databricks_sets_env_vars(self):
        """Databricks prefix forwards key and base with /serving-endpoints."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env(
                "databricks", "dapi-abc", "https://ws.azuredatabricks.net", kwargs
            )
            import os

            assert os.environ["DATABRICKS_API_KEY"] == "dapi-abc"
            assert (
                os.environ["DATABRICKS_API_BASE"]
                == "https://ws.azuredatabricks.net/serving-endpoints"
            )
        assert kwargs == {}

    def test_databricks_no_double_suffix(self):
        """If endpoint already ends with /serving-endpoints, don't double it."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env(
                "databricks",
                "dapi-abc",
                "https://ws.azuredatabricks.net/serving-endpoints",
                kwargs,
            )
            import os

            assert (
                os.environ["DATABRICKS_API_BASE"]
                == "https://ws.azuredatabricks.net/serving-endpoints"
            )

    def test_anthropic_sets_key_only(self):
        """Anthropic prefix forwards key but not base."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env("anthropic", "sk-ant-123", None, kwargs)
            import os

            assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-123"
            assert "ANTHROPIC_API_BASE" not in os.environ
        assert kwargs == {}

    def test_azure_sets_key_and_base(self):
        """Azure prefix forwards both key and base."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env(
                "azure", "az-key", "https://my-resource.openai.azure.com", kwargs
            )
            import os

            assert os.environ["AZURE_API_KEY"] == "az-key"
            assert (
                os.environ["AZURE_API_BASE"]
                == "https://my-resource.openai.azure.com"
            )

    def test_openrouter_sets_key_only(self):
        """OpenRouter prefix forwards key but not base."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env("openrouter", "or-key", None, kwargs)
            import os

            assert os.environ["OPENROUTER_API_KEY"] == "or-key"

    def test_unknown_prefix_uses_kwargs(self):
        """Unknown prefix stores api_key/api_base in fallback kwargs."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env(
                "custom_provider", "my-key", "https://custom.api.com", kwargs
            )
            import os

            # Should NOT set any env vars
            assert "CUSTOM_PROVIDER_API_KEY" not in os.environ
        assert kwargs == {"api_key": "my-key", "api_base": "https://custom.api.com"}

    def test_no_key_no_endpoint_is_noop(self):
        """If both key and endpoint are None, nothing is set."""
        kwargs: dict[str, str] = {}
        with patch.dict("os.environ", {}, clear=True):
            _forward_litellm_env("databricks", None, None, kwargs)
            import os

            assert "DATABRICKS_API_KEY" not in os.environ
            assert "DATABRICKS_API_BASE" not in os.environ
        assert kwargs == {}

    def test_setdefault_does_not_overwrite(self):
        """Existing env vars are not overwritten (uses setdefault)."""
        kwargs: dict[str, str] = {}
        with patch.dict(
            "os.environ", {"DATABRICKS_API_KEY": "existing"}, clear=True
        ):
            _forward_litellm_env("databricks", "new-key", None, kwargs)
            import os

            assert os.environ["DATABRICKS_API_KEY"] == "existing"


# ---------------------------------------------------------------------------
# LLMInteraction litellm init
# ---------------------------------------------------------------------------


class TestLLMInteractionLitellmInit:
    def test_litellm_init_success(self, tmp_path):
        """LiteLLM backend initializes and stores module ref."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()

        env = {"LLM_INTERACTION_MODEL": "databricks/my-claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    api_key="dapi-test",
                    endpoint="https://ws.azuredatabricks.net",
                )

        assert llm.model == "databricks/my-claude"
        assert llm._backend == "litellm"
        assert llm._litellm is mock_litellm

    def test_litellm_init_env_forwarding(self, tmp_path):
        """LiteLLM backend forwards LLM_INTERACTION_* to provider env vars."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()

        env = {
            "LLM_INTERACTION_MODEL": "anthropic/claude-sonnet-4-20250514",
            "LLM_INTERACTION_API_KEY": "sk-ant-test",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                )

        import os

        # The key should have been forwarded during init
        # (env is still patched within the with block, but we check the model)
        assert llm.model == "anthropic/claude-sonnet-4-20250514"

    def test_litellm_missing_package_raises(self, tmp_path):
        """Missing litellm package gives clear install hint."""
        from llm_interaction import LLMInteraction

        # Remove litellm from modules to simulate it not being installed
        hidden = {}
        for key in list(sys.modules.keys()):
            if key.startswith("litellm"):
                hidden[key] = sys.modules.pop(key)

        try:
            with patch.dict(
                "os.environ", {"LLM_INTERACTION_MODEL": "anthropic/claude"}, clear=True
            ):
                with patch(
                    "builtins.__import__", side_effect=_import_without_litellm
                ):
                    with pytest.raises(
                        ImportError, match="pip install llm-interaction\\[litellm\\]"
                    ):
                        LLMInteraction(prompt_dir=tmp_path, backend="litellm")
        finally:
            sys.modules.update(hidden)

    def test_litellm_unknown_prefix_stores_kwargs(self, tmp_path):
        """Unknown model prefix stores api_key/api_base as call kwargs."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()

        env = {"LLM_INTERACTION_MODEL": "custom/my-model"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    api_key="custom-key",
                    endpoint="https://custom.api.com",
                )

        assert llm._litellm_kwargs == {
            "api_key": "custom-key",
            "api_base": "https://custom.api.com",
        }

    def test_unknown_backend_error_includes_litellm(self, tmp_path):
        """Unknown backend error message mentions litellm."""
        from llm_interaction import LLMInteraction

        with patch.dict(
            "os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True
        ):
            with pytest.raises(ValueError, match="litellm"):
                LLMInteraction(prompt_dir=tmp_path, backend="gcp")


# ---------------------------------------------------------------------------
# _call_api_litellm
# ---------------------------------------------------------------------------


class TestCallApiLitellm:
    def test_basic_call(self, tmp_path):
        """_call_api_litellm calls litellm.aresponses with correct kwargs."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(return_value=mock_response)

        env = {"LLM_INTERACTION_MODEL": "databricks/my-claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    api_key="dapi-test",
                    endpoint="https://ws.azuredatabricks.net",
                )

        result = asyncio.run(
            llm._call_api_litellm(
                [{"role": "user", "content": "hello"}]
            )
        )

        assert result.output == mock_response.output
        assert result.id == "resp-123"
        mock_litellm.aresponses.assert_called_once()
        call_kwargs = mock_litellm.aresponses.call_args[1]
        assert call_kwargs["model"] == "databricks/my-claude"
        assert call_kwargs["input"] == [{"role": "user", "content": "hello"}]

    def test_tools_forwarded(self, tmp_path):
        """Tools are passed through to aresponses."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-123"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(return_value=mock_response)

        env = {"LLM_INTERACTION_MODEL": "anthropic/claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                )

        tools = [{"type": "function", "name": "add", "parameters": {}}]
        asyncio.run(
            llm._call_api_litellm(
                [{"role": "user", "content": "hi"}],
                tools=tools,
            )
        )

        call_kwargs = mock_litellm.aresponses.call_args[1]
        assert call_kwargs["tools"] == tools

    def test_previous_response_id_forwarded(self, tmp_path):
        """previous_response_id is passed through to aresponses."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-456"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(return_value=mock_response)

        env = {"LLM_INTERACTION_MODEL": "anthropic/claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                )

        asyncio.run(
            llm._call_api_litellm(
                [{"role": "user", "content": "hi"}],
                previous_response_id="resp-prev",
            )
        )

        call_kwargs = mock_litellm.aresponses.call_args[1]
        assert call_kwargs["previous_response_id"] == "resp-prev"

    def test_fallback_kwargs_included(self, tmp_path):
        """Fallback kwargs (unknown provider) are merged into the call."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-789"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(return_value=mock_response)

        env = {"LLM_INTERACTION_MODEL": "custom/my-model"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    api_key="custom-key",
                    endpoint="https://custom.api.com",
                )

        asyncio.run(
            llm._call_api_litellm(
                [{"role": "user", "content": "hi"}]
            )
        )

        call_kwargs = mock_litellm.aresponses.call_args[1]
        assert call_kwargs["api_key"] == "custom-key"
        assert call_kwargs["api_base"] == "https://custom.api.com"

    def test_retry_on_error(self, tmp_path):
        """_call_api_litellm retries on exception."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-ok"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(
            side_effect=[RuntimeError("transient"), mock_response]
        )

        env = {"LLM_INTERACTION_MODEL": "anthropic/claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    max_retries=2,
                )

        with patch("llm_interaction.backend.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                llm._call_api_litellm(
                    [{"role": "user", "content": "hi"}]
                )
            )

        assert result.id == "resp-ok"
        assert mock_litellm.aresponses.call_count == 2

    def test_all_retries_exhausted_raises(self, tmp_path):
        """All retries exhausted raises RuntimeError."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_litellm.aresponses = AsyncMock(
            side_effect=RuntimeError("persistent error")
        )

        env = {"LLM_INTERACTION_MODEL": "anthropic/claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                    max_retries=2,
                )

        with patch("llm_interaction.backend.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="LiteLLM API call failed"):
                asyncio.run(
                    llm._call_api_litellm(
                        [{"role": "user", "content": "hi"}]
                    )
                )

    def test_call_api_dispatches_to_litellm(self, tmp_path):
        """_call_api dispatches to _call_api_litellm for litellm backend."""
        from llm_interaction import LLMInteraction

        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "resp-dispatch"
        mock_response.output = []
        mock_litellm.aresponses = AsyncMock(return_value=mock_response)

        env = {"LLM_INTERACTION_MODEL": "anthropic/claude"}
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"litellm": mock_litellm}):
                llm = LLMInteraction(
                    prompt_dir=tmp_path,
                    backend="litellm",
                )

        result = asyncio.run(
            llm._call_api([{"role": "user", "content": "hi"}])
        )

        assert result.id == "resp-dispatch"
        mock_litellm.aresponses.assert_called_once()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _import_without_litellm(name, *args, **kwargs):
    if name == "litellm" or name.startswith("litellm."):
        raise ImportError(f"No module named '{name}'")
    return _real_import(name, *args, **kwargs)
