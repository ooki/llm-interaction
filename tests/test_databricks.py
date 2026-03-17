"""Tests for Databricks backend: client construction, token resolution, and error handling."""

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from llm_interaction.client import _resolve_databricks_token


class TestResolveDatabricksToken:
    def test_pat_token(self):
        """Static PAT token is returned directly."""
        w = MagicMock()
        w.config.token = "dapi-abc123"
        assert _resolve_databricks_token(w) == "dapi-abc123"

    def test_cli_oauth_token(self):
        """CLI OAuth: config.token is None, authenticate() returns headers."""
        w = MagicMock()
        w.config.token = None
        w.config.authenticate.return_value = {
            "Authorization": "Bearer eyJ-cli-oauth-token"
        }
        assert _resolve_databricks_token(w) == "eyJ-cli-oauth-token"

    def test_auth_failure_raises(self):
        """No token from either path raises RuntimeError with hint."""
        w = MagicMock()
        w.config.token = None
        w.config.authenticate.return_value = {}
        w.config.host = "https://my-workspace.azuredatabricks.net"
        w.config.auth_type = "databricks-cli"

        with pytest.raises(RuntimeError, match="databricks auth login"):
            _resolve_databricks_token(w)


def _make_databricks_env(tmp_path, ws_mock, extra_env=None):
    """Helper to create LLMInteraction with mocked Databricks backend."""
    import sys

    mock_module = MagicMock()
    mock_module.WorkspaceClient = MagicMock(return_value=ws_mock)

    env = {"LLM_INTERACTION_MODEL": "my-model"}
    if extra_env:
        env.update(extra_env)

    from llm_interaction import LLMInteraction

    with patch.dict("os.environ", env, clear=True):
        with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
            with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                llm = LLMInteraction(prompt_dir=tmp_path, backend="databricks", **{
                    k: v for k, v in (extra_env or {}).items()
                    if k == "databricks_host"
                })
    return llm, mock_module, mock_aoai


class TestLLMInteractionDatabricksInit:
    def _make_workspace_mock(self, token=None, auth_type="databricks-cli"):
        w = MagicMock()
        w.config.host = "https://my-workspace.azuredatabricks.net"
        w.config.token = token
        w.config.auth_type = auth_type
        w.config.authenticate.return_value = {
            "Authorization": "Bearer eyJ-test-token"
        }
        return w

    def test_databricks_init_with_host_kwarg(self, tmp_path):
        """Databricks backend initializes with host kwarg."""
        import sys
        from llm_interaction import LLMInteraction

        w = self._make_workspace_mock()
        mock_module = MagicMock()
        mock_module.WorkspaceClient = MagicMock(return_value=w)

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "my-model"}, clear=True):
            with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
                with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                    llm = LLMInteraction(
                        prompt_dir=tmp_path,
                        backend="databricks",
                        databricks_host="https://my-workspace.azuredatabricks.net",
                        model="my-model",
                    )

        assert llm.model == "my-model"
        assert llm._workspace_client is not None
        mock_module.WorkspaceClient.assert_called_once_with(
            host="https://my-workspace.azuredatabricks.net"
        )
        mock_aoai.assert_called_once_with(
            api_key="eyJ-test-token",
            base_url="https://my-workspace.azuredatabricks.net/serving-endpoints",
        )

    def test_databricks_init_with_env_var(self, tmp_path):
        """Databricks backend reads host from LLM_INTERACTION_DATABRICKS_HOST."""
        import sys
        from llm_interaction import LLMInteraction

        w = MagicMock()
        w.config.host = "https://env-workspace.azuredatabricks.net"
        w.config.token = "dapi-pat-token"
        w.config.auth_type = "pat"

        mock_module = MagicMock()
        mock_module.WorkspaceClient = MagicMock(return_value=w)

        env = {
            "LLM_INTERACTION_DATABRICKS_HOST": "https://env-workspace.azuredatabricks.net",
            "LLM_INTERACTION_MODEL": "my-model",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
                with patch("llm_interaction.client.AsyncOpenAI"):
                    llm = LLMInteraction(
                        prompt_dir=tmp_path,
                        backend="databricks",
                    )

        assert llm.model == "my-model"
        mock_module.WorkspaceClient.assert_called_once_with(
            host="https://env-workspace.azuredatabricks.net"
        )

    def test_databricks_init_pat_token(self, tmp_path):
        """PAT token path works (config.token is set)."""
        import sys
        from llm_interaction import LLMInteraction

        w = self._make_workspace_mock(token="dapi-static-pat")
        mock_module = MagicMock()
        mock_module.WorkspaceClient = MagicMock(return_value=w)

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
                with patch("llm_interaction.client.AsyncOpenAI") as mock_aoai:
                    llm = LLMInteraction(
                        prompt_dir=tmp_path,
                        backend="databricks",
                        databricks_host="https://ws.azuredatabricks.net",
                    )

        mock_aoai.assert_called_once_with(
            api_key="dapi-static-pat",
            base_url="https://my-workspace.azuredatabricks.net/serving-endpoints",
        )

    def test_databricks_missing_sdk_raises(self, tmp_path):
        """Missing databricks-sdk gives clear install hint."""
        import sys
        from llm_interaction import LLMInteraction

        # Remove databricks.sdk from modules to simulate it not being installed
        hidden = {}
        for key in list(sys.modules.keys()):
            if key.startswith("databricks"):
                hidden[key] = sys.modules.pop(key)

        try:
            with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
                with patch("builtins.__import__", side_effect=_import_without_databricks):
                    with pytest.raises(ImportError, match="pip install llm-interaction\\[databricks\\]"):
                        LLMInteraction(prompt_dir=tmp_path, backend="databricks")
        finally:
            sys.modules.update(hidden)

    def test_unknown_backend_raises(self, tmp_path):
        """Unknown backend string raises ValueError."""
        from llm_interaction import LLMInteraction

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with pytest.raises(ValueError, match="Unknown backend"):
                LLMInteraction(prompt_dir=tmp_path, backend="gcp")


class TestDatabricksTokenRefresh:
    def test_refresh_updates_client_api_key(self, tmp_path):
        """_refresh_databricks_token updates the client's api_key."""
        import sys
        from llm_interaction import LLMInteraction

        w = MagicMock()
        w.config.host = "https://ws.azuredatabricks.net"
        w.config.token = None
        w.config.auth_type = "databricks-cli"
        w.config.authenticate.return_value = {
            "Authorization": "Bearer initial-token"
        }

        mock_module = MagicMock()
        mock_module.WorkspaceClient = MagicMock(return_value=w)

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
                with patch("llm_interaction.client.AsyncOpenAI") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client_cls.return_value = mock_client
                    llm = LLMInteraction(prompt_dir=tmp_path, backend="databricks")

        # Now simulate token refresh with a new token
        w.config.authenticate.return_value = {
            "Authorization": "Bearer refreshed-token"
        }
        llm._refresh_databricks_token()
        assert mock_client.api_key == "refreshed-token"

    def test_refresh_noop_for_azure(self, tmp_path):
        """_refresh_databricks_token is a no-op for Azure backend."""
        from llm_interaction import LLMInteraction

        with patch.dict(
            "os.environ",
            {
                "LLM_INTERACTION_API_KEY": "k",
                "LLM_INTERACTION_ENDPOINT": "https://e.openai.azure.com",
                "LLM_INTERACTION_MODEL": "m",
            },
            clear=True,
        ):
            llm = LLMInteraction(prompt_dir=tmp_path)

        # Should not raise — workspace_client is None
        llm._refresh_databricks_token()

    def test_refresh_expired_token_raises_with_hint(self, tmp_path):
        """Expired token raises RuntimeError with re-auth hint."""
        import sys
        from llm_interaction import LLMInteraction

        w = MagicMock()
        w.config.host = "https://ws.azuredatabricks.net"
        w.config.token = "initial"
        w.config.auth_type = "databricks-cli"

        mock_module = MagicMock()
        mock_module.WorkspaceClient = MagicMock(return_value=w)

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            with patch.dict(sys.modules, {"databricks.sdk": mock_module, "databricks": MagicMock()}):
                with patch("llm_interaction.client.AsyncOpenAI"):
                    llm = LLMInteraction(prompt_dir=tmp_path, backend="databricks")

        # Simulate expired token
        w.config.token = None
        w.config.authenticate.return_value = {}

        with pytest.raises(RuntimeError, match="Re-authenticate"):
            llm._refresh_databricks_token()


class TestPrebuiltClient:
    def test_prebuilt_client(self, tmp_path):
        """Pre-built client skips all auth."""
        from llm_interaction import LLMInteraction

        mock_client = MagicMock()

        with patch.dict("os.environ", {"LLM_INTERACTION_MODEL": "m"}, clear=True):
            llm = LLMInteraction(prompt_dir=tmp_path, client=mock_client, model="m")

        assert llm._client is mock_client
        assert llm.model == "m"
        assert llm._workspace_client is None


# Helper for simulating missing databricks-sdk
_original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__


def _import_without_databricks(name, *args, **kwargs):
    if name == "databricks.sdk" or name.startswith("databricks"):
        raise ImportError(f"No module named '{name}'")
    return _original_import(name, *args, **kwargs)
