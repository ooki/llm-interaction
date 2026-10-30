"""LLMInteraction — multi-provider LLM client with Azure, Databricks, OpenRouter,
LiteLLM, DeepSeek, and local (llama-server) backends."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jinja2
import json_repair
from openai import AsyncOpenAI, OpenAI

from llm_interaction.backend import (
    Backend,
    LiteLLMBackend,
    ResponsesBackend,
    StatelessResponsesBackend,
)
from llm_interaction.parsing import _extract_function_calls, _extract_text_from_output
from llm_interaction.response import (
    AgentResult,
    LLMResponse,
    StreamAgentLoop,
    StreamTextDelta,
    StreamToolCall,
    StreamToolResult,
)
from llm_interaction.tool import ToolContext, ToolDef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context matching
# ---------------------------------------------------------------------------


def _build_context_map(context: list[Any] | Any | None) -> dict[type, Any]:
    """Build a {type -> object} map from the context argument."""
    if context is None:
        return {}
    if not isinstance(context, list):
        context = [context]

    ctx_map: dict[type, Any] = {}
    for obj in context:
        obj_type = type(obj)
        if obj_type in ctx_map:
            raise ValueError(
                f"Duplicate context type: {obj_type.__name__}. "
                f"Each type may only appear once in the context list."
            )
        ctx_map[obj_type] = obj
    return ctx_map


def _inject_context(
    tool_def: ToolDef,
    llm_args: dict[str, Any],
    ctx_map: dict[type, Any],
) -> dict[str, Any]:
    """Build the full kwargs for a tool function, injecting ToolContext params."""
    kwargs = dict(llm_args)
    for param_name, required_type in tool_def.context_params.items():
        matched = None
        for _ctx_type, ctx_obj in ctx_map.items():
            if required_type is not None and isinstance(ctx_obj, required_type):
                matched = ctx_obj
                break
            elif required_type is None:
                # Unparameterized ToolContext — match first available
                matched = ctx_obj
                break

        if matched is None:
            type_name = required_type.__name__ if required_type else "?"
            raise RuntimeError(
                f"Tool '{tool_def.name}' requires ToolContext[{type_name}] "
                f"but no matching context was provided."
            )
        kwargs[param_name] = ToolContext(matched)
    return kwargs


# ---------------------------------------------------------------------------
# LLMInteraction — main client
# ---------------------------------------------------------------------------


def _run_sync(coro):
    """Run a coroutine synchronously.

    Raises a clear error if an event loop is already running (e.g. inside a
    Jupyter notebook or another async function), pointing users to the async
    variant or to ``nest-asyncio``.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot call sync methods from within a running event loop. "
        "Use the async version instead (e.g. `await agent_loop(...)`), or "
        "install nest-asyncio:\n"
        "  pip install nest-asyncio\n"
        "  import nest_asyncio; nest_asyncio.apply()"
    )


def _resolve_databricks_token(workspace_client: Any) -> str:
    """Extract bearer token from a Databricks WorkspaceClient.

    Handles both PAT auth (``config.token``) and CLI OAuth auth
    (``config.authenticate()``).  The latter triggers the OAuth flow
    and returns headers with the bearer token.
    """
    # Static token (PAT or env-var based)
    if workspace_client.config.token:
        return workspace_client.config.token
    # CLI OAuth / notebook auth — triggers token refresh
    headers = workspace_client.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        host = workspace_client.config.host or "<unknown>"
        raise RuntimeError(
            f"Databricks authentication failed for {host}.\n"
            f"Auth type detected: {workspace_client.config.auth_type}\n"
            f"If running locally, authenticate first:\n"
            f"  databricks auth login --host {host}"
        )
    return token


# ---------------------------------------------------------------------------
# LiteLLM env-var forwarding
# ---------------------------------------------------------------------------

# Maps model prefix -> (api_key env var, api_base env var or None)
_LITELLM_ENV_MAP: dict[str, tuple[str, str | None]] = {
    "databricks": ("DATABRICKS_API_KEY", "DATABRICKS_API_BASE"),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "azure": ("AZURE_API_KEY", "AZURE_API_BASE"),
    "openrouter": ("OPENROUTER_API_KEY", None),
}


def _forward_litellm_env(
    prefix: str,
    api_key: str | None,
    endpoint: str | None,
    fallback_kwargs: dict[str, str],
) -> None:
    """Forward LLM_INTERACTION_* values to provider-specific env vars.

    For known providers, sets the env vars that LiteLLM reads.
    For unknown providers, stores ``api_key``/``api_base`` in
    *fallback_kwargs* which are passed to each ``aresponses()`` call.
    """
    mapping = _LITELLM_ENV_MAP.get(prefix)

    if mapping is not None:
        key_env, base_env = mapping
        if api_key:
            os.environ.setdefault(key_env, api_key)
        if base_env and endpoint:
            base_value = endpoint.rstrip("/")
            if prefix == "databricks" and not base_value.endswith("/serving-endpoints"):
                base_value += "/serving-endpoints"
            os.environ.setdefault(base_env, base_value)
    else:
        # Unknown provider — pass as kwargs to aresponses()
        if api_key:
            fallback_kwargs["api_key"] = api_key
        if endpoint:
            fallback_kwargs["api_base"] = endpoint.rstrip("/")


class LLMInteraction:
    """Multi-provider LLM client with typed tool calling and lazy output parsing.

    **Azure OpenAI** (default)::

        llm = LLMInteraction(prompt_dir=Path("prompts"))

    **Databricks** (on-site or off-site with CLI auth)::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            backend="databricks",
            model="your-serving-endpoint",
        )

    **OpenRouter**::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            backend="openrouter",
            api_key="your-openrouter-key",
            model="openai/gpt-4",
        )

    **LiteLLM** (universal provider support — Anthropic, Databricks Claude, etc.)::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            backend="litellm",
            model="databricks/databricks-claude-opus-4-8",
        )

    **DeepSeek** (stateless Responses API)::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            backend="deepseek",
            model="deepseek-flash",
        )

    **Local** (llama-server or any OpenAI-compat server)::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            backend="local",
            model="my-model",
            endpoint="http://localhost:8080/v1",
        )

    **Pre-built client** (escape hatch)::

        llm = LLMInteraction(
            prompt_dir=Path("prompts"),
            client=my_openai_client,
            model="my-model",
        )

    The backend can also be set via the ``LLM_INTERACTION_BACKEND`` env var
    so you can switch providers without changing code.

    Args:
        prompt_dir: Directory containing Jinja2 prompt templates.
        max_retries: Number of API retry attempts on transient errors.
        backend: ``"azure"`` (default), ``"databricks"``, ``"openrouter"``,
                 ``"litellm"``, ``"deepseek"``, or ``"local"``.
                 Falls back to ``LLM_INTERACTION_BACKEND`` env var.
        api_key: Override for ``LLM_INTERACTION_API_KEY`` env var.
        endpoint: Override for ``LLM_INTERACTION_ENDPOINT`` env var.
        model: Override for ``LLM_INTERACTION_MODEL`` env var.
        databricks_host: Override for ``LLM_INTERACTION_DATABRICKS_HOST`` env var.
        client: Pre-built ``OpenAI`` or ``AsyncOpenAI`` instance (skips all auth).
    """

    def __init__(
        self,
        prompt_dir: Path,
        max_retries: int = 3,
        backend: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
        databricks_host: str | None = None,
        client: AsyncOpenAI | OpenAI | None = None,
    ):
        self.model = model or os.getenv("LLM_INTERACTION_MODEL")
        if not self.model:
            raise ValueError(
                "Model required. Set LLM_INTERACTION_MODEL or pass model=."
            )

        self._max_retries = max_retries
        self._backend = backend or os.getenv("LLM_INTERACTION_BACKEND", "azure")
        self._workspace_client: Any | None = None  # Databricks WorkspaceClient
        self._backend_impl: Backend | None = None
        self.__client: Any | None = None  # backing store for _client property

        if client is not None:
            # Pre-built client — skip all auth, wrap in ResponsesBackend
            self.__client = client
            self._backend_impl = ResponsesBackend(
                client=client, model=self.model, max_retries=max_retries
            )
            logger.info(
                "LLMInteraction initialized — pre-built client, model=%s",
                self.model,
            )

        elif self._backend == "databricks":
            self._init_databricks(databricks_host)

        elif self._backend == "azure":
            self._init_azure(api_key, endpoint)

        elif self._backend == "openrouter":
            self._init_openrouter(api_key)

        elif self._backend == "litellm":
            self._init_litellm(api_key, endpoint)

        elif self._backend == "deepseek":
            self._init_deepseek(api_key)

        elif self._backend == "local":
            self._init_local(endpoint)

        else:
            raise ValueError(
                f"Unknown backend: {self._backend!r}. "
                f"Use 'azure', 'databricks', 'openrouter', 'litellm', 'deepseek', or 'local'."
            )

        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompt_dir)),
            autoescape=jinja2.select_autoescape(),
        )

    # -- _client property (syncs with backend) ------------------------------

    @property
    def _client(self) -> Any:
        return self.__client

    @_client.setter
    def _client(self, value: Any) -> None:
        self.__client = value
        if self._backend_impl is not None and hasattr(self._backend_impl, 'client'):
            self._backend_impl.client = value

    @property
    def last_call(self) -> dict[str, Any] | None:
        """Return debugging details for the most recent API call.

        Returns a dict with ``request`` (the kwargs sent to the provider)
        and ``response`` (the raw SDK response object), or ``None`` if no
        call has been made yet.  Useful for inspecting token usage, model
        info, finish reasons, and the exact payload that was sent.
        """
        backend = self._backend_impl
        if backend is None:
            return None
        request = getattr(backend, "_last_request", None)
        response = getattr(backend, "_last_response", None)
        if request is None and response is None:
            return None
        return {"request": request, "response": response}

    def _init_azure(self, api_key: str | None, endpoint: str | None) -> None:
        """Initialize the Azure OpenAI backend."""
        resolved_key = api_key or os.getenv("LLM_INTERACTION_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required. Set LLM_INTERACTION_API_KEY or pass api_key=."
            )

        resolved_endpoint = endpoint or os.getenv("LLM_INTERACTION_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError(
                "Endpoint required. Set LLM_INTERACTION_ENDPOINT or pass endpoint=."
            )

        base_url = resolved_endpoint.rstrip("/") + "/openai/v1/"
        self._client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self._backend_impl = ResponsesBackend(
            client=self._client, model=self.model, max_retries=self._max_retries
        )

        logger.info(
            "LLMInteraction initialized — azure endpoint=%s model=%s",
            resolved_endpoint,
            self.model,
        )

    def _init_databricks(self, databricks_host: str | None) -> None:
        """Initialize the Databricks backend via WorkspaceClient."""
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as err:
            raise ImportError(
                "Databricks backend requires the databricks-sdk package.\n"
                "Install it with: pip install llm-interaction[databricks]"
            ) from err

        resolved_host = databricks_host or os.getenv("LLM_INTERACTION_DATABRICKS_HOST")
        kwargs: dict[str, str] = {}
        if resolved_host:
            kwargs["host"] = resolved_host

        try:
            self._workspace_client = WorkspaceClient(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Databricks WorkspaceClient.\n"
                f"Host: {resolved_host or '(auto-detect)'}\n"
                f"If running locally, authenticate first:\n"
                f"  databricks auth login --host {resolved_host or '<your-host>'}\n"
                f"Error: {e}"
            ) from e

        resolved_host = self._workspace_client.config.host
        token = _resolve_databricks_token(self._workspace_client)
        self._client = AsyncOpenAI(
            api_key=token,
            base_url=f"{resolved_host}/serving-endpoints",
        )
        self._backend_impl = ResponsesBackend(
            client=self._client,
            model=self.model,
            max_retries=self._max_retries,
            token_refresher=self._refresh_databricks_token,
        )

        logger.info(
            "LLMInteraction initialized — databricks host=%s model=%s auth=%s",
            resolved_host,
            self.model,
            self._workspace_client.config.auth_type,
        )

    def _init_openrouter(self, api_key: str | None) -> None:
        """Initialize the OpenRouter backend."""
        resolved_key = api_key or os.getenv("LLM_INTERACTION_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required. Set LLM_INTERACTION_API_KEY or pass api_key=."
            )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self._backend_impl = ResponsesBackend(
            client=self._client, model=self.model, max_retries=self._max_retries
        )

        logger.info(
            "LLMInteraction initialized — openrouter model=%s",
            self.model,
        )

    def _init_litellm(self, api_key: str | None, endpoint: str | None) -> None:
        """Initialize the LiteLLM backend.

        Forwards ``LLM_INTERACTION_*`` env vars to the provider-specific
        env vars that LiteLLM expects, based on the model prefix.
        """
        try:
            import litellm as _litellm
        except ImportError as err:
            raise ImportError(
                "LiteLLM backend requires the litellm package.\n"
                "Install it with: pip install llm-interaction[litellm]"
            ) from err

        self._litellm = _litellm
        self._litellm_kwargs: dict[str, str] = {}

        resolved_key = api_key or os.getenv("LLM_INTERACTION_API_KEY")
        resolved_endpoint = endpoint or os.getenv("LLM_INTERACTION_ENDPOINT")

        prefix = self.model.split("/")[0] if "/" in self.model else ""
        _forward_litellm_env(prefix, resolved_key, resolved_endpoint,
                             self._litellm_kwargs)

        self._backend_impl = LiteLLMBackend(
            litellm_module=_litellm,
            model=self.model,
            max_retries=self._max_retries,
            extra_kwargs=self._litellm_kwargs,
        )

        logger.info(
            "LLMInteraction initialized — litellm provider=%s model=%s",
            prefix or "(auto)",
            self.model,
        )

    def _init_deepseek(self, api_key: str | None) -> None:
        """Initialize the DeepSeek backend (stateless Responses API).

        DeepSeek ignores ``previous_response_id``, so conversation history is
        kept client-side and resent on every turn.
        """
        resolved_key = api_key or os.getenv("LLM_INTERACTION_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key required. Set LLM_INTERACTION_API_KEY or pass api_key=."
            )

        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url="https://api.deepseek.com",
        )
        self._backend_impl = StatelessResponsesBackend(
            client=self._client, model=self.model, max_retries=self._max_retries
        )

        logger.info(
            "LLMInteraction initialized — deepseek model=%s",
            self.model,
        )

    def _init_local(self, endpoint: str | None) -> None:
        """Initialize a local backend (llama-server or any OpenAI-compat server).

        Uses the Responses API (``/v1/responses``) which llama-server supports
        natively.
        """
        resolved_endpoint = endpoint or os.getenv(
            "LLM_INTERACTION_ENDPOINT", "http://localhost:8080/v1"
        )

        self._client = AsyncOpenAI(
            api_key="no-key",
            base_url=resolved_endpoint,
        )
        self._backend_impl = ResponsesBackend(
            client=self._client, model=self.model, max_retries=self._max_retries
        )

        logger.info(
            "LLMInteraction initialized — local endpoint=%s model=%s",
            resolved_endpoint,
            self.model,
        )

    # -- Jinja rendering ----------------------------------------------------

    def render(self, template_name: str, variables: dict[str, Any]) -> str:
        """Render a single Jinja template."""
        try:
            tmpl = self._jinja_env.get_template(template_name)
            return tmpl.render(**variables)
        except jinja2.exceptions.TemplateError as e:
            raise ValueError(f"Template rendering error: {e}") from e

    def _render_prompt(
        self, prompt_name: str, variables: dict[str, Any]
    ) -> dict[str, str]:
        """Render ``{name}_system.jinja`` and ``{name}_user.jinja``."""
        try:
            system = self._jinja_env.get_template(
                f"{prompt_name}_system.jinja"
            ).render(**variables)
            user = self._jinja_env.get_template(
                f"{prompt_name}_user.jinja"
            ).render(**variables)
            return {"system": system, "user": user}
        except jinja2.exceptions.TemplateError as e:
            raise ValueError(f"Template rendering error: {e}") from e

    # -- Low-level API call with retry --------------------------------------

    def _refresh_databricks_token(self) -> None:
        """Re-resolve the Databricks token and update the client.

        CLI OAuth tokens expire, so we refresh before each API call.
        """
        if self._workspace_client is None:
            return
        try:
            token = _resolve_databricks_token(self._workspace_client)
            self._client.api_key = token
        except RuntimeError as err:
            host = self._workspace_client.config.host or "<unknown>"
            raise RuntimeError(
                f"Databricks token refresh failed for {host}.\n"
                f"Your session may have expired. Re-authenticate with:\n"
                f"  databricks auth login --host {host}"
            ) from err

    async def _call_api(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        """Delegate to the backend implementation."""
        return await self._backend_impl.call(
            input_items, tools, previous_response_id
        )

    async def _call_api_litellm(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        """Backward-compat alias — delegates to ``_call_api``."""
        return await self._call_api(input_items, tools, previous_response_id)

    async def _stream_api(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ):
        """Delegate streaming to the backend implementation."""

        async for event in self._backend_impl.stream(
            input_items, tools, previous_response_id
        ):
            yield event

    # -- query --------------------------------------------------------------

    async def query(
        self,
        system: str,
        user: str,
        tools: list[ToolDef] | None = None,
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 20,
    ) -> LLMResponse:
        """Send a query to the LLM.

        Args:
            system: System prompt.
            user: User prompt.
            tools: Optional list of @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Max tool call rounds before returning.

        Returns:
            LLMResponse with lazy parsing methods.
        """
        input_items = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        tool_schemas = [t.schema for t in tools] if tools else None
        tool_map = {t.name: t for t in tools} if tools else {}
        ctx_map = _build_context_map(context)
        previous_response_id = None

        for _ in range(max_tool_calls):
            response = await self._call_api(
                input_items, tools=tool_schemas, previous_response_id=previous_response_id
            )

            function_calls = _extract_function_calls(response.output)
            if not function_calls:
                break

            if not tool_map:
                raise RuntimeError(
                    "LLM returned tool calls but no tools were provided."
                )

            input_items = []
            for fc in function_calls:
                tool_def = tool_map.get(fc["name"])
                if tool_def is None:
                    raise RuntimeError(
                        f"LLM called unknown tool '{fc['name']}'"
                    )

                try:
                    llm_args = json.loads(fc["arguments"])
                except json.JSONDecodeError:
                    llm_args = json.loads(
                        json_repair.repair_json(fc["arguments"])
                    )

                full_kwargs = _inject_context(tool_def, llm_args, ctx_map)
                result = tool_def.function(**full_kwargs)
                if asyncio.iscoroutine(result):
                    result = await result

                result_str = (
                    json.dumps(result) if not isinstance(result, str) else result
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": result_str,
                    }
                )

            previous_response_id = response.id

        content = _extract_text_from_output(response.output)
        return LLMResponse(
            text=content, response_id=response.id, _client=self
        )

    # -- query_template -----------------------------------------------------

    async def query_template(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        tools: list[ToolDef] | None = None,
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 20,
    ) -> LLMResponse:
        """Render Jinja templates and send a query.

        Looks for ``{prompt_name}_system.jinja`` and
        ``{prompt_name}_user.jinja`` in the prompt directory.

        Args:
            prompt_name: Template name prefix.
            variables: Variables for template rendering.
            tools: Optional list of @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Max tool call rounds before returning.

        Returns:
            LLMResponse with lazy parsing methods.
        """
        prompts = self._render_prompt(prompt_name, variables)
        return await self.query(
            system=prompts["system"],
            user=prompts["user"],
            tools=tools,
            context=context,
            max_tool_calls=max_tool_calls,
        )

    # -- sync wrappers ------------------------------------------------------

    def sync_query(
        self,
        system: str,
        user: str,
        tools: list[ToolDef] | None = None,
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 20,
    ) -> LLMResponse:
        """Synchronous wrapper for query(). Useful in notebooks.

        Args:
            system: System prompt.
            user: User prompt.
            tools: Optional list of @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Max tool call rounds before returning.

        Returns:
            LLMResponse with lazy parsing methods.
        """
        return _run_sync(
            self.query(system, user, tools, context, max_tool_calls)
        )

    def sync_query_template(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        tools: list[ToolDef] | None = None,
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 20,
    ) -> LLMResponse:
        """Synchronous wrapper for query_template(). Useful in notebooks.

        Looks for ``{prompt_name}_system.jinja`` and
        ``{prompt_name}_user.jinja`` in the prompt directory.

        Args:
            prompt_name: Template name prefix.
            variables: Variables for template rendering.
            tools: Optional list of @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Max tool call rounds before returning.

        Returns:
            LLMResponse with lazy parsing methods.
        """
        return _run_sync(
            self.query_template(prompt_name, variables, tools, context, max_tool_calls)
        )

    def sync_agent_loop(
        self,
        system: str,
        user: str,
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for agent_loop(). Useful in notebooks.

        The loop continues until:
        - The LLM responds with content (no tool calls)
        - A tool marked ``stop=True`` is called
        - ``max_tool_calls`` is reached

        Args:
            system: System prompt.
            user: User prompt.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            AgentResult with tool call count, stop reason, and response ID.
        """
        return _run_sync(
            self.agent_loop(
                system, user, tools, context, max_tool_calls, on_tool_call, previous_response_id
            )
        )

    def sync_agent_loop_template(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> AgentResult:
        """Synchronous wrapper for agent_loop_template(). Useful in notebooks.

        Looks for ``{prompt_name}_system.jinja`` and
        ``{prompt_name}_user.jinja`` in the prompt directory.

        The loop continues until:
        - The LLM responds with content (no tool calls)
        - A tool marked ``stop=True`` is called
        - ``max_tool_calls`` is reached

        Args:
            prompt_name: Template name prefix.
            variables: Variables for template rendering.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            AgentResult with tool call count, stop reason, and response ID.
        """
        return _run_sync(
            self.agent_loop_template(
                prompt_name, variables, tools, context, max_tool_calls,
                on_tool_call, previous_response_id,
            )
        )

    # -- agent_loop ---------------------------------------------------------

    async def agent_loop_template(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> AgentResult:
        """Render Jinja templates and run an agentic tool-calling loop.

        Looks for ``{prompt_name}_system.jinja`` and
        ``{prompt_name}_user.jinja`` in the prompt directory.

        The loop continues until:
        - The LLM responds with content (no tool calls)
        - A tool marked ``stop=True`` is called
        - ``max_tool_calls`` is reached

        Args:
            prompt_name: Template name prefix.
            variables: Variables for template rendering.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            AgentResult with tool call count, stop reason, and response ID.
        """
        prompts = self._render_prompt(prompt_name, variables)
        return await self.agent_loop(
            system=prompts["system"],
            user=prompts["user"],
            tools=tools,
            context=context,
            max_tool_calls=max_tool_calls,
            on_tool_call=on_tool_call,
            previous_response_id=previous_response_id,
        )

    async def agent_loop(
        self,
        system: str,
        user: str,
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> AgentResult:
        """Run an agentic tool-calling loop.

        The loop continues until:
        - The LLM responds with content (no tool calls)
        - A tool marked ``stop=True`` is called
        - ``max_tool_calls`` is reached

        Args:
            system: System prompt.
            user: User prompt.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            AgentResult with tool call count, stop reason, and response ID.
        """
        tool_schemas = [t.schema for t in tools]
        tool_map = {t.name: t for t in tools}
        ctx_map = _build_context_map(context)
        current_response_id = previous_response_id
        tool_call_count = 0

        # Build initial input
        if current_response_id is None:
            input_items: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            input_items = [{"role": "user", "content": user}]

        while tool_call_count < max_tool_calls:
            response = await self._call_api(
                input_items,
                tools=tool_schemas,
                previous_response_id=current_response_id,
            )
            current_response_id = response.id

            function_calls = _extract_function_calls(response.output)
            if not function_calls:
                content = _extract_text_from_output(response.output)
                return AgentResult(
                    tool_call_count=tool_call_count,
                    stop_reason="content",
                    final_content=content,
                    response_id=current_response_id,
                )

            input_items = []
            for fc in function_calls:
                if tool_call_count >= max_tool_calls:
                    return AgentResult(
                        tool_call_count=tool_call_count,
                        stop_reason="max_calls",
                        final_content=None,
                        response_id=current_response_id,
                    )

                tool_def = tool_map.get(fc["name"])
                if tool_def is None:
                    raise RuntimeError(
                        f"LLM called unknown tool '{fc['name']}'"
                    )

                try:
                    llm_args = json.loads(fc["arguments"]) if fc["arguments"] else {}
                except json.JSONDecodeError:
                    llm_args = json.loads(
                        json_repair.repair_json(fc["arguments"])
                    )

                full_kwargs = _inject_context(tool_def, llm_args, ctx_map)

                try:
                    result = tool_def.function(**full_kwargs)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as e:
                    logger.error("Tool '%s' raised: %s", tool_def.name, e)
                    result = {"error": str(e)}

                if on_tool_call:
                    on_tool_call(tool_call_count, fc["name"], llm_args, result)

                result_str = (
                    json.dumps(result) if not isinstance(result, str) else result
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": result_str,
                    }
                )

                tool_call_count += 1

                if tool_def.is_stop:
                    return AgentResult(
                        tool_call_count=tool_call_count,
                        stop_reason="stop_tool",
                        final_content=None,
                        response_id=current_response_id,
                    )

        return AgentResult(
            tool_call_count=tool_call_count,
            stop_reason="max_calls",
            final_content=None,
            response_id=current_response_id,
        )

    # -- streaming agent loop -------------------------------------------------

    def agent_loop_streaming(
        self,
        system: str,
        user: str,
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> StreamAgentLoop:
        """Streaming version of :meth:`agent_loop`.

        Returns a :class:`StreamAgentLoop` that yields :class:`StreamEvent`
        items as they arrive in real time.  After the stream is exhausted,
        ``.result`` provides the final :class:`AgentResult`, or call
        ``await .collect()`` to drain silently.

        Args:
            system: System prompt.
            user: User prompt.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            :class:`StreamAgentLoop` — async-iterable with ``.result``
            and ``.collect()``.
        """
        stream = StreamAgentLoop()
        stream._bind(self._streaming_agent_loop(
            stream,
            system=system,
            user=user,
            tools=tools,
            context=context,
            max_tool_calls=max_tool_calls,
            on_tool_call=on_tool_call,
            previous_response_id=previous_response_id,
        ))
        return stream

    def agent_loop_streaming_template(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        tools: list[ToolDef],
        context: list[Any] | Any | None = None,
        max_tool_calls: int = 50,
        on_tool_call: Callable[[int, str, dict, Any], None] | None = None,
        previous_response_id: str | None = None,
    ) -> StreamAgentLoop:
        """Render Jinja templates and run a streaming agent loop.

        Looks for ``{prompt_name}_system.jinja`` and
        ``{prompt_name}_user.jinja`` in the prompt directory.

        Args:
            prompt_name: Template name prefix.
            variables: Variables for template rendering.
            tools: @tool-decorated functions.
            context: Objects to inject into ToolContext parameters.
            max_tool_calls: Maximum total tool calls.
            on_tool_call: Optional callback(index, name, args, result).
            previous_response_id: Continue from a previous response.

        Returns:
            :class:`StreamAgentLoop` — async-iterable with ``.result``
            and ``.collect()``.
        """
        prompts = self._render_prompt(prompt_name, variables)
        return self.agent_loop_streaming(
            system=prompts["system"],
            user=prompts["user"],
            tools=tools,
            context=context,
            max_tool_calls=max_tool_calls,
            on_tool_call=on_tool_call,
            previous_response_id=previous_response_id,
        )

    async def _streaming_agent_loop(
        self,
        stream: StreamAgentLoop,
        *,
        system: str,
        user: str,
        tools: list[ToolDef],
        context: list[Any] | Any | None,
        max_tool_calls: int,
        on_tool_call: Callable[[int, str, dict, Any], None] | None,
        previous_response_id: str | None,
    ):
        from llm_interaction.backend import (
            _StreamResponseDone,
            _StreamTextDelta,
            _StreamToolCallDone,
        )

        tool_schemas = [t.schema for t in tools]
        tool_map = {t.name: t for t in tools}
        ctx_map = _build_context_map(context)
        current_response_id = previous_response_id
        tool_call_count = 0

        if current_response_id is None:
            input_items: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            input_items = [{"role": "user", "content": user}]

        while tool_call_count < max_tool_calls:
            had_tool_calls = False
            accumulated_text: list[str] = []
            outputs: list[dict[str, Any]] = []

            async for backend_event in self._stream_api(
                input_items,
                tools=tool_schemas,
                previous_response_id=current_response_id,
            ):
                if isinstance(backend_event, _StreamTextDelta):
                    accumulated_text.append(backend_event.text)
                    yield StreamTextDelta(text=backend_event.text)

                elif isinstance(backend_event, _StreamToolCallDone):
                    had_tool_calls = True
                    try:
                        llm_args = json.loads(
                            backend_event.arguments or "{}"
                        )
                    except json.JSONDecodeError:
                        llm_args = json.loads(
                            json_repair.repair_json(backend_event.arguments)
                        )

                    yield StreamToolCall(
                        call_id=backend_event.call_id,
                        name=backend_event.name,
                        arguments=llm_args,
                    )

                    tool_def = tool_map.get(backend_event.name)
                    if tool_def is None:
                        raise RuntimeError(
                            f"LLM called unknown tool '{backend_event.name}'"
                        )

                    full_kwargs = _inject_context(tool_def, llm_args, ctx_map)

                    try:
                        result = tool_def.function(**full_kwargs)
                        if asyncio.iscoroutine(result):
                            result = await result
                    except Exception as e:
                        logger.error(
                            "Tool '%s' raised: %s", tool_def.name, e
                        )
                        result = {"error": str(e)}

                    if on_tool_call:
                        on_tool_call(
                            tool_call_count,
                            backend_event.name,
                            llm_args,
                            result,
                        )

                    yield StreamToolResult(
                        call_id=backend_event.call_id,
                        name=backend_event.name,
                        result=result,
                    )

                    result_str = (
                        json.dumps(result)
                        if not isinstance(result, str)
                        else result
                    )
                    outputs.append({
                        "type": "function_call_output",
                        "call_id": backend_event.call_id,
                        "output": result_str,
                    })

                    tool_call_count += 1

                    if tool_call_count >= max_tool_calls:
                        stream._result = AgentResult(
                            tool_call_count=tool_call_count,
                            stop_reason="max_calls",
                            final_content=None,
                            response_id=current_response_id or "",
                        )
                        stream._drained = True
                        return

                    if tool_def.is_stop:
                        stream._result = AgentResult(
                            tool_call_count=tool_call_count,
                            stop_reason="stop_tool",
                            final_content=None,
                            response_id=current_response_id or "",
                        )
                        stream._drained = True
                        return

                elif isinstance(backend_event, _StreamResponseDone):
                    current_response_id = backend_event.response_id

            if not had_tool_calls:
                final_text = "".join(accumulated_text)
                stream._result = AgentResult(
                    tool_call_count=tool_call_count,
                    stop_reason="content",
                    final_content=final_text,
                    response_id=current_response_id or "",
                )
                stream._drained = True
                return

            input_items = outputs

        stream._result = AgentResult(
            tool_call_count=tool_call_count,
            stop_reason="max_calls",
            final_content=None,
            response_id=current_response_id or "",
        )
        stream._drained = True
