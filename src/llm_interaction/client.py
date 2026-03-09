"""LLMInteraction — Azure OpenAI Responses API client."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable

import jinja2
import json_repair
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI

from llm_interaction.parsing import _extract_function_calls, _extract_text_from_output
from llm_interaction.response import AgentResult, LLMResponse
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
        for ctx_type, ctx_obj in ctx_map.items():
            if required_type is not None and isinstance(ctx_obj, required_type):
                matched = ctx_obj
                break
            elif required_type is None:
                # Unparameterized ToolContext — match first available
                matched = ctx_obj
                break

        if matched is None:
            raise RuntimeError(
                f"Tool '{tool_def.name}' requires ToolContext[{required_type.__name__ if required_type else '?'}] "
                f"but no matching context was provided."
            )
        kwargs[param_name] = ToolContext(matched)
    return kwargs


# ---------------------------------------------------------------------------
# LLMInteraction — main client
# ---------------------------------------------------------------------------


class LLMInteraction:
    """Azure OpenAI Responses API client.

    Args:
        prompt_dir: Directory containing Jinja2 prompt templates.
        max_retries: Number of API retry attempts on transient errors.
        api_key: Override for ``LLM_INTERACTION_API_KEY`` env var.
        endpoint: Override for ``LLM_INTERACTION_ENDPOINT`` env var.
        model: Override for ``LLM_INTERACTION_MODEL`` env var.
    """

    def __init__(
        self,
        prompt_dir: Path,
        max_retries: int = 3,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str | None = None,
    ):
        load_dotenv()

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

        self.model = model or os.getenv("LLM_INTERACTION_MODEL")
        if not self.model:
            raise ValueError(
                "Model required. Set LLM_INTERACTION_MODEL or pass model=."
            )

        base_url = resolved_endpoint.rstrip("/") + "/openai/v1/"

        logger.info(
            "LLMInteraction initialized — endpoint=%s model=%s",
            resolved_endpoint,
            self.model,
        )

        self._client = AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        self._max_retries = max_retries

        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(prompt_dir)),
            autoescape=jinja2.select_autoescape(),
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

    async def _call_api(
        self,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        """Make a single Responses API call with exponential backoff retry."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if tools:
            kwargs["tools"] = tools
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await self._client.responses.create(**kwargs)
            except (
                openai.APIError,
                openai.RateLimitError,
                openai.APIConnectionError,
            ) as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    wait = (2**attempt) + random.uniform(0, 1)
                    logger.warning(
                        "API error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1,
                        self._max_retries,
                        e,
                        wait,
                    )
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"API call failed after {self._max_retries} retries. Last error: {last_exc}"
        )

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

    # -- agent_loop ---------------------------------------------------------

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
