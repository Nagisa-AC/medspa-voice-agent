"""Anthropic Messages adapter. Implements Model. No payload logging."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from types import MappingProxyType, TracebackType
from typing import Any, Final

from anthropic import APIError, APIStatusError, AsyncAnthropic

from core.llm import (
    Done,
    FirstTokenTimeout,
    Message,
    ModelEvent,
    Stop,
    StreamStalled,
    TextDelta,
    ToolCall,
    ToolCallParseError,
    ToolSchema,
)

DEFAULT_MODEL: Final = "claude-haiku-5-5"
DEFAULT_EFFORT: Final = "low"
CALL_STARTED: Final = "[call started]"
_EFFORTS: Final = frozenset({"low", "medium", "high"})
_STOP: Final[dict[str, Stop]] = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
}
_GEN_BLOCKS: Final = frozenset({"text", "tool_use"})


class AnthropicStreamError(Exception):
    def __init__(self, status: int | None, request_id: str | None) -> None:
        self.status = status
        self.request_id = request_id
        super().__init__(f"AnthropicStreamError(status={status}, request_id={request_id!r})")

    def __repr__(self) -> str:
        return f"AnthropicStreamError(status={self.status}, request_id={self.request_id!r})"

    def __str__(self) -> str:
        return repr(self)


class AnthropicFormatError(Exception):
    def __init__(self, reason: str, *, messages: int = 0, tools: int = 0) -> None:
        self.reason = reason
        self.messages = messages
        self.tools = tools
        super().__init__(
            f"AnthropicFormatError(reason={reason!r}, messages={messages}, tools={tools})"
        )

    def __repr__(self) -> str:
        return (
            f"AnthropicFormatError(reason={self.reason!r}, "
            f"messages={self.messages}, tools={self.tools})"
        )

    def __str__(self) -> str:
        return repr(self)


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _effort(value: str | None) -> str:
    effort = value if value is not None else os.environ.get("ANTHROPIC_EFFORT", DEFAULT_EFFORT)
    if effort not in _EFFORTS:
        raise ValueError("effort must be low, medium, or high")
    return effort


def to_api_messages(
    messages: Sequence[Message],
) -> tuple[str, list[dict[str, object]]]:
    if not messages or messages[0].role != "system":
        raise AnthropicFormatError("system_first", messages=len(messages))
    if any(message.role == "system" for message in messages[1:]):
        raise AnthropicFormatError("system_later", messages=len(messages))
    api: list[dict[str, object]] = []
    index = 1
    while index < len(messages):
        message = messages[index]
        if message.role == "tool":
            raise AnthropicFormatError("tool_without_assistant", messages=len(messages))
        if message.role == "user":
            _append_user(api, message.content)
            index += 1
            continue
        if message.role != "assistant":
            raise AnthropicFormatError("unknown_role", messages=len(messages))
        blocks: list[dict[str, object]] = []
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        for call in message.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": dict(call.args),
                }
            )
        index += 1
        results: list[dict[str, object]] = []
        while index < len(messages) and messages[index].role == "tool":
            row = messages[index]
            if not row.tool_call_id:
                raise AnthropicFormatError("tool_id", messages=len(messages))
            results.append(
                {"type": "tool_result", "tool_use_id": row.tool_call_id, "content": row.content}
            )
            index += 1
        if (
            not message.tool_calls
            and not results
            and api
            and api[-1]["role"] == "assistant"
            and _text_only(api[-1])
        ):
            _merge_assistant_text(api[-1], message.content)
        elif message.tool_calls and api and api[-1]["role"] == "assistant" and _text_only(api[-1]):
            _fold_tool_use(api[-1], blocks)
        elif blocks:
            api.append({"role": "assistant", "content": blocks})
        elif results:
            raise AnthropicFormatError("tool_without_assistant", messages=len(messages))
        if results:
            api.append({"role": "user", "content": results})
    if not api:
        raise AnthropicFormatError("empty", messages=len(messages))
    if api[0]["role"] != "user":
        api.insert(0, {"role": "user", "content": CALL_STARTED})
    if api[-1]["role"] != "user":
        raise AnthropicFormatError("last_not_user", messages=len(messages))
    return messages[0].content, api


def _text_only(row: Mapping[str, object]) -> bool:
    content = row.get("content")
    if isinstance(content, str):
        return True
    if not isinstance(content, list):
        return False
    return all(isinstance(block, dict) and block.get("type") == "text" for block in content)


def _append_user(api: list[dict[str, object]], text: str) -> None:
    if api and api[-1]["role"] == "user":
        prev = api[-1]["content"]
        if isinstance(prev, str):
            api[-1]["content"] = f"{prev}\n{text}"
            return
        if isinstance(prev, list):
            prev.append({"type": "text", "text": text})
            return
    api.append({"role": "user", "content": text})


def _merge_assistant_text(row: dict[str, object], text: str) -> None:
    if not text:
        return
    content = row["content"]
    if isinstance(content, str):
        row["content"] = f"{content}\n{text}"
        return
    if not isinstance(content, list):
        return
    texts = [
        str(block["text"])
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    texts.append(text)
    row["content"] = [{"type": "text", "text": "\n".join(texts)}]


def _fold_tool_use(row: dict[str, object], blocks: list[dict[str, object]]) -> None:
    content = row["content"]
    if isinstance(content, str):
        prefix: list[dict[str, object]] = [{"type": "text", "text": content}] if content else []
    elif isinstance(content, list):
        prefix = [block for block in content if isinstance(block, dict)]
    else:
        prefix = []
    row["content"] = prefix + blocks


def _event_type(event: object) -> str:
    return str(getattr(event, "type", "") or "")


def _block(event: object) -> object:
    return getattr(event, "content_block", None)


def _block_type(event: object) -> str:
    block = _block(event)
    return str(getattr(block, "type", "") or "") if block is not None else ""


def _is_generation_start(event: object) -> bool:
    return _event_type(event) == "content_block_start" and _block_type(event) in _GEN_BLOCKS


def _api_error(exc: BaseException) -> AnthropicStreamError:
    if isinstance(exc, APIStatusError):
        return AnthropicStreamError(exc.status_code, exc.request_id)
    return AnthropicStreamError(None, None)


class AnthropicLLM:
    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        effort: str | None = None,
        clock: Any | None = None,
    ) -> None:
        env_model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.model = model if model is not None else env_model
        self.effort = _effort(effort)
        if client is None:
            key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY is required")
            client = AsyncAnthropic(api_key=key, max_retries=0)
        self._client: Any = client
        self._clock = clock if clock is not None else time.monotonic
        self._calls = 0
        self._first_token_s: list[float] = []

    @property
    def call_count(self) -> int:
        return self._calls

    def take_first_token_times(self) -> list[float]:
        times = self._first_token_s
        self._first_token_s = []
        return times

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> _AnthropicStream:
        _require_positive("max_output_tokens", max_output_tokens)
        _require_positive("first_token_timeout", first_token_timeout)
        _require_positive("idle_timeout", idle_timeout)
        self._calls += 1
        return _AnthropicStream(
            self,
            messages,
            tools,
            max_output_tokens=int(max_output_tokens),
            first_token_timeout=first_token_timeout,
            idle_timeout=idle_timeout,
        )

    def __repr__(self) -> str:
        return f"AnthropicLLM(model={self.model!r}, effort={self.effort!r}, calls={self._calls})"

    def __str__(self) -> str:
        return repr(self)


class _AnthropicStream:
    def __init__(
        self,
        owner: AnthropicLLM,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> None:
        self._owner = owner
        self._messages = messages
        self._tools = tools
        self._max_tokens = max_output_tokens
        self._first = first_token_timeout
        self._idle = idle_timeout
        self._play = self._run()
        self._raw: Any = None
        self._closed = False

    def __aiter__(self) -> _AnthropicStream:
        return self

    async def __anext__(self) -> ModelEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._play.__anext__()
        except StopAsyncIteration:
            self._closed = True
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._close_raw()
        close = getattr(self._play, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except (GeneratorExit, asyncio.CancelledError, StopAsyncIteration):
            return

    async def __aenter__(self) -> _AnthropicStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _run(self) -> AsyncIterator[ModelEvent]:
        started = self._owner._clock()
        deadline = started + self._first
        stop: Stop = "stop"
        yielded_done = False
        tool: _ToolAcc | None = None
        seen_start = False
        try:
            self._raw = await self._open(deadline)
            iterator = self._raw.__aiter__()
            while True:
                event = await self._next_event(iterator, seen_start, deadline)
                if event is None:
                    break
                if _is_generation_start(event) and not seen_start:
                    seen_start = True
                    self._owner._first_token_s.append(self._owner._clock() - started)
                kind = _event_type(event)
                if kind == "content_block_start" and _block_type(event) == "tool_use":
                    block = _block(event)
                    tool = _ToolAcc(
                        id=str(getattr(block, "id", "") or ""),
                        name=str(getattr(block, "name", "") or ""),
                    )
                    continue
                if kind == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    dtype = str(getattr(delta, "type", "") or "")
                    if dtype == "text_delta":
                        text = str(getattr(delta, "text", "") or "")
                        if text:
                            yield TextDelta(text)
                    elif dtype == "input_json_delta" and tool is not None:
                        tool.chunks.append(str(getattr(delta, "partial_json", "") or ""))
                    continue
                if kind == "content_block_stop" and tool is not None:
                    yield _parse_tool(tool)
                    tool = None
                    continue
                if kind == "message_delta":
                    reason = getattr(getattr(event, "delta", None), "stop_reason", None)
                    if isinstance(reason, str) and reason:
                        stop = _STOP.get(reason, "stop")
                    continue
                if kind == "message_stop":
                    yield Done(stop)
                    yielded_done = True
                    break
            if not yielded_done:
                yield Done(stop)
        finally:
            await self._close_raw()

    async def _open(self, deadline: float) -> Any:
        remaining = deadline - self._owner._clock()
        if remaining <= 0:
            raise FirstTokenTimeout()
        system, api_messages = to_api_messages(self._messages)
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters_json(),
            }
            for tool in self._tools
        ]
        try:
            return await asyncio.wait_for(
                self._owner._client.messages.create(
                    model=self._owner.model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=api_messages,
                    tools=tools,
                    stream=True,
                    thinking={"type": "disabled"},
                    output_config={"effort": self._owner.effort},
                ),
                remaining,
            )
        except TimeoutError:
            raise FirstTokenTimeout() from None
        except APIError as exc:
            raise _api_error(exc) from None

    async def _next_event(
        self, iterator: AsyncIterator[object], seen_start: bool, deadline: float
    ) -> object | None:
        if not seen_start:
            remaining = deadline - self._owner._clock()
            if remaining <= 0:
                raise FirstTokenTimeout()
            timeout: float = remaining
            error: type[FirstTokenTimeout | StreamStalled] = FirstTokenTimeout
        else:
            timeout = self._idle
            error = StreamStalled
        try:
            return await asyncio.wait_for(iterator.__anext__(), timeout)
        except TimeoutError:
            raise error() from None
        except StopAsyncIteration:
            return None
        except APIError as exc:
            raise _api_error(exc) from None

    async def _close_raw(self) -> None:
        raw = self._raw
        self._raw = None
        if raw is None:
            return
        for name in ("aclose", "close"):
            close = getattr(raw, name, None)
            if close is None:
                continue
            try:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
            except (GeneratorExit, asyncio.CancelledError, StopAsyncIteration):
                return
            return


class _ToolAcc:
    def __init__(self, id: str, name: str) -> None:
        self.id = id
        self.name = name
        self.chunks: list[str] = []


def _parse_tool(tool: _ToolAcc) -> ToolCall:
    try:
        loaded = json.loads("".join(tool.chunks)) if tool.chunks else {}
    except json.JSONDecodeError:
        raise ToolCallParseError(tool.id, tool.name) from None
    if not isinstance(loaded, dict):
        raise ToolCallParseError(tool.id, tool.name)
    return ToolCall(tool.id, tool.name, MappingProxyType(dict(loaded)))
