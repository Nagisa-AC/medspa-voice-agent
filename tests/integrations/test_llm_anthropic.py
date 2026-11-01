from __future__ import annotations

import ast
import asyncio
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import integrations.llm_anthropic
from core.llm import (
    Done,
    FirstTokenTimeout,
    Message,
    ModelEvent,
    StreamStalled,
    TextDelta,
    ToolCall,
    ToolCallParseError,
    ToolSchema,
)
from core.prompt import tool_result_messages
from integrations.llm_anthropic import (
    CALL_STARTED,
    AnthropicLLM,
    to_api_messages,
)

QUOTE = ToolSchema(
    name="quote_price",
    description="Quote a service",
    parameters={"type": "object", "properties": {"service_text": {"type": "string"}}},
)
SYSTEM = Message("system", "You are the spa assistant.")
USER = Message("user", "how much is lip filler")


def ev(
    type: str,
    *,
    block_type: str | None = None,
    block_id: str = "",
    block_name: str = "",
    delta_type: str | None = None,
    text: str = "",
    partial_json: str = "",
    stop_reason: str | None = None,
) -> SimpleNamespace:
    content_block = None
    if block_type is not None:
        content_block = SimpleNamespace(type=block_type, id=block_id, name=block_name)
    delta = None
    if delta_type is not None or stop_reason is not None:
        delta = SimpleNamespace(type=delta_type, text=text, partial_json=partial_json)
        if stop_reason is not None:
            delta.stop_reason = stop_reason
    return SimpleNamespace(type=type, content_block=content_block, delta=delta)


class FakeStream:
    def __init__(self, events: Sequence[object], delays: dict[int, float] | None = None) -> None:
        self._events = list(events)
        self._index = 0
        self.closed = False
        self.delays = delays or {}

    def __aiter__(self) -> FakeStream:
        return self

    async def __anext__(self) -> object:
        if self.closed or self._index >= len(self._events):
            raise StopAsyncIteration
        delay = self.delays.get(self._index)
        if delay:
            await asyncio.sleep(delay)
        item = self._events[self._index]
        self._index += 1
        return item

    async def aclose(self) -> None:
        self.closed = True


class FakeMessages:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream
        self.kwargs: dict[str, object] | None = None

    async def create(self, **kwargs: object) -> FakeStream:
        self.kwargs = kwargs
        return self.stream


class FakeClient:
    def __init__(self, stream: FakeStream) -> None:
        self.messages = FakeMessages(stream)


def llm_for(events: Sequence[object], delays: dict[int, float] | None = None) -> AnthropicLLM:
    return AnthropicLLM(FakeClient(FakeStream(events, delays)))


async def collect(
    llm: AnthropicLLM,
    messages: list[Message] | None = None,
    tools: list[ToolSchema] | None = None,
    **limits: float | int,
) -> list[ModelEvent]:
    stream = llm.stream(
        messages if messages is not None else [SYSTEM, USER],
        tools if tools is not None else [QUOTE],
        max_output_tokens=int(limits.get("max_output_tokens", 32)),
        first_token_timeout=float(limits.get("first_token_timeout", 5)),
        idle_timeout=float(limits.get("idle_timeout", 5)),
    )
    try:
        return [event async for event in stream]
    finally:
        await stream.aclose()


TEXT_STREAM = (
    ev("message_start"),
    ev("content_block_start", block_type="text"),
    ev("content_block_delta", delta_type="text_delta", text="Hello "),
    ev("content_block_delta", delta_type="text_delta", text="there."),
    ev("content_block_stop"),
    ev("message_delta", stop_reason="end_turn"),
    ev("message_stop"),
)

TOOL_STREAM = (
    ev("message_start"),
    ev("content_block_start", block_type="tool_use", block_id="c1", block_name="quote_price"),
    ev("content_block_delta", delta_type="input_json_delta", partial_json='{"service_text":'),
    ev("content_block_delta", delta_type="input_json_delta", partial_json=' "lip filler"}'),
    ev("content_block_stop"),
    ev("message_delta", stop_reason="tool_use"),
    ev("message_stop"),
)


async def test_text_deltas_then_done() -> None:
    events = await collect(llm_for(TEXT_STREAM))
    assert [type(event) for event in events] == [TextDelta, TextDelta, Done]
    assert events[0] == TextDelta("Hello ")
    assert events[1] == TextDelta("there.")
    assert events[2] == Done("stop")


async def test_tool_call_split_across_chunks() -> None:
    llm = llm_for(TOOL_STREAM)
    events = await collect(llm)
    assert events[0] == ToolCall("c1", "quote_price", {"service_text": "lip filler"})
    assert events[1] == Done("tool_calls")
    assert llm.take_first_token_times()
    kwargs = llm._client.messages.kwargs
    assert kwargs is not None
    assert kwargs["max_tokens"] == 32
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["output_config"] == {"effort": "low"}
    tools = kwargs["tools"]
    assert isinstance(tools, list)
    schema = tools[0]
    assert isinstance(schema, dict)
    assert schema["name"] == "quote_price"
    assert schema["input_schema"] == QUOTE.parameters_json()


async def test_malformed_tool_json_raises_typed_error() -> None:
    stream = (
        ev("content_block_start", block_type="tool_use", block_id="c1", block_name="quote_price"),
        ev("content_block_delta", delta_type="input_json_delta", partial_json="not-json"),
        ev("content_block_stop"),
        ev("message_stop"),
    )
    with pytest.raises(ToolCallParseError) as info:
        await collect(llm_for(stream))
    assert info.value.id == "c1"
    assert info.value.name == "quote_price"
    assert "not-json" not in repr(info.value)
    assert "service_text" not in str(info.value)


async def test_message_start_does_not_satisfy_first_token() -> None:
    stream = (
        ev("message_start"),
        ev("content_block_start", block_type="text"),
        ev("content_block_delta", delta_type="text_delta", text="Hi"),
        ev("message_stop"),
    )
    with pytest.raises(FirstTokenTimeout):
        await collect(llm_for(stream, delays={1: 0.05}), first_token_timeout=0.01, idle_timeout=2)


async def test_idle_stall_after_content_block_start() -> None:
    stream = (
        ev("content_block_start", block_type="text"),
        ev("content_block_delta", delta_type="text_delta", text="Hi"),
        ev("message_stop"),
    )
    with pytest.raises(StreamStalled):
        await collect(llm_for(stream, delays={1: 0.05}), first_token_timeout=2, idle_timeout=0.01)


async def test_early_close_closes_transport() -> None:
    fake = FakeStream(TEXT_STREAM)
    llm = AnthropicLLM(FakeClient(fake))
    stream = llm.stream(
        [SYSTEM, USER],
        [QUOTE],
        max_output_tokens=32,
        first_token_timeout=5,
        idle_timeout=5,
    )
    first = await stream.__anext__()
    assert isinstance(first, TextDelta)
    await stream.aclose()
    assert fake.closed is True
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


def test_translate_system_tools_and_call_started() -> None:
    call = ToolCall("c1", "quote_price", {"service_text": "lip filler"})
    assistant, tool = tool_result_messages(call, {"found": True})
    greeting = Message("assistant", "Thanks for calling.")
    follow = Message("assistant", "How can I help?")
    system, api = to_api_messages([SYSTEM, greeting, follow, USER, assistant, tool])
    assert system == SYSTEM.content
    assert api[0] == {"role": "user", "content": CALL_STARTED}
    assert api[1]["role"] == "assistant"
    content = api[1]["content"]
    assert isinstance(content, list)
    assert content == [{"type": "text", "text": "Thanks for calling.\nHow can I help?"}]
    assert api[2] == {"role": "user", "content": USER.content}
    assert api[3]["role"] == "assistant"
    blocks = api[3]["content"]
    assert isinstance(blocks, list)
    assert blocks[0] == {
        "type": "tool_use",
        "id": "c1",
        "name": "quote_price",
        "input": {"service_text": "lip filler"},
    }
    assert api[4]["role"] == "user"
    results = api[4]["content"]
    assert isinstance(results, list)
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "c1"


def test_missing_key_and_repr_hide_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_EFFORT", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()
    secret = "sk-secret-must-not-leak"
    llm = AnthropicLLM(FakeClient(FakeStream(())), api_key=secret)
    assert secret not in repr(llm)
    assert secret not in str(llm)
    assert llm.model == "claude-haiku-5-5"
    assert llm.effort == "low"


def test_effort_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
    llm = AnthropicLLM(FakeClient(FakeStream(())))
    assert llm.effort == "medium"
    monkeypatch.setenv("ANTHROPIC_EFFORT", "xhigh")
    with pytest.raises(ValueError, match="effort"):
        AnthropicLLM(FakeClient(FakeStream(())))


def test_adapter_does_not_import_retell() -> None:
    tree = ast.parse(Path(integrations.llm_anthropic.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "retell" not in imported
    assert "tests" not in imported and not any(name.startswith("tests.") for name in imported)


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="no ANTHROPIC_API_KEY")
async def test_live_smoke_hello() -> None:
    llm = AnthropicLLM()
    events = await collect(
        llm,
        [SYSTEM, Message("user", "Reply with the single word hello.")],
        [],
        max_output_tokens=32,
        first_token_timeout=15,
        idle_timeout=15,
    )
    assert any(isinstance(event, TextDelta) for event in events)
    assert isinstance(events[-1], Done)
