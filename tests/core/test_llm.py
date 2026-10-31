from __future__ import annotations

import ast
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import pytest

import core.llm
from core.llm import (
    CannedDone,
    CannedText,
    CannedTool,
    CannedTurn,
    Done,
    FirstTokenTimeout,
    Message,
    Model,
    ModelEvent,
    ModelStream,
    ScriptedLLM,
    ScriptedLLMCalledError,
    ScriptExhaustedError,
    ScriptExpectationError,
    ScriptFormatError,
    StreamStalled,
    TextDelta,
    ToolCall,
    ToolCallParseError,
    ToolSchema,
)

FIXTURES = Path(__file__).parent / "fixtures" / "llm"
PLANTED = "I'm pregnant, can I get Botox if I have a cold"
PHONE = "+12035550148"
QUOTE = ToolSchema(
    name="quote_price",
    description="Quote a service",
    parameters={"type": "object", "properties": {"service_text": {"type": "string"}}},
)
USER = Message("user", "how much is lip filler")
SYSTEM = Message("system", "You are the spa assistant.")
CLINICAL_USER = Message("user", PLANTED)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class _StubStream:
    def __init__(self) -> None:
        self._items = iter((TextDelta("ok"), Done("stop")))
        self._closed = False

    def __aiter__(self) -> _StubStream:
        return self

    async def __anext__(self) -> ModelEvent:
        if self._closed:
            raise StopAsyncIteration
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self._closed = True


class StubModel:
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> ModelStream:
        return _StubStream()


def _scripted(*turns: CannedTurn, clock: FakeClock | None = None) -> ScriptedLLM:
    if clock is None:
        return ScriptedLLM(turns)
    return ScriptedLLM(turns, sleep=clock.sleep, clock=clock)


def _texts(*chunks: str, turn_id: str = "t1") -> CannedTurn:
    events: list[CannedText | CannedDone] = [CannedText(chunk) for chunk in chunks]
    events.append(CannedDone())
    return CannedTurn(id=turn_id, events=tuple(events))


async def collect(
    llm: ScriptedLLM,
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
    return [event async for event in stream]


# --------------------------------------------------------------------------- chunks and tools


async def test_chunk_order_from_fixture() -> None:
    llm = ScriptedLLM.load(FIXTURES / "chunk_order.yaml")
    events = await collect(llm)
    texts = [event.text for event in events if isinstance(event, TextDelta)]
    assert texts == ["So that'll be about ", "$450 ", "total."]
    assert "".join(texts) == "So that'll be about $450 total."
    assert events[-1] == Done("stop")
    assert [type(event) for event in events] == [TextDelta, TextDelta, TextDelta, Done]


async def test_tool_call_split_across_chunks_is_one_parsed_call() -> None:
    llm = ScriptedLLM.load(FIXTURES / "tool_split.yaml")
    events = await collect(llm)
    assert events[0] == TextDelta("Let me check.")
    assert events[1] == ToolCall("call_1", "quote_price", {"service_text": "lip filler"})
    assert events[2] == Done("tool_calls")
    assert isinstance(events[1], ToolCall)
    assert dict(events[1].args) == {"service_text": "lip filler"}


async def test_broken_tool_args_raise_typed_parse_error() -> None:
    llm = ScriptedLLM.load(FIXTURES / "tool_bad_json.yaml")
    with pytest.raises(ToolCallParseError) as info:
        await collect(llm)
    assert info.value.id == "call_1"
    assert info.value.name == "quote_price"
    assert "not-json" not in repr(info.value)
    assert "service_text" not in str(info.value)


async def test_tool_call_mid_stream_is_atomic() -> None:
    llm = _scripted(
        CannedTurn(
            id="mid",
            events=(
                CannedText("Sure, "),
                CannedTool("call_1", "quote_price", args={"service_text": "lip filler"}),
                CannedText("done checking."),
                CannedDone("tool_calls"),
            ),
        )
    )
    types = [type(event) for event in await collect(llm)]
    assert types == [TextDelta, ToolCall, TextDelta, Done]


# --------------------------------------------------------------------------- limits


async def test_first_token_timeout_raises_and_consumes_the_turn() -> None:
    clock = FakeClock()
    llm = ScriptedLLM.load(FIXTURES / "two_turns.yaml", sleep=clock.sleep, clock=clock)
    with pytest.raises(FirstTokenTimeout):
        await collect(llm, first_token_timeout=2)
    events = await collect(llm, first_token_timeout=2)
    assert events[0] == TextDelta("second")


async def test_idle_timeout_stalls_after_the_first_delta() -> None:
    clock = FakeClock()
    llm = _scripted(
        CannedTurn(
            id="stall",
            events=(
                CannedText("Hello"),
                CannedText("world", delay_s=10),
                CannedDone(),
            ),
        ),
        clock=clock,
    )
    stream = llm.stream([USER], [], max_output_tokens=8, first_token_timeout=5, idle_timeout=2)
    first = await stream.__anext__()
    assert first == TextDelta("Hello")
    with pytest.raises(StreamStalled):
        await stream.__anext__()
    assert repr(StreamStalled()) == "StreamStalled()"


async def test_max_output_tokens_truncates_then_done_length() -> None:
    llm = _scripted(_texts("a", "b", "c", "d"))
    events = await collect(llm, max_output_tokens=2)
    assert events == [TextDelta("a"), TextDelta("b"), Done("length")]


async def test_invalid_limits_do_not_consume_a_turn() -> None:
    llm = _scripted(_texts("hello"))
    with pytest.raises(ValueError, match="max_output_tokens"):
        llm.stream([USER], [], max_output_tokens=0, first_token_timeout=1, idle_timeout=1)
    with pytest.raises(ValueError, match="first_token_timeout"):
        llm.stream([USER], [], max_output_tokens=1, first_token_timeout=0, idle_timeout=1)
    with pytest.raises(ValueError, match="idle_timeout"):
        llm.stream([USER], [], max_output_tokens=1, first_token_timeout=1, idle_timeout=0)
    events = await collect(llm)
    assert events[0] == TextDelta("hello")
    assert llm.call_count == 1


# --------------------------------------------------------------------------- cancel-safe


async def test_early_close_does_not_raise_or_leave_tasks() -> None:
    llm = ScriptedLLM.load(FIXTURES / "three_then_next.yaml")
    before = set(asyncio.all_tasks())
    stream = llm.stream([USER], [], max_output_tokens=8, first_token_timeout=5, idle_timeout=5)
    first = await stream.__anext__()
    assert first == TextDelta("one")
    await stream.aclose()
    after = set(asyncio.all_tasks())
    assert after <= before
    events = await collect(llm)
    assert events[0] == TextDelta("after close")


async def test_breaking_async_for_plays_the_next_turn() -> None:
    llm = ScriptedLLM.load(FIXTURES / "three_then_next.yaml")
    async for event in llm.stream(
        [USER], [], max_output_tokens=8, first_token_timeout=5, idle_timeout=5
    ):
        assert event == TextDelta("one")
        break
    events = await collect(llm)
    assert events[0] == TextDelta("after close")


# --------------------------------------------------------------------------- script / expect


async def test_script_exhaustion_fails_loudly() -> None:
    llm = ScriptedLLM.load(FIXTURES / "chunk_order.yaml")
    first = await collect(llm)
    with pytest.raises(ScriptExhaustedError) as info:
        await collect(llm)
    assert info.value.call_count == 2
    assert info.value.turns == 1
    assert isinstance(first[0], TextDelta)
    assert llm.call_count == 2


def test_assert_never_called() -> None:
    llm = _scripted(_texts("x"))
    llm.assert_never_called()
    llm.stream([USER], [], max_output_tokens=8, first_token_timeout=5, idle_timeout=5)
    with pytest.raises(ScriptedLLMCalledError) as info:
        llm.assert_never_called()
    assert info.value.call_count == 1
    assert "x" not in repr(info.value)


async def test_expect_rejects_planted_clinical_utterance() -> None:
    llm = ScriptedLLM.load(FIXTURES / "expect_clinical.yaml")
    with pytest.raises(ScriptExpectationError) as info:
        await collect(llm, messages=[SYSTEM, CLINICAL_USER])
    assert info.value.rule == "messages_must_not_contain"
    assert info.value.turn_id == "invent_price"
    shown = repr(info.value) + str(info.value)
    assert PLANTED not in shown
    assert "pregnant" not in shown
    assert "Botox" not in shown


async def test_expect_allows_a_clean_transcript() -> None:
    llm = ScriptedLLM.load(FIXTURES / "expect_clinical.yaml")
    events = await collect(llm, messages=[SYSTEM, USER], tools=[QUOTE])
    assert events[0] == TextDelta("So that'll be about $450 total.")


async def test_expect_roles_and_count_fail_without_content() -> None:
    llm = ScriptedLLM.load(FIXTURES / "expect_clinical.yaml")
    with pytest.raises(ScriptExpectationError) as info:
        await collect(llm, messages=[USER])
    assert info.value.rule == "message_count"
    assert USER.content not in repr(info.value)
    llm_roles = ScriptedLLM.load(FIXTURES / "expect_clinical.yaml")
    with pytest.raises(ScriptExpectationError) as info:
        await collect(llm_roles, messages=[USER, SYSTEM])
    assert info.value.rule == "roles"


# --------------------------------------------------------------------------- privacy and layers


def test_reprs_hide_payloads() -> None:
    delta = TextDelta(PLANTED)
    call = ToolCall("call_1", "quote_price", {"phone": PHONE, "note": PLANTED})
    message = Message("user", PLANTED)
    llm = _scripted(_texts(PLANTED, turn_id="plant"))
    shown = "".join(
        (
            repr(delta),
            str(delta),
            repr(call),
            str(call),
            repr(message),
            str(message),
            repr(llm),
            str(llm),
            repr(Done("stop")),
            repr(QUOTE),
        )
    )
    assert PLANTED not in shown
    assert PHONE not in shown
    assert "2035550148" not in shown
    assert delta.text == PLANTED
    assert call.args["phone"] == PHONE


def test_scripted_and_stub_share_the_protocol() -> None:
    def takes_model(model: Model) -> None:
        assert hasattr(model, "stream")

    takes_model(_scripted(_texts("ok")))
    takes_model(StubModel())
    assert set(get_args(ModelEvent)) == {TextDelta, ToolCall, Done}


def test_empty_script_fails_at_construct() -> None:
    with pytest.raises(ScriptFormatError, match="non-empty"):
        ScriptedLLM(())


def test_empty_text_is_rejected_at_load(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(
        "turns:\n  - id: t\n    events:\n      - {type: text, text: ''}\n      - {type: done}\n"
    )
    with pytest.raises(ScriptFormatError, match="text"):
        ScriptedLLM.load(path)


def test_llm_module_imports_are_closed() -> None:
    tree = ast.parse(Path(core.llm.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.policy" not in imported
    assert "tools" not in imported
    assert "retell" not in imported
    assert imported <= {
        "__future__",
        "asyncio",
        "json",
        "re",
        "time",
        "collections.abc",
        "dataclasses",
        "pathlib",
        "types",
        "typing",
        "yaml",
    }
