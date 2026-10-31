"""Model protocol and ScriptedLLM. No provider client; no payload logging."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Literal, Protocol, Self

import yaml

Role = Literal["system", "user", "assistant", "tool"]
Stop = Literal["stop", "tool_calls", "length"]
Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]
_SLUG = re.compile(r"[a-z][a-z0-9_]*")
_TURN_KEYS = frozenset({"id", "delay_s", "expect", "events"})
_EXPECT_KEYS = frozenset({"message_count", "roles", "tool_names", "messages_must_not_contain"})
_TEXT_KEYS = frozenset({"type", "text", "delay_s"})
_TOOL_KEYS = frozenset({"type", "id", "name", "args", "args_chunks", "delay_s"})
_DONE_KEYS = frozenset({"type", "stop", "delay_s"})
_ROLES = frozenset({"system", "user", "assistant", "tool"})
_STOPS = frozenset({"stop", "tool_calls", "length"})


# --------------------------------------------------------------------------- errors


class FirstTokenTimeout(Exception):
    def __repr__(self) -> str:
        return "FirstTokenTimeout()"

    def __str__(self) -> str:
        return repr(self)


class StreamStalled(Exception):
    def __repr__(self) -> str:
        return "StreamStalled()"

    def __str__(self) -> str:
        return repr(self)


class ToolCallParseError(Exception):
    def __init__(self, call_id: str, name: str) -> None:
        self.id = call_id
        self.name = name
        super().__init__(f"ToolCallParseError(id={call_id!r}, name={name!r})")

    def __repr__(self) -> str:
        return f"ToolCallParseError(id={self.id!r}, name={self.name!r})"

    def __str__(self) -> str:
        return repr(self)


class ScriptExhaustedError(Exception):
    def __init__(self, call_count: int, turns: int) -> None:
        self.call_count = call_count
        self.turns = turns
        super().__init__(f"ScriptExhaustedError(call_count={call_count}, turns={turns})")

    def __repr__(self) -> str:
        return f"ScriptExhaustedError(call_count={self.call_count}, turns={self.turns})"

    def __str__(self) -> str:
        return repr(self)


class ScriptExpectationError(Exception):
    def __init__(self, turn_id: str, rule: str) -> None:
        self.turn_id = turn_id
        self.rule = rule
        super().__init__(f"ScriptExpectationError(turn_id={turn_id!r}, rule={rule!r})")

    def __repr__(self) -> str:
        return f"ScriptExpectationError(turn_id={self.turn_id!r}, rule={self.rule!r})"

    def __str__(self) -> str:
        return repr(self)


class ScriptedLLMCalledError(Exception):
    def __init__(self, call_count: int) -> None:
        self.call_count = call_count
        super().__init__(f"ScriptedLLMCalledError(call_count={call_count})")

    def __repr__(self) -> str:
        return f"ScriptedLLMCalledError(call_count={self.call_count})"

    def __str__(self) -> str:
        return repr(self)


class ScriptFormatError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


# --------------------------------------------------------------------------- events and carriers


@dataclass(frozen=True)
class TextDelta:
    text: str

    def __repr__(self) -> str:
        return f"TextDelta(chars={len(self.text)})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: Mapping[str, object]

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, name={self.name!r}, arg_keys={len(self.args)})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Done:
    stop: Stop = "stop"

    def __repr__(self) -> str:
        return f"Done(stop={self.stop!r})"

    def __str__(self) -> str:
        return repr(self)


ModelEvent = TextDelta | ToolCall | Done


@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, chars={len(self.content)})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    description: str
    parameters: Mapping[str, object]

    def parameters_json(self) -> dict[str, object]:
        """Plain dicts and lists, recursively, so json.dumps succeeds."""
        converted = _jsonable(self.parameters)
        if not isinstance(converted, dict):
            raise TypeError("parameters must be a JSON object")
        return converted

    def __repr__(self) -> str:
        return f"ToolSchema(name={self.name!r})"

    def __str__(self) -> str:
        return repr(self)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


class ModelStream(Protocol):
    def __aiter__(self) -> ModelStream: ...

    async def __anext__(self) -> ModelEvent: ...

    async def aclose(self) -> None: ...


class Model(Protocol):
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> ModelStream:
        """Stream one model turn: TextDelta*, ToolCall*, then one Done.

        A tool call that arrives split across the wire is assembled here and
        yielded once, with parsed args. Real adapters must close their
        transport in ``finally`` so an early ``aclose()`` cannot leak a
        connection or a pending read.
        """
        ...


# --------------------------------------------------------------------------- canned script


@dataclass(frozen=True)
class Expect:
    message_count: int | None = None
    roles: tuple[Role, ...] | None = None
    tool_names: tuple[str, ...] | None = None
    messages_must_not_contain: tuple[str, ...] = ()


@dataclass(frozen=True)
class CannedText:
    text: str
    delay_s: float = 0


@dataclass(frozen=True)
class CannedTool:
    id: str
    name: str
    args: Mapping[str, object] | None = None
    args_chunks: tuple[str, ...] = ()
    delay_s: float = 0


@dataclass(frozen=True)
class CannedDone:
    stop: Stop = "stop"
    delay_s: float = 0


CannedEvent = CannedText | CannedTool | CannedDone


@dataclass(frozen=True)
class CannedTurn:
    id: str
    events: tuple[CannedEvent, ...]
    delay_s: float = 0
    expect: Expect | None = None


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class ScriptedLLM:
    """Replay canned turns from memory or a YAML fixture. Fail loudly when empty."""

    def __init__(
        self,
        turns: Sequence[CannedTurn],
        *,
        sleep: Sleep | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not turns:
            raise ScriptFormatError("<memory>", "turns must be a non-empty list")
        self._turns = tuple(turns)
        self._index = 0
        self._calls = 0
        self._sleep: Sleep = sleep if sleep is not None else asyncio.sleep
        self._clock: Clock = clock if clock is not None else time.monotonic

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        sleep: Sleep | None = None,
        clock: Clock | None = None,
    ) -> ScriptedLLM:
        return cls(_load_turns(path), sleep=sleep, clock=clock)

    @property
    def call_count(self) -> int:
        return self._calls

    def assert_never_called(self) -> None:
        if self._calls:
            raise ScriptedLLMCalledError(self._calls)

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> _TurnStream:
        _require_positive("max_output_tokens", max_output_tokens)
        _require_positive("first_token_timeout", first_token_timeout)
        _require_positive("idle_timeout", idle_timeout)
        self._calls += 1
        if self._index >= len(self._turns):
            raise ScriptExhaustedError(self._calls, len(self._turns))
        turn = self._turns[self._index]
        self._index += 1
        _check_expect(turn, messages, tools)
        return _TurnStream(self._play(turn, max_output_tokens, first_token_timeout, idle_timeout))

    async def _play(
        self,
        turn: CannedTurn,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[ModelEvent]:
        yielded = 0
        for index, event in enumerate(turn.events):
            token = not isinstance(event, CannedDone)
            if token and yielded >= max_output_tokens:
                yield Done("length")
                return
            delay = turn.delay_s if index == 0 else event.delay_s
            timeout = first_token_timeout if index == 0 else idle_timeout
            error: type[FirstTokenTimeout | StreamStalled]
            error = FirstTokenTimeout if index == 0 else StreamStalled
            await self._wait(delay, timeout, error)
            if token:
                yield _materialize(event)
                yielded += 1
                continue
            assert isinstance(event, CannedDone)
            yield Done(event.stop)
            return

    async def _wait(
        self,
        delay: float,
        timeout: float,
        error: type[FirstTokenTimeout | StreamStalled],
    ) -> None:
        if delay <= 0:
            return
        start = self._clock()
        await self._sleep(min(delay, timeout))
        if self._clock() - start >= timeout:
            raise error()

    def __repr__(self) -> str:
        return f"ScriptedLLM(turns={len(self._turns)}, calls={self._calls})"

    def __str__(self) -> str:
        return repr(self)


class _TurnStream:
    """Async iterator that swallows aclose/cancel so the caller never sees them."""

    def __init__(self, play: AsyncIterator[ModelEvent]) -> None:
        self._play = play
        self._closed = False

    def __aiter__(self) -> Self:
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
        close = getattr(self._play, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except (GeneratorExit, asyncio.CancelledError, StopAsyncIteration):
            return

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


def _materialize(event: CannedEvent) -> ModelEvent:
    if isinstance(event, CannedText):
        return TextDelta(event.text)
    if isinstance(event, CannedDone):
        return Done(event.stop)
    args = event.args
    if args is None:
        try:
            loaded = json.loads("".join(event.args_chunks))
        except json.JSONDecodeError:
            raise ToolCallParseError(event.id, event.name) from None
        if not isinstance(loaded, dict):
            raise ToolCallParseError(event.id, event.name)
        args = loaded
    return ToolCall(event.id, event.name, MappingProxyType(dict(args)))


def _check_expect(
    turn: CannedTurn, messages: Sequence[Message], tools: Sequence[ToolSchema]
) -> None:
    expect = turn.expect
    if expect is None:
        return
    if expect.message_count is not None and len(messages) != expect.message_count:
        raise ScriptExpectationError(turn.id, "message_count")
    if expect.roles is not None:
        roles = tuple(message.role for message in messages)
        if roles != expect.roles:
            raise ScriptExpectationError(turn.id, "roles")
    if expect.tool_names is not None:
        names = tuple(tool.name for tool in tools)
        if names != expect.tool_names:
            raise ScriptExpectationError(turn.id, "tool_names")
    for needle in expect.messages_must_not_contain:
        if any(needle in message.content for message in messages):
            raise ScriptExpectationError(turn.id, "messages_must_not_contain")


# --------------------------------------------------------------------------- YAML loader


def _load_turns(path: Path) -> tuple[CannedTurn, ...]:
    source = str(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScriptFormatError(source, f"invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise ScriptFormatError(source, "script must be a mapping")
    extra = set(raw) - {"turns"}
    if extra:
        raise ScriptFormatError(source, f"unknown fields {sorted(extra)}")
    rows = raw.get("turns")
    if not isinstance(rows, list) or not rows:
        raise ScriptFormatError(source, "turns must be a non-empty list")
    turns: list[CannedTurn] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        turns.append(_load_turn(source, index, item, seen))
    return tuple(turns)


def _load_turn(source: str, index: int, item: object, seen: set[str]) -> CannedTurn:
    loc = f"turns[{index}]"
    if not isinstance(item, dict):
        raise ScriptFormatError(source, f"{loc} must be a mapping")
    extra = set(item) - _TURN_KEYS
    if extra:
        raise ScriptFormatError(source, f"{loc} unknown fields {sorted(extra)}")
    name = item.get("id")
    if not isinstance(name, str) or not _SLUG.fullmatch(name):
        raise ScriptFormatError(source, f"{loc}.id must be a slug")
    if name in seen:
        raise ScriptFormatError(source, f"duplicate turn {name!r}")
    seen.add(name)
    delay = _delay(source, f"{loc}.delay_s", item.get("delay_s", 0))
    expect = _load_expect(source, loc, item.get("expect"))
    events_raw = item.get("events")
    if not isinstance(events_raw, list) or not events_raw:
        raise ScriptFormatError(source, f"{loc}.events must be a non-empty list")
    events = tuple(
        _load_event(source, f"{loc}.events[{i}]", row) for i, row in enumerate(events_raw)
    )
    if not isinstance(events[-1], CannedDone):
        raise ScriptFormatError(source, f"{loc}.events must end with done")
    if any(isinstance(event, CannedDone) for event in events[:-1]):
        raise ScriptFormatError(source, f"{loc}.events may have only one done")
    return CannedTurn(id=name, events=events, delay_s=delay, expect=expect)


def _load_expect(source: str, loc: str, raw: object) -> Expect | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ScriptFormatError(source, f"{loc}.expect must be a mapping")
    extra = set(raw) - _EXPECT_KEYS
    if extra:
        raise ScriptFormatError(source, f"{loc}.expect unknown fields {sorted(extra)}")
    count = raw.get("message_count")
    if count is not None and not (isinstance(count, int) and not isinstance(count, bool)):
        raise ScriptFormatError(source, f"{loc}.expect.message_count must be an int")
    roles_raw = raw.get("roles")
    roles: tuple[Role, ...] | None = None
    if roles_raw is not None:
        if not isinstance(roles_raw, list) or not roles_raw:
            raise ScriptFormatError(source, f"{loc}.expect.roles must be a non-empty list")
        parsed: list[Role] = []
        for role in roles_raw:
            if role not in _ROLES:
                raise ScriptFormatError(source, f"{loc}.expect.roles has an unknown role")
            parsed.append(role)
        roles = tuple(parsed)
    names_raw = raw.get("tool_names")
    names: tuple[str, ...] | None = None
    if names_raw is not None:
        if not isinstance(names_raw, list) or not all(isinstance(n, str) for n in names_raw):
            raise ScriptFormatError(source, f"{loc}.expect.tool_names must be a list of strings")
        names = tuple(names_raw)
    needles = raw.get("messages_must_not_contain", [])
    if not isinstance(needles, list) or not all(isinstance(n, str) and n for n in needles):
        raise ScriptFormatError(
            source, f"{loc}.expect.messages_must_not_contain must be non-empty strings"
        )
    return Expect(
        message_count=count,
        roles=roles,
        tool_names=names,
        messages_must_not_contain=tuple(needles),
    )


def _load_event(source: str, loc: str, item: object) -> CannedEvent:
    if not isinstance(item, dict) or "type" not in item:
        raise ScriptFormatError(source, f"{loc} must be a mapping with type")
    kind = item["type"]
    if kind == "text":
        return _load_text(source, loc, item)
    if kind == "tool_call":
        return _load_tool(source, loc, item)
    if kind == "done":
        return _load_done(source, loc, item)
    raise ScriptFormatError(source, f"{loc}.type must be text, tool_call, or done")


def _load_text(source: str, loc: str, item: dict[str, object]) -> CannedText:
    extra = set(item) - _TEXT_KEYS
    if extra:
        raise ScriptFormatError(source, f"{loc} unknown fields {sorted(extra)}")
    text = item.get("text")
    if not isinstance(text, str) or not text:
        raise ScriptFormatError(source, f"{loc}.text must be a non-empty string")
    return CannedText(text=text, delay_s=_delay(source, f"{loc}.delay_s", item.get("delay_s", 0)))


def _load_tool(source: str, loc: str, item: dict[str, object]) -> CannedTool:
    extra = set(item) - _TOOL_KEYS
    if extra:
        raise ScriptFormatError(source, f"{loc} unknown fields {sorted(extra)}")
    call_id = item.get("id")
    name = item.get("name")
    if not isinstance(call_id, str) or not call_id:
        raise ScriptFormatError(source, f"{loc}.id is required")
    if not isinstance(name, str) or not name:
        raise ScriptFormatError(source, f"{loc}.name is required")
    has_args = "args" in item
    has_chunks = "args_chunks" in item
    if has_args == has_chunks:
        raise ScriptFormatError(source, f"{loc} needs args or args_chunks, not both")
    delay = _delay(source, f"{loc}.delay_s", item.get("delay_s", 0))
    if has_args:
        args = item.get("args")
        if not isinstance(args, dict):
            raise ScriptFormatError(source, f"{loc}.args must be a mapping")
        return CannedTool(id=call_id, name=name, args=MappingProxyType(dict(args)), delay_s=delay)
    chunks = item.get("args_chunks")
    if not isinstance(chunks, list) or not chunks or not all(isinstance(c, str) for c in chunks):
        raise ScriptFormatError(source, f"{loc}.args_chunks must be a non-empty list of strings")
    return CannedTool(id=call_id, name=name, args_chunks=tuple(chunks), delay_s=delay)


def _load_done(source: str, loc: str, item: dict[str, object]) -> CannedDone:
    extra = set(item) - _DONE_KEYS
    if extra:
        raise ScriptFormatError(source, f"{loc} unknown fields {sorted(extra)}")
    stop = item.get("stop", "stop")
    if stop == "stop" or stop == "tool_calls" or stop == "length":
        return CannedDone(
            stop=stop, delay_s=_delay(source, f"{loc}.delay_s", item.get("delay_s", 0))
        )
    raise ScriptFormatError(source, f"{loc}.stop must be stop, tool_calls, or length")


def _delay(source: str, loc: str, raw: object) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or raw < 0:
        raise ScriptFormatError(source, f"{loc} must be a number >= 0")
    return float(raw)
