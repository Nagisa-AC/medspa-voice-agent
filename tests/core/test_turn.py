from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import core.turn
from core.crypto import InMemoryKeyStore
from core.escalation import EMERGENCY_LINE, TRANSFER_LINE
from core.events import InMemoryEventLog
from core.guardrail import load_output_pack
from core.llm import (
    CannedDone,
    CannedText,
    CannedTool,
    CannedTurn,
    Message,
    ScriptedLLM,
    ToolSchema,
)
from core.numbers import find_numbers
from core.policy import load_policy_engine
from core.prompt import build
from core.state import AgentLine, InMemorySessionStore, new_call_state
from core.tenant import extra_price_words, load_tenant_file
from core.turn import (
    ASK_BEST,
    ASK_DIGITS,
    GIVE_UP,
    GOODBYE,
    HEDGE,
    NUMBER_HEDGE,
    OutputChunk,
    TurnLoop,
)
from integrations.booking import MockBookingAdapter
from tools.scheduling import check_availability

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NY = ZoneInfo("America/New_York")
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
PACK = load_output_pack(ROOT / "packs" / "medspa.output.yaml")
ENGINE = load_policy_engine(GLOW)
_WORDS = extra_price_words(GLOW)
EXTRACT = partial(
    find_numbers,
    extra_price_words=_WORDS.extra_price_words,
    service_words=_WORDS.service_words,
)
SECRET = b"n" * 32
CALLER = "+12035550148"
TUE = datetime(2026, 11, 3, 11, 0, tzinfo=NY)
SUN = datetime(2026, 11, 8, 14, 0, tzinfo=NY)
CLINICAL = "I'm pregnant, can I get Botox if I have a cold"
EMERGENCY = "I can't breathe"
IDENTITY = "is this a bot"
FILLER_SAY = "Dermal Filler typically runs $700 to $950 per syringe."
TOX_SAY = GLOW.faq[0].say


class FakeMono:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _text(turn_id: str, *parts: str) -> CannedTurn:
    return CannedTurn(id=turn_id, events=(*tuple(CannedText(part) for part in parts), CannedDone()))


def _tool(turn_id: str, name: str, args: dict[str, object], call_id: str = "c1") -> CannedTurn:
    return CannedTurn(
        id=turn_id,
        events=(CannedTool(id=call_id, name=name, args=args), CannedDone("tool_calls")),
    )


def _scripted(*turns: CannedTurn, mono: FakeMono | None = None) -> ScriptedLLM:
    if mono is None:
        return ScriptedLLM(turns)
    return ScriptedLLM(turns, sleep=mono.sleep, clock=mono)


class RecordingLLM:
    def __init__(self, inner: ScriptedLLM) -> None:
        self.inner = inner
        self.calls: list[tuple[Message, ...]] = []

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> Any:
        self.calls.append(tuple(messages))
        return self.inner.stream(
            messages,
            tools,
            max_output_tokens=max_output_tokens,
            first_token_timeout=first_token_timeout,
            idle_timeout=idle_timeout,
        )


async def collect(iterator: AsyncIterator[OutputChunk]) -> list[OutputChunk]:
    return [chunk async for chunk in iterator]


async def setup(
    *,
    model: ScriptedLLM | None = None,
    started: datetime = TUE,
    caller_number: str | None = CALLER,
    now: datetime | None = None,
) -> tuple[TurnLoop, Any, InMemoryEventLog, InMemorySessionStore]:
    keys = InMemoryKeyStore()
    data_key_id = await keys.create("call_1")
    log = InMemoryEventLog(keys)
    store = InMemorySessionStore()
    state = new_call_state(
        tenant=GLOW,
        call_id="call_1",
        data_key_id=data_key_id,
        now=started,
        extract=EXTRACT,
    )
    state.caller_number = caller_number
    await store.create(state)
    clock_now = now if now is not None else started

    def clock() -> datetime:
        return clock_now

    llm = model if model is not None else _scripted(_text("unused", "ok"))
    loop = TurnLoop(
        GLOW,
        store,
        log,
        llm,
        MockBookingAdapter.from_yaml(
            ROOT / "tenants" / "glow_aesthetics.mock.yaml",
            tz=GLOW.tz,
            clock=lambda: clock_now,
        ),
        SECRET,
        clock=clock,
        engine=ENGINE,
        pack=PACK,
        extract=EXTRACT,
    )
    return loop, state, log, store


async def kinds(log: InMemoryEventLog) -> list[str]:
    return [event.kind for event in await log.read("call_1")]


def history_text(state: Any) -> str:
    return "\n".join(
        line.text
        for line in state.transcript
        if isinstance(line, AgentLine) and line.history != "omit"
    )


# --------------------------------------------------------------------------- connect / greeting


async def test_connect_speaks_greeting_included() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.connect(state))
    assert [chunk.text for chunk in chunks] == [GLOW.greeting]
    assert chunks[0].origin == "backend"
    assert chunks[-1].content_complete is True
    assert state.turn_count == 0
    line = state.transcript[0]
    assert isinstance(line, AgentLine)
    assert line.history == "include"
    model.assert_never_called()


# --------------------------------------------------------------------------- model pass / block


async def test_passed_model_sentence_is_included() -> None:
    loop, state, _log, _store = await setup(model=_scripted(_text("ok", "Happy to help.")))
    chunks = await collect(loop.handle_caller_turn(state, "I'd like to book filler Thursday"))
    assert any(chunk.text == "Happy to help." and chunk.origin == "model" for chunk in chunks)
    assert any(
        isinstance(line, AgentLine) and line.text == "Happy to help." and line.history == "include"
        for line in state.transcript
    )


async def test_invented_price_is_replaced_and_dropped() -> None:
    loop, state, log, _store = await setup(
        model=_scripted(_text("price", "So that'll be about $450 total. And more."))
    )
    chunks = await collect(loop.handle_caller_turn(state, "how much is filler"))
    spoken = [chunk.text for chunk in chunks]
    assert NUMBER_HEDGE in spoken
    assert "And more." not in spoken
    assert "$450" not in history_text(state)
    assert not any(isinstance(line, AgentLine) and "$450" in line.text for line in state.transcript)
    events = await log.read("call_1")
    blocked = [event for event in events if event.kind == "guardrail_blocked"]
    assert blocked[0].payload["check"] == "numbers"
    built = build(GLOW, state)
    joined = "\n".join(m.content for m in built.messages if m.role != "system")
    assert "$450" not in joined
    assert "450" not in joined


async def test_clinical_model_sentence_never_enters_history() -> None:
    loop, state, log, _store = await setup(
        model=_scripted(_text("clin", "Filler lasts about a year."))
    )
    chunks = await collect(loop.handle_caller_turn(state, "tell me about filler"))
    assert PACK.deflect in [chunk.text for chunk in chunks]
    assert not any("lasts about a year" in line.text for line in state.transcript)
    events = await log.read("call_1")
    blocked = [event for event in events if event.kind == "guardrail_blocked"]
    assert blocked[0].payload["check"] == "clinical"


async def test_whitespace_only_flush_is_not_spoken() -> None:
    loop, state, _log, _store = await setup(model=_scripted(_text("ws", "Hello.   ")))
    chunks = await collect(loop.handle_caller_turn(state, "hi there"))
    assert any(chunk.text.startswith("Hello.") for chunk in chunks)
    assert not any(chunk.text.strip() == "" for chunk in chunks)


# --------------------------------------------------------------------------- tools


async def test_quote_price_say_direct_before_follow_up() -> None:
    model = _scripted(
        _tool("q", "quote_price", {"service_text": "lip filler"}),
        _text("follow", "Want me to look at times?"),
    )
    loop, state, log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "how much is lip filler"))
    texts = [chunk.text for chunk in chunks]
    assert texts[0] == FILLER_SAY
    assert texts[1] == "Want me to look at times?"
    assert any(
        isinstance(line, AgentLine)
        and line.history == "marker"
        and line.marker == "price"
        and line.label == "Dermal Filler"
        for line in state.transcript
    )
    built = build(GLOW, state)
    history = "\n".join(m.content for m in built.messages if m.role != "system")
    assert "[price quoted for Dermal Filler]" in history
    assert "$700" not in history
    assert "tool_invoked" in await kinds(log)
    assert "tool_result" in await kinds(log)


async def test_unquotable_quote_is_included_not_price_marker() -> None:
    model = _scripted(
        _tool("q", "quote_price", {"service_text": "weight loss"}),
        _text("follow", "I can look at consult times."),
    )
    loop, state, _log, _store = await setup(model=model)
    await collect(loop.handle_caller_turn(state, "how much is weight loss"))
    quote_lines = [
        line for line in state.transcript if isinstance(line, AgentLine) and "consult" in line.text
    ]
    assert quote_lines
    assert all(line.history == "include" and line.marker is None for line in quote_lines)
    assert not any(
        isinstance(line, AgentLine) and line.marker == "price" for line in state.transcript
    )
    built = build(GLOW, state)
    history = "\n".join(m.content for m in built.messages if m.role != "system")
    assert "[price quoted" not in history


async def test_in_turn_recall_is_user_assistant_tool_without_marker() -> None:
    inner = _scripted(
        CannedTurn(
            id="q",
            events=(
                CannedText("Sure. "),
                CannedTool(id="c1", name="quote_price", args={"service_text": "lip filler"}),
                CannedDone("tool_calls"),
            ),
        ),
        _text("follow", "Want me to look at times?"),
    )
    recorder = RecordingLLM(inner)
    loop, state, _log, _store = await setup(model=inner)
    loop.model = recorder
    await collect(loop.handle_caller_turn(state, "how much is lip filler"))
    assert len(recorder.calls) == 2
    recall = [message for message in recorder.calls[1] if message.role != "system"]
    assert [message.role for message in recall] == ["user", "assistant", "tool"]
    user, assistant, tool = recall
    assert user.content == "how much is lip filler"
    assert assistant.content.startswith("Sure.")
    assert assistant.tool_calls
    assert assistant.tool_calls[0].name == "quote_price"
    assert tool.role == "tool"
    assert "[price quoted" not in assistant.content
    assert "[price quoted" not in user.content
    assert "[price quoted" not in tool.content
    joined = "\n".join(message.content for message in recall)
    assert "[price quoted for Dermal Filler]" not in joined
    next_history = "\n".join(m.content for m in build(GLOW, state).messages if m.role != "system")
    assert "[price quoted for Dermal Filler]" in next_history


async def test_lookup_faq_hit_is_marker_and_logged() -> None:
    model = _scripted(
        _tool("f", "lookup_faq", {"question_text": "how long does botox last"}),
        _text("follow", "Anything else?"),
    )
    loop, state, log, _store = await setup(model=model)
    await collect(loop.handle_caller_turn(state, "how long does botox last"))
    built = build(GLOW, state)
    history = "\n".join(m.content for m in built.messages if m.role != "system")
    assert TOX_SAY not in history
    assert "[answered from the spa's FAQ]" in history
    assert any(
        isinstance(line, AgentLine) and line.history == "marker" and line.marker == "faq"
        for line in state.transcript
    )
    assert "faq_served" in await kinds(log)


async def test_check_availability_say_is_included() -> None:
    model = _scripted(
        _tool("a", "check_availability", {"service_text": "botox"}),
        _text("follow", "Which of those works?"),
    )
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "when can I come in for botox"))
    spoken = [chunk.text for chunk in chunks]
    assert any(text.startswith("I have ") for text in spoken)
    assert any(
        isinstance(line, AgentLine)
        and line.history == "include"
        and line.text.startswith("I have ")
        for line in state.transcript
    )
    built = build(GLOW, state)
    history = "\n".join(m.content for m in built.messages if m.role != "system")
    assert "I have " in history


async def test_book_say_is_omitted_and_logged() -> None:
    loop, state, log, _store = await setup(model=_scripted(_text("unused", "x")))
    listed = await check_availability(
        GLOW,
        state,
        loop.adapter,
        "botox",
        earliest_iso=None,
        extract=EXTRACT,
        now=TUE,
        limit=1,
    )
    slots = listed.model["slots"]
    assert isinstance(slots, tuple) and slots
    iso = slots[0]["start"]
    assert isinstance(iso, str)
    loop.model = _scripted(
        _tool(
            "b",
            "book",
            {
                "slot_start_iso": iso,
                "patient_name": "Dana Reyes",
                "patient_phone": "+12035551234",
                "confirmed": True,
            },
        ),
        _text("follow", "See you then."),
    )
    chunks = await collect(loop.handle_caller_turn(state, "yes book that time please"))
    spoken = [chunk.text for chunk in chunks]
    assert any(text.startswith("You're all set") for text in spoken)
    assert any(
        isinstance(line, AgentLine)
        and line.history == "omit"
        and line.text.startswith("You're all set")
        for line in state.transcript
    )
    built = build(GLOW, state)
    history = "\n".join(m.content for m in built.messages if m.role != "system")
    assert "You're all set" not in history
    assert "$" not in history
    kind_list = await kinds(log)
    assert "booking_attempted" in kind_list
    assert "booking_confirmed" in kind_list


async def test_three_tool_rounds_then_hedge() -> None:
    model = _scripted(
        *(
            _tool(f"q{i}", "quote_price", {"service_text": "botox"}, call_id=f"c{i}")
            for i in range(4)
        )
    )
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "how much is botox"))
    quotes = [chunk.text for chunk in chunks if "typically runs" in chunk.text]
    assert len(quotes) == 3
    assert HEDGE in [chunk.text for chunk in chunks]


# --------------------------------------------------------------------------- timeouts / errors


async def test_first_token_timeout_hedges() -> None:
    mono = FakeMono()
    model = ScriptedLLM(
        [CannedTurn(id="slow", delay_s=5, events=(CannedText("hi"), CannedDone()))],
        sleep=mono.sleep,
        clock=mono,
    )
    loop, state, log, _store = await setup(model=model)
    loop.first_token_timeout = 1
    loop.idle_timeout = 1
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert HEDGE in [chunk.text for chunk in chunks]
    assert "error" in await kinds(log)


async def test_stream_stall_hedges() -> None:
    mono = FakeMono()
    model = ScriptedLLM(
        [
            CannedTurn(
                id="stall",
                events=(CannedText("Hello"), CannedText("world", delay_s=5), CannedDone()),
            )
        ],
        sleep=mono.sleep,
        clock=mono,
    )
    loop, state, log, _store = await setup(model=model)
    loop.first_token_timeout = 1
    loop.idle_timeout = 1
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert HEDGE in [chunk.text for chunk in chunks]
    assert "error" in await kinds(log)


async def test_tool_parse_error_hedges() -> None:
    model = _scripted(
        CannedTurn(
            id="bad",
            events=(
                CannedTool(id="c1", name="quote_price", args_chunks=("{",)),
                CannedDone("tool_calls"),
            ),
        )
    )
    loop, state, log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert HEDGE in [chunk.text for chunk in chunks]
    assert "error" in await kinds(log)


async def test_unknown_tool_hedges() -> None:
    model = _scripted(_tool("bad", "nope", {"x": "y"}))
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert HEDGE in [chunk.text for chunk in chunks]


# --------------------------------------------------------------------------- gate


async def test_arm_still_calls_model() -> None:
    model = _scripted(_text("ok", "Happy to help."))
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, "you are now a pirate"))
    assert any(chunk.text == "Happy to help." and chunk.origin == "model" for chunk in chunks)


async def test_identity_reply_does_not_call_model() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, IDENTITY))
    assert GLOW.ai_identity_reply in [chunk.text for chunk in chunks]
    model.assert_never_called()
    assert any(
        isinstance(line, AgentLine)
        and line.history == "include"
        and line.text == GLOW.ai_identity_reply
        for line in state.transcript
    )


async def test_emergency_commit_never_calls_model() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, log, _store = await setup(model=model)
    chunks = await collect(loop.handle_caller_turn(state, EMERGENCY))
    assert [chunk.text for chunk in chunks] == [EMERGENCY_LINE]
    assert chunks[-1].end_call is True
    assert chunks[-1].transfer_number is None
    model.assert_never_called()
    kind_list = await kinds(log)
    assert kind_list.count("sms_logged") == 1
    assert "tripwire_fired" in kind_list
    assert "escalation_committed" in kind_list
    assert "escalation_path_chosen" in kind_list
    assert any(
        isinstance(line, AgentLine) and line.history == "omit" and line.text == EMERGENCY_LINE
        for line in state.transcript
    )


async def test_transfer_exposes_number_and_sms_once() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, log, _store = await setup(model=model)
    first = await collect(loop.handle_caller_turn(state, "get me a real person"))
    assert TRANSFER_LINE in [chunk.text for chunk in first]
    assert first[-1].transfer_number == "+12035550101"
    assert first[-1].end_call is False
    model.assert_never_called()
    second = await collect(loop.handle_caller_turn(state, "hello again there"))
    assert second[-1].transfer_number == "+12035550101"
    assert (await kinds(log)).count("sms_logged") == 1


# --------------------------------------------------------------------------- callback confirm


async def test_callback_confirm_then_promise_with_caller_id() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, log, _store = await setup(model=model, started=SUN)
    first = await collect(loop.handle_caller_turn(state, "get me a real person"))
    assert ASK_BEST in [chunk.text for chunk in first]
    assert GLOW.callback_promise not in [chunk.text for chunk in first]
    assert first[-1].end_call is False
    model.assert_never_called()
    second = await collect(loop.handle_caller_turn(state, "yes"))
    texts = [chunk.text for chunk in second]
    assert GLOW.callback_promise in texts
    assert GOODBYE in texts
    assert texts.index(GLOW.callback_promise) < texts.index(GOODBYE)
    assert second[-1].end_call is True
    assert state.caller_number_confirmed is True
    assert any(
        isinstance(line, AgentLine)
        and line.text == GLOW.callback_promise
        and line.history == "omit"
        for line in state.transcript
    )
    assert (await kinds(log)).count("sms_logged") == 1


@pytest.mark.parametrize(
    "utterance",
    ("yes please", "yeah that\u2019s fine", "yes it is"),
)
async def test_lenient_yes_confirms(utterance: str) -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN)
    await collect(loop.handle_caller_turn(state, "get me a real person"))
    second = await collect(loop.handle_caller_turn(state, utterance))
    assert GLOW.callback_promise in [chunk.text for chunk in second]
    assert GOODBYE in [chunk.text for chunk in second]
    assert second[-1].end_call is True
    assert state.caller_number_confirmed is True
    model.assert_never_called()


async def test_lenient_no_asks_digits() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN)
    await collect(loop.handle_caller_turn(state, "get me a real person"))
    second = await collect(loop.handle_caller_turn(state, "no use my cell"))
    assert ASK_DIGITS in [chunk.text for chunk in second]
    assert GLOW.callback_promise not in [chunk.text for chunk in second]
    assert state.caller_number_confirmed is False
    model.assert_never_called()


async def test_yes_no_wait_is_not_yes() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN)
    await collect(loop.handle_caller_turn(state, "get me a real person"))
    second = await collect(loop.handle_caller_turn(state, "yes no wait"))
    assert GLOW.callback_promise not in [chunk.text for chunk in second]
    assert state.caller_number_confirmed is False
    assert second[-1].end_call is False
    model.assert_never_called()


async def test_callback_no_then_new_number() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN)
    first = await collect(loop.handle_caller_turn(state, "get me a real person"))
    assert ASK_BEST in [chunk.text for chunk in first]
    second = await collect(loop.handle_caller_turn(state, "no"))
    assert ASK_DIGITS in [chunk.text for chunk in second]
    third = await collect(loop.handle_caller_turn(state, "203 555 0148"))
    texts = [chunk.text for chunk in third]
    assert any("two zero three" in chunk.text for chunk in third)
    assert GLOW.callback_promise in texts
    assert GOODBYE in texts
    assert third[-1].end_call is True
    assert state.caller_number_confirmed is True
    model.assert_never_called()


async def test_callback_without_caller_id_asks_digits() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN, caller_number=None)
    first = await collect(loop.handle_caller_turn(state, "get me a real person"))
    assert ASK_DIGITS in [chunk.text for chunk in first]
    assert ASK_BEST not in [chunk.text for chunk in first]
    second = await collect(loop.handle_caller_turn(state, "203 555 0148"))
    texts = [chunk.text for chunk in second]
    assert any("two zero three" in chunk.text for chunk in second)
    assert GLOW.callback_promise in texts
    assert GOODBYE in texts
    assert state.caller_number_confirmed is True
    assert state.caller_number == CALLER


async def test_callback_give_up_has_goodbye_no_promise() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, _log, _store = await setup(model=model, started=SUN, caller_number=None)
    await collect(loop.handle_caller_turn(state, "get me a real person"))
    await collect(loop.handle_caller_turn(state, "not a number"))
    third = await collect(loop.handle_caller_turn(state, "still not"))
    assert GIVE_UP in [chunk.text for chunk in third]
    assert GOODBYE in GIVE_UP
    assert GLOW.callback_promise not in [chunk.text for chunk in third]
    assert third[-1].end_call is True
    assert state.caller_number_confirmed is False


async def test_two_commits_log_sms_once() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, log, _store = await setup(model=model)
    await collect(loop.handle_caller_turn(state, CLINICAL))
    await collect(loop.handle_caller_turn(state, CLINICAL))
    assert (await kinds(log)).count("sms_logged") == 1
    model.assert_never_called()


async def test_clinical_utterance_redacted_in_event_log() -> None:
    model = _scripted(_text("unused", "nope"))
    loop, state, log, _store = await setup(model=model)
    await collect(loop.handle_caller_turn(state, CLINICAL))
    events = await log.read("call_1")
    uttered = [event for event in events if event.kind == "caller_utterance"]
    assert uttered
    text = uttered[0].payload["text"]
    assert isinstance(text, str)
    assert text.startswith("[redacted:")
    assert "pregnant" not in text
    assert CLINICAL not in json.dumps(
        [{k: str(v) for k, v in event.payload.items()} for event in events]
    )
    assert state.transcript[0].text.startswith("[redacted:")


# --------------------------------------------------------------------------- max_seconds


async def test_duration_limit_weekday_transfers() -> None:
    model = _scripted(_text("unused", "nope"))
    later = TUE + timedelta(seconds=GLOW.max_seconds)
    loop, state, _log, _store = await setup(model=model, started=TUE, now=later)
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert TRANSFER_LINE in [chunk.text for chunk in chunks]
    assert chunks[-1].transfer_number == "+12035550101"
    model.assert_never_called()


async def test_duration_limit_sunday_confirms_then_promise() -> None:
    model = _scripted(_text("unused", "nope"))
    later = SUN + timedelta(seconds=GLOW.max_seconds)
    loop, state, _log, _store = await setup(model=model, started=SUN, now=later)
    first = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert ASK_BEST in [chunk.text for chunk in first]
    assert GLOW.callback_promise not in [chunk.text for chunk in first]
    second = await collect(loop.handle_caller_turn(state, "yes"))
    assert GLOW.callback_promise in [chunk.text for chunk in second]
    assert GOODBYE in [chunk.text for chunk in second]
    assert second[-1].end_call is True
    model.assert_never_called()


async def test_emergency_wins_on_duration_limit_turn() -> None:
    model = _scripted(_text("unused", "nope"))
    later = TUE + timedelta(seconds=GLOW.max_seconds)
    loop, state, _log, _store = await setup(model=model, started=TUE, now=later)
    chunks = await collect(loop.handle_caller_turn(state, EMERGENCY))
    assert [chunk.text for chunk in chunks] == [EMERGENCY_LINE]
    assert chunks[-1].end_call is True
    assert chunks[-1].transfer_number is None
    model.assert_never_called()


# --------------------------------------------------------------------------- mid-turn


async def test_mid_turn_continue_returns_none() -> None:
    loop, state, _log, _store = await setup()
    result = await loop.handle_mid_turn_update(state, "hello there friend")
    assert result is None


async def test_mid_turn_reply_does_not_interrupt() -> None:
    loop, state, _log, _store = await setup()
    chunk = await loop.handle_mid_turn_update(state, IDENTITY)
    assert chunk is not None
    assert chunk.text == GLOW.ai_identity_reply
    assert chunk.no_interruption_allowed is False
    assert chunk.end_call is False


async def test_mid_turn_commit_returns_interrupt_chunk() -> None:
    loop, state, _log, _store = await setup()
    chunk = await loop.handle_mid_turn_update(state, EMERGENCY)
    assert chunk is not None
    assert chunk.text == EMERGENCY_LINE
    assert chunk.no_interruption_allowed is True
    assert chunk.end_call is True
    assert chunk.transfer_number is None


# --------------------------------------------------------------------------- save / imports


async def test_stale_save_hedges() -> None:
    model = _scripted(_text("ok", "Happy to help."))
    loop, state, log, store = await setup(model=model)
    other = await store.get("call_1")
    assert other is not None
    await store.save(other)
    chunks = await collect(loop.handle_caller_turn(state, "hello there friend"))
    assert HEDGE in [chunk.text for chunk in chunks]
    assert "error" in await kinds(log)


def test_turn_module_does_not_import_retell() -> None:
    tree = ast.parse(Path(core.turn.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "retell" not in imported
