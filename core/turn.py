"""Turn loop: gate, speak, tools, guardrails. Does not talk to Retell.

A model sentence blocked by the guardrail is never appended to the transcript;
only its replacement is.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, Literal

from core.escalation import EMERGENCY_LINE, TRANSFER_LINE, EscalationPlan, plan
from core.events import EventDraft, EventKind, EventLog, record_error
from core.guardrail import (
    Blocked,
    OutputPack,
    Pass,
    SentenceBuffer,
    check_sentence,
)
from core.guardrail import (
    NumberExtractor as GuardExtractor,
)
from core.llm import (
    Done,
    FirstTokenTimeout,
    Message,
    Model,
    StreamStalled,
    TextDelta,
    ToolCall,
    ToolCallParseError,
)
from core.policy import Arm, Commit, Continue, GateResult, PolicyEngine, Reply, normalize_utterance
from core.prompt import TOOLS, BuiltPrompt, build, tool_result_messages
from core.state import (
    AgentLine,
    Callback,
    CallerLine,
    CallState,
    Committed,
    EmergencyEnd,
    HistoryKind,
    MarkerKind,
    SessionStore,
    StaleStateError,
    Transfer,
    admit_numbers,
)
from core.tenant import Tenant
from integrations.booking import BookingAdapter
from tools.faq import LookupFaqResult, lookup_faq
from tools.pricing import QuotePriceResult, quote_price
from tools.scheduling import (
    BookResult,
    CheckAvailabilityResult,
    book,
    check_availability,
    normalize_phone,
)

Clock = Callable[[], datetime]
Origin = Literal["backend", "model"]

MAX_OUTPUT_TOKENS: Final = 256
FIRST_TOKEN_TIMEOUT: Final = 2.0
IDLE_TIMEOUT: Final = 2.0
MAX_TOOL_ROUNDS: Final = 3
_MIN_SECRET: Final = 32

HEDGE = "Sorry, I missed that. Could you say it one more time?"
NUMBER_HEDGE = "Let me get you an exact number on that."
GOODBYE = "Thanks for calling. Goodbye."
ASK_BEST = "Is the number you're calling from the best one to reach you?"
ASK_DIGITS = "What number should we use? Area code first, please."
READBACK = "I'll have them call you at {spoken}."
BAD_NUMBER = "I didn't catch that. One more time, area code first?"
GIVE_UP = "I'll have the team follow up with the number we have. Thanks for calling. Goodbye."

_YES_FIRST = frozenset({"yes", "yeah", "yep", "yup", "sure", "correct"})
_NO_FIRST = frozenset({"no", "nope"})
_NO_TOKENS = frozenset({"no", "nope"})
_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)


ToolResult = QuotePriceResult | LookupFaqResult | CheckAvailabilityResult | BookResult


@dataclass(frozen=True)
class OutputChunk:
    text: str
    origin: Origin
    content_complete: bool = False
    end_call: bool = False
    transfer_number: str | None = None
    no_interruption_allowed: bool = False


class TurnLoop:
    def __init__(
        self,
        tenant: Tenant,
        store: SessionStore,
        log: EventLog,
        model: Model,
        adapter: BookingAdapter,
        secret: bytes,
        *,
        clock: Clock,
        engine: PolicyEngine,
        pack: OutputPack,
        extract: GuardExtractor,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
        idle_timeout: float = IDLE_TIMEOUT,
    ) -> None:
        if len(secret) < _MIN_SECRET:
            raise ValueError("secret must be at least 32 bytes")
        self.tenant = tenant
        self.store = store
        self.log = log
        self.model = model
        self.adapter = adapter
        self.secret = secret
        self.clock = clock
        self.engine = engine
        self.pack = pack
        self.extract = extract
        self.max_output_tokens = max_output_tokens
        self.first_token_timeout = first_token_timeout
        self.idle_timeout = idle_timeout

    async def connect(self, state: CallState) -> AsyncIterator[OutputChunk]:
        session = _Session(self, state)
        await session.speak(self.tenant.greeting, "backend", history="include")
        await session.save()
        async for chunk in session.flush():
            yield chunk

    async def handle_caller_turn(
        self, state: CallState, utterance: str
    ) -> AsyncIterator[OutputChunk]:
        session = _Session(self, state)
        try:
            await session.caller_turn(utterance)
        except Exception:
            await record_error(
                self.log,
                call_id=state.call_id,
                tenant_id=state.tenant_id,
                at=self.clock(),
                code="hedge",
                where="turn",
            )
            await session.speak(HEDGE, "backend", history="include")
        await session.save()
        async for chunk in session.flush():
            yield chunk

    async def handle_mid_turn_update(self, state: CallState, utterance: str) -> OutputChunk | None:
        session = _Session(self, state)
        result = await session.record_utterance(utterance)
        if isinstance(state.escalation, Committed):
            await session.replay_committed(utterance)
        elif isinstance(result, Commit):
            await session.first_commit(result)
        elif isinstance(result, Reply):
            await session.speak(result.script, "backend", history="include")
        elif isinstance(result, Arm):
            state.record_arm(result.trip)
        await session.save()
        if session.chunks:
            chunks = [chunk async for chunk in session.flush()]
            return chunks[-1]
        return None


class _Session:
    def __init__(self, loop: TurnLoop, state: CallState) -> None:
        self.loop = loop
        self.state = state
        self.chunks: list[OutputChunk] = []
        self.end_call = False
        self.transfer_number: str | None = None
        self.interrupt = False
        self.ended: str | None = None
        self.extra: list[Message] = []
        self.rounds = 0
        self.turn_start = 0
        self.passed_model: list[str] = []
        self.round_payloads: list[tuple[ToolCall, Mapping[str, object]]] = []

    async def caller_turn(self, utterance: str) -> None:
        result = await self.record_utterance(utterance)
        self.turn_start = len(self.state.transcript)
        if isinstance(self.state.escalation, Committed):
            await self.replay_committed(utterance)
            return
        if isinstance(result, Commit):
            await self.first_commit(result)
            return
        if isinstance(result, Reply):
            await self.speak(result.script, "backend", history="include")
            return
        if isinstance(result, Arm):
            self.state.record_arm(result.trip)
        await self.model_turn()

    async def record_utterance(self, utterance: str) -> GateResult:
        self.state.turn_count += 1
        now = self.loop.clock()
        result = self.loop.engine.evaluate(utterance, self.state, now=now)
        clinical = False
        trip: str | None = None
        if not isinstance(result, Continue):
            clinical = result.clinical_class
            trip = result.trip
        stored = await self.loop.log.record_caller_utterance(
            call_id=self.state.call_id,
            tenant_id=self.state.tenant_id,
            at=now,
            turn=self.state.turn_count,
            text=utterance,
            clinical_class=clinical,
            trip=trip,
        )
        self.state.transcript.append(CallerLine(stored, now))
        return result

    async def first_commit(self, result: Commit) -> None:
        now = self.loop.clock()
        first = not isinstance(self.state.escalation, Committed)
        planned = plan(self.loop.tenant, self.state, result, now)
        if first:
            await self._log_commit(result.trip, planned)
        await self._run_path(planned)

    async def replay_committed(self, utterance: str) -> None:
        escalation = self.state.escalation
        assert isinstance(escalation, Committed)
        match escalation.path:
            case Callback():
                await self.callback_confirm(utterance)
            case Transfer():
                self.transfer_number = self.state.pending_transfer_number
                self.interrupt = True
                await self.speak(TRANSFER_LINE, "backend", history="include")
            case EmergencyEnd():
                self.interrupt = True
                await self.speak(EMERGENCY_LINE, "backend", history="omit")
                await self._end("emergency")

    async def _run_path(self, planned: EscalationPlan) -> None:
        self.interrupt = True
        match planned.path:
            case EmergencyEnd():
                await self.speak(EMERGENCY_LINE, "backend", history="omit")
                await self._end("emergency")
            case Transfer():
                self.transfer_number = self.state.pending_transfer_number
                await self.speak(TRANSFER_LINE, "backend", history="include")
            case Callback():
                await self.start_confirm()

    async def start_confirm(self) -> None:
        if self.state.caller_number:
            self.state.callback_phase = "ask_best"
            await self.speak(ASK_BEST, "backend", history="include")
            return
        self.state.callback_phase = "ask_digits"
        await self.speak(ASK_DIGITS, "backend", history="include")

    async def callback_confirm(self, utterance: str) -> None:
        self.interrupt = True
        phase = self.state.callback_phase
        folded = normalize_utterance(utterance)
        if phase == "done":
            await self.speak(GOODBYE, "backend", history="include")
            await self._end("callback")
            return
        if phase == "ask_best":
            if _is_yes(folded):
                self.state.caller_number_confirmed = True
                await self._finish_callback()
                return
            if _is_no(folded):
                self.state.callback_phase = "ask_digits"
                await self.speak(ASK_DIGITS, "backend", history="include")
                return
            phone = normalize_phone(utterance)
            if phone is not None:
                await self._accept_phone(phone)
                return
            self.state.callback_phase = "ask_digits"
            await self.speak(ASK_DIGITS, "backend", history="include")
            return
        phone = normalize_phone(utterance)
        if phone is not None:
            await self._accept_phone(phone)
            return
        self.state.callback_digit_failures += 1
        if self.state.callback_digit_failures >= 2:
            self.state.callback_phase = "done"
            await self.speak(GIVE_UP, "backend", history="include")
            await self._end("callback")
            return
        await self.speak(BAD_NUMBER, "backend", history="include")

    async def _accept_phone(self, phone: str) -> None:
        self.state.caller_number = phone
        self.state.caller_number_confirmed = True
        await self.speak(READBACK.format(spoken=_speak_phone(phone)), "backend", history="include")
        await self._finish_callback()

    async def _finish_callback(self) -> None:
        self.state.callback_phase = "done"
        await self.speak(self.loop.tenant.callback_promise, "backend", history="omit")
        await self.speak(GOODBYE, "backend", history="include")
        await self._end("callback")

    async def model_turn(self) -> None:
        try:
            await self._stream_once()
        except (FirstTokenTimeout, StreamStalled, ToolCallParseError):
            await record_error(
                self.loop.log,
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                code="hedge",
                where="turn",
            )
            await self.speak(HEDGE, "backend", history="include")

    async def _stream_once(self) -> None:
        self.round_payloads = []
        built = self._prompt()
        stream = self.loop.model.stream(
            built.messages,
            built.tools,
            max_output_tokens=self.loop.max_output_tokens,
            first_token_timeout=self.loop.first_token_timeout,
            idle_timeout=self.loop.idle_timeout,
        )
        buffer = SentenceBuffer()
        tools: list[ToolCall] = []
        blocked = False
        try:
            async for event in stream:
                if isinstance(event, TextDelta):
                    if tools:
                        continue
                    for sentence in buffer.feed(event.text):
                        if not await self._model_sentence(sentence):
                            blocked = True
                            break
                    if blocked:
                        break
                elif isinstance(event, ToolCall):
                    for sentence in buffer.flush():
                        if not await self._model_sentence(sentence):
                            blocked = True
                            break
                    if blocked:
                        break
                    tools.append(event)
                elif isinstance(event, Done):
                    break
            if not blocked:
                for sentence in buffer.flush():
                    if not await self._model_sentence(sentence):
                        blocked = True
                        break
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        if blocked:
            return
        if not tools:
            return
        for call in tools:
            if self.rounds >= MAX_TOOL_ROUNDS:
                await self.speak(HEDGE, "backend", history="include")
                return
            if not await self._dispatch(call):
                return
            self.rounds += 1
        self._remember_round()
        await self._stream_once()

    async def _model_sentence(self, sentence: str) -> bool:
        if not sentence.strip():
            return True
        checked = check_sentence(
            sentence,
            self.loop.pack,
            allowed_amounts=self.state.allowed_amounts,
            allowed_numbers=self.state.allowed_numbers,
            extract=self.loop.extract,
        )
        if isinstance(checked, Pass):
            self.passed_model.append(sentence)
            await self.speak(sentence, "model", history="include")
            return True
        assert isinstance(checked, Blocked)
        replacement = self.loop.pack.deflect if checked.check == "clinical" else NUMBER_HEDGE
        await self.speak(replacement, "backend", history="include")
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="guardrail_blocked",
                payload={"check": checked.check, "pattern_ids": list(checked.pattern_ids)},
            )
        )
        return False

    async def _dispatch(self, call: ToolCall) -> bool:
        name = call.name
        if name not in {tool.name for tool in TOOLS}:
            await self.speak(HEDGE, "backend", history="include")
            return False
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="tool_invoked",
                payload={"tool": name, "args": json.dumps(_jsonable(call.args), sort_keys=True)},
            )
        )
        try:
            result = await self._run_tool(call)
        except (KeyError, TypeError, ValueError):
            await self.speak(HEDGE, "backend", history="include")
            return False
        history, marker, label = _tool_tag(name, result.model, self.loop.tenant)
        await self.speak(result.say, "backend", history=history, marker=marker, label=label)
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="tool_result",
                payload={
                    "tool": name,
                    "result": json.dumps(_jsonable(result.model), sort_keys=True),
                },
            )
        )
        if name == "lookup_faq" and result.model.get("found") is True:
            faq_id = result.model.get("faq_id")
            if isinstance(faq_id, str):
                await self.loop.log.append(
                    EventDraft(
                        call_id=self.state.call_id,
                        tenant_id=self.state.tenant_id,
                        at=self.loop.clock(),
                        kind="faq_served",
                        payload={"faq_id": faq_id},
                    )
                )
        if name == "book":
            await self._booking_events(call, result)
        self.round_payloads.append((call, result.model))
        return True

    async def _run_tool(self, call: ToolCall) -> ToolResult:
        args = call.args
        now = self.loop.clock()
        extract = self.loop.extract
        tenant = self.loop.tenant
        state = self.state
        if call.name == "quote_price":
            return quote_price(tenant, state, str(args["service_text"]), extract=extract)
        if call.name == "lookup_faq":
            return lookup_faq(
                tenant,
                state,
                str(args["question_text"]),
                deflect=self.loop.pack.deflect,
                extract=extract,
            )
        if call.name == "check_availability":
            raw_limit = args.get("limit", 3)
            if isinstance(raw_limit, int) and not isinstance(raw_limit, bool):
                limit = raw_limit
            else:
                limit = 3
            earliest = args.get("earliest_iso")
            return await check_availability(
                tenant,
                state,
                self.loop.adapter,
                str(args["service_text"]),
                earliest_iso=earliest if isinstance(earliest, str) else None,
                extract=extract,
                now=now,
                limit=limit,
            )
        return await book(
            tenant,
            state,
            self.loop.adapter,
            slot_start_iso=str(args["slot_start_iso"]),
            patient_name=str(args["patient_name"]),
            patient_phone=str(args["patient_phone"]),
            confirmed=bool(args["confirmed"]),
            extract=extract,
            now=now,
            secret=self.loop.secret,
        )

    async def _booking_events(self, call: ToolCall, result: ToolResult) -> None:
        if not isinstance(result, BookResult):
            return
        model = result.model
        args = call.args
        service_id = result.service_id or ""
        slot_start = str(args.get("slot_start_iso", ""))
        name = str(args.get("patient_name", ""))
        phone = str(args.get("patient_phone", ""))
        confirmed = bool(args.get("confirmed", False))
        base = {
            "patient_name": name,
            "patient_phone": phone,
            "service_id": service_id,
            "slot_start": slot_start,
        }
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="booking_attempted",
                payload={**base, "confirmed": confirmed},
            )
        )
        if model.get("booked"):
            booking_id = result.booking_id or ""
            await self.loop.log.append(
                EventDraft(
                    call_id=self.state.call_id,
                    tenant_id=self.state.tenant_id,
                    at=self.loop.clock(),
                    kind="booking_confirmed",
                    payload={**base, "booking_id": booking_id, "confirmed": confirmed},
                )
            )
            return
        if model.get("needs_review"):
            await self.loop.log.append(
                EventDraft(
                    call_id=self.state.call_id,
                    tenant_id=self.state.tenant_id,
                    at=self.loop.clock(),
                    kind="booking_needs_review",
                    payload={**base, "reason": "review"},
                )
            )
            return
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="booking_failed",
                payload={**base, "reason": "retry" if model.get("retry") else "refused"},
            )
        )

    def _prompt(self) -> BuiltPrompt:
        saved = self.state.transcript
        self.state.transcript = saved[: self.turn_start]
        try:
            return build(self.loop.tenant, self.state, extra=tuple(self.extra))
        finally:
            self.state.transcript = saved

    def _remember_round(self) -> None:
        content = "".join(self.passed_model)
        self.passed_model.clear()
        calls = tuple(call for call, _payload in self.round_payloads)
        self.extra.append(Message(role="assistant", content=content, tool_calls=calls))
        for call, payload in self.round_payloads:
            self.extra.append(tool_result_messages(call, payload)[1])
        self.round_payloads = []

    async def speak(
        self,
        text: str,
        origin: Origin,
        *,
        history: HistoryKind,
        marker: MarkerKind | None = None,
        label: str | None = None,
        interrupt: bool = False,
    ) -> None:
        if origin == "backend":
            admit_numbers(self.state, text, self.loop.extract)
        self.state.transcript.append(
            AgentLine(
                text,
                origin,
                self.loop.clock(),
                history=history,
                marker=marker,
                label=label,
            )
        )
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="agent_sentence",
                payload={"text": text, "origin": origin, "turn": self.state.turn_count},
            )
        )
        self.chunks.append(
            OutputChunk(
                text=text,
                origin=origin,
                no_interruption_allowed=interrupt or self.interrupt,
                transfer_number=self.transfer_number,
            )
        )

    async def save(self) -> None:
        try:
            await self.loop.store.save(self.state)
        except StaleStateError:
            await record_error(
                self.loop.log,
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                code="stale_state",
                where="save",
            )
            await self.speak(HEDGE, "backend", history="include")

    async def flush(self) -> AsyncIterator[OutputChunk]:
        if not self.chunks:
            return
        last = len(self.chunks) - 1
        for index, chunk in enumerate(self.chunks):
            yield replace(
                chunk,
                content_complete=index == last,
                end_call=self.end_call,
                transfer_number=self.transfer_number or chunk.transfer_number,
                no_interruption_allowed=chunk.no_interruption_allowed or self.interrupt,
            )

    async def _log_commit(self, trip: str, planned: EscalationPlan) -> None:
        path = _path_label(planned.path)
        fired: dict[str, str | int | bool | list[str]] = {"trip": trip, "path": path}
        chosen: dict[str, str | int | bool | list[str]] = dict(fired)
        if isinstance(planned.path, Transfer):
            chosen["tier"] = planned.path.tier
        now = self.loop.clock()
        kinds: tuple[tuple[EventKind, dict[str, str | int | bool | list[str]]], ...] = (
            ("tripwire_fired", fired),
            ("escalation_committed", chosen),
            ("escalation_path_chosen", chosen),
        )
        for kind, body in kinds:
            await self.loop.log.append(
                EventDraft(
                    call_id=self.state.call_id,
                    tenant_id=self.state.tenant_id,
                    at=now,
                    kind=kind,
                    payload=body,
                )
            )
        sms: dict[str, str | int | bool | list[str]] = {"category": planned.summary.category}
        if planned.summary.caller_name is not None:
            sms["patient_name"] = planned.summary.caller_name
        if planned.summary.callback_number is not None:
            sms["callback_number"] = planned.summary.callback_number
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=now,
                kind="sms_logged",
                payload=sms,
            )
        )

    async def _end(self, reason: str) -> None:
        self.end_call = True
        if self.ended is not None:
            return
        self.ended = reason
        await self.loop.log.append(
            EventDraft(
                call_id=self.state.call_id,
                tenant_id=self.state.tenant_id,
                at=self.loop.clock(),
                kind="call_ended",
                payload={"reason": reason},
            )
        )


def _is_yes(folded: str) -> bool:
    words = folded.split()
    if not words or words[0] not in _YES_FIRST:
        return False
    return all(word not in _NO_TOKENS for word in words)


def _is_no(folded: str) -> bool:
    words = folded.split()
    if words and words[0] in _NO_FIRST:
        return True
    return "different" in words or "new number" in folded


def _path_label(path: EmergencyEnd | Transfer | Callback) -> str:
    if isinstance(path, EmergencyEnd):
        return "emergency_end"
    if isinstance(path, Transfer):
        return "transfer"
    return "callback"


def _tool_tag(
    name: str, model: Mapping[str, object], tenant: Tenant
) -> tuple[HistoryKind, MarkerKind | None, str | None]:
    if name == "quote_price":
        service_id = model.get("service_id")
        if model.get("found") is True and isinstance(service_id, str):
            for service in tenant.services:
                if service.id == service_id and service.quotable:
                    return "marker", "price", service.name
        return "include", None, None
    if name == "lookup_faq":
        return "marker", "faq", None
    if name == "book":
        return "omit", None, None
    return "include", None, None


def _speak_phone(e164: str) -> str:
    if e164.startswith("+1") and len(e164) == 12 and e164[2:].isdigit():
        digits = e164[2:]
        return f"{_words(digits[:3])}, {_words(digits[3:6])}, {_words(digits[6:])}"
    return " ".join(_DIGIT_WORDS[int(char)] for char in e164 if char.isdigit())


def _words(digits: str) -> str:
    return " ".join(_DIGIT_WORDS[int(char)] for char in digits)


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
