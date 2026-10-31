"""What we know about a live call. Data and recorders only; no decisions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal, Protocol

from core.tenant import Tenant

# --------------------------------------------------------------------------- transcript


@dataclass(frozen=True)
class CallerLine:
    """One caller utterance, stored exactly as given.

    The turn loop gates an utterance before recording it, so `text` may be a redaction
    marker such as "[redacted:clinical_question]" rather than the caller's words.
    """

    text: str
    at: datetime


HistoryKind = Literal["include", "omit", "marker"]
MarkerKind = Literal["price", "faq"]


@dataclass(frozen=True)
class AgentLine:
    """One spoken agent line. Task 19 tags `history` at write time; the prompt builder reads it.

    A model sentence blocked by the guardrail is never appended; only its replacement is.
    """

    text: str
    origin: Literal["backend", "model"]
    at: datetime
    history: HistoryKind = "include"
    marker: MarkerKind | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.history not in ("include", "omit", "marker"):
            raise ValueError("history must be include, omit, or marker")
        if self.history == "marker":
            if self.marker not in ("price", "faq"):
                raise ValueError("history='marker' requires marker='price' or marker='faq'")
            if self.marker == "price":
                if self.label is None or not self.label.strip():
                    raise ValueError("marker='price' requires a non-empty label")
            elif self.label is not None:
                raise ValueError("marker='faq' forbids label")
            return
        if self.marker is not None:
            raise ValueError(f"history={self.history!r} forbids marker")
        if self.label is not None:
            raise ValueError(f"history={self.history!r} forbids label")


@dataclass(frozen=True)
class OfferedSlot:
    service_id: str
    start: datetime
    end: datetime
    provider_id: str
    room_id: str
    version: int

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("end", self.end)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
            if value.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be UTC")
        if self.end <= self.start:
            raise ValueError("end must be after start")


# --------------------------------------------------------------------------- escalation


@dataclass(frozen=True)
class NotEscalated:
    pass


@dataclass(frozen=True)
class Armed:
    trips: frozenset[str]

    def __post_init__(self) -> None:
        if not self.trips:
            raise ValueError("Armed needs at least one trip")


@dataclass(frozen=True)
class EmergencyEnd:
    pass


@dataclass(frozen=True)
class Transfer:
    tier: int
    target: str


@dataclass(frozen=True)
class Callback:
    pass


EscalationPath = EmergencyEnd | Transfer | Callback


@dataclass(frozen=True)
class Committed:
    trip: str
    path: EscalationPath


Escalation = NotEscalated | Armed | Committed


class EscalationCommittedError(Exception):
    """A committed escalation is terminal; nothing may change it."""


# --------------------------------------------------------------------------- call state


@dataclass(slots=True)
class CallState:
    call_id: str
    tenant_id: str
    started_at: datetime
    data_key_id: str
    transcript: list[CallerLine | AgentLine] = field(default_factory=list)
    allowed_amounts: set[Decimal] = field(default_factory=set)
    allowed_numbers: set[Decimal] = field(default_factory=set)
    offered_slots: list[OfferedSlot] = field(default_factory=list)
    turn_count: int = 0
    language: Literal["en", "es"] = "en"
    caller_name: str | None = None
    caller_number: str | None = None
    caller_number_confirmed: bool = False
    callback_phase: Literal["idle", "ask_best", "ask_digits", "done"] = "idle"
    callback_digit_failures: int = 0
    # Built by `book` from the booking's intent; never derived from call_id, so a
    # callback after a dropped call collapses to the same key.
    idempotency_keys: set[str] = field(default_factory=set)
    version: int = field(default=0, init=False)  # advanced only by SessionStore.save
    _escalation: Escalation = field(default_factory=NotEscalated, init=False)

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("call_id must not be empty")
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware")

    @property
    def escalation(self) -> Escalation:
        return self._escalation

    @property
    def pending_transfer_number(self) -> str | None:
        match self._escalation:
            case Committed(path=Transfer(target=target)):
                return target
            case _:
                return None

    def record_arm(self, trip: str) -> None:
        match self._escalation:
            case Committed():
                raise EscalationCommittedError(self._escalation.trip)
            case Armed(trips):
                self._escalation = Armed(trips | {trip})
            case NotEscalated():
                self._escalation = Armed(frozenset({trip}))

    def record_commit(self, trip: str, path: EscalationPath) -> None:
        if isinstance(self._escalation, Committed):
            raise EscalationCommittedError(self._escalation.trip)
        self._escalation = Committed(trip, path)


# --------------------------------------------------------------------------- numbers

NumberKind = Literal["amount", "price_context", "plain"]


class FoundNumber(Protocol):
    @property
    def value(self) -> Decimal: ...

    @property
    def kind(self) -> NumberKind: ...


NumberExtractor = Callable[[str], Iterable[FoundNumber]]


def admit_numbers(state: CallState, text: str, extract: NumberExtractor) -> None:
    """Allow every number in a backend string; amounts and prices also as amounts."""
    for found in extract(text):
        state.allowed_numbers.add(found.value)
        if found.kind in ("amount", "price_context"):
            state.allowed_amounts.add(found.value)


def new_call_state(
    *,
    tenant: Tenant,
    call_id: str,
    data_key_id: str,
    now: datetime,
    extract: NumberExtractor,
) -> CallState:
    state = CallState(call_id=call_id, tenant_id=tenant.id, started_at=now, data_key_id=data_key_id)
    seeds = [
        tenant.greeting,
        tenant.record_disclosure,
        *tenant.facts.values(),
        *tenant.policies.values(),
        *(s.notes for s in tenant.services if s.notes is not None),
    ]
    for text in seeds:
        admit_numbers(state, text, extract)
    return state


# --------------------------------------------------------------------------- session store


class CallExistsError(Exception):
    pass


class UnknownCallError(Exception):
    pass


class StaleStateError(Exception):
    """The stored state moved on since this copy was read; re-read and retry."""


class SessionStore(Protocol):
    async def create(self, state: CallState) -> None:
        """Store a new call at version 0."""
        ...

    async def get(self, call_id: str) -> CallState | None: ...

    async def save(self, state: CallState) -> None:
        """Compare-and-set: store only if `state.version` matches the stored version.

        On success the stored copy and `state` both move to version + 1; otherwise raise
        StaleStateError and leave the stored state untouched.
        """
        ...

    async def delete(self, call_id: str) -> None: ...


class InMemorySessionStore:
    """Copies on every read and write so a missing `save` loses updates, as Redis would."""

    def __init__(self) -> None:
        self._states: dict[str, CallState] = {}

    async def create(self, state: CallState) -> None:
        if state.call_id in self._states:
            raise CallExistsError(state.call_id)
        state.version = 0
        self._states[state.call_id] = copy.deepcopy(state)

    async def get(self, call_id: str) -> CallState | None:
        state = self._states.get(call_id)
        return None if state is None else copy.deepcopy(state)

    async def save(self, state: CallState) -> None:
        stored = self._states.get(state.call_id)
        if stored is None:
            raise UnknownCallError(state.call_id)
        if stored.version != state.version:
            raise StaleStateError(state.call_id)
        state.version += 1
        self._states[state.call_id] = copy.deepcopy(state)

    async def delete(self, call_id: str) -> None:
        self._states.pop(call_id, None)
