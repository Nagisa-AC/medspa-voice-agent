"""What we know about a live call. Data and recorders only; no decisions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
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


@dataclass(frozen=True)
class AgentLine:
    text: str
    origin: Literal["backend", "model"]
    at: datetime


@dataclass(frozen=True)
class OfferedSlot:
    service_id: str
    start: datetime
    provider_id: str | None


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
    # Built by `book` from the booking's intent; never derived from call_id, so a
    # callback after a dropped call collapses to the same key.
    idempotency_keys: set[str] = field(default_factory=set)
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


class SessionStore(Protocol):
    async def create(self, state: CallState) -> None: ...

    async def get(self, call_id: str) -> CallState | None: ...

    async def save(self, state: CallState) -> None: ...

    async def delete(self, call_id: str) -> None: ...


class InMemorySessionStore:
    """Copies on every read and write so a missing `save` loses updates, as Redis would."""

    def __init__(self) -> None:
        self._states: dict[str, CallState] = {}

    async def create(self, state: CallState) -> None:
        if state.call_id in self._states:
            raise CallExistsError(state.call_id)
        self._states[state.call_id] = copy.deepcopy(state)

    async def get(self, call_id: str) -> CallState | None:
        state = self._states.get(call_id)
        return None if state is None else copy.deepcopy(state)

    async def save(self, state: CallState) -> None:
        if state.call_id not in self._states:
            raise UnknownCallError(state.call_id)
        self._states[state.call_id] = copy.deepcopy(state)

    async def delete(self, call_id: str) -> None:
        self._states.pop(call_id, None)
