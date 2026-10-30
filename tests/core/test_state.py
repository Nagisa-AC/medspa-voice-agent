import ast
import copy
import dataclasses
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

import core.state
from core.state import (
    AgentLine,
    Armed,
    Callback,
    CallerLine,
    CallExistsError,
    CallState,
    Committed,
    EmergencyEnd,
    EscalationCommittedError,
    InMemorySessionStore,
    NotEscalated,
    NumberKind,
    OfferedSlot,
    StaleStateError,
    Transfer,
    UnknownCallError,
    admit_numbers,
    new_call_state,
)
from core.tenant import Tenant, parse_tenant, read_yaml

BASE = Path(__file__).parent / "fixtures" / "tenants" / "test_spa.yaml"
NOW = datetime(2026, 11, 3, 16, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Found:
    value: Decimal
    kind: NumberKind


def stub_extract(text: str) -> list[Found]:
    """Test stand-in for the real extractor: "$N" is an amount, a bare N is plain."""
    return [
        Found(Decimal(m[2]), "amount" if m[1] else "plain")
        for m in re.finditer(r"(\$)?(\d+(?:\.\d+)?)", text)
    ]


class RecordingExtractor:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str) -> list[Found]:
        self.texts.append(text)
        return []


def tenant(**overrides: Any) -> Tenant:
    data: Any = copy.deepcopy(read_yaml(BASE))
    data.update(overrides)
    return parse_tenant(data, source="test_spa.yaml", today=date(2026, 10, 29))


SEEDED = tenant(
    greeting="Hi, this is assistant 7.",
    record_disclosure="Recorded for 30 days.",
    facts={"hours": "Open 9 to 5, Suite 300."},
    policies={"cancellation": "Cancel 24 hours ahead or pay $40."},
)


def state(**kwargs: Any) -> CallState:
    return new_call_state(
        tenant=kwargs.pop("tenant", SEEDED),
        call_id=kwargs.pop("call_id", "call_1"),
        data_key_id="key_1",
        now=NOW,
        extract=kwargs.pop("extract", stub_extract),
    )


# --------------------------------------------------------------------------- creation and seeding


def test_new_state_starts_empty_except_seeded_numbers() -> None:
    s = state()
    assert (s.call_id, s.tenant_id, s.started_at, s.data_key_id) == (
        "call_1",
        "test_spa",
        NOW,
        "key_1",
    )
    assert s.transcript == []
    assert s.offered_slots == []
    assert s.idempotency_keys == set()
    assert s.turn_count == 0
    assert s.language == "en"
    assert s.caller_name is None
    assert s.caller_number is None
    assert s.caller_number_confirmed is False
    assert s.escalation == NotEscalated()
    assert s.pending_transfer_number is None


def test_seeding_uses_every_seed_source() -> None:
    s = state()
    assert s.allowed_amounts == {Decimal(40)}
    assert s.allowed_numbers == {Decimal(n) for n in (7, 30, 9, 5, 300, 24, 40)}


def test_seed_texts_in_order_and_nothing_else() -> None:
    notes = tenant(
        services=[
            {**svc, "notes": "Takes 15 minutes."} if svc["id"] == "alpha" else svc
            for svc in read_yaml_services()
        ]
    )
    recorder = RecordingExtractor()
    state(tenant=notes, extract=recorder)
    assert recorder.texts == [
        notes.greeting,
        notes.record_disclosure,
        *notes.facts.values(),
        *notes.policies.values(),
        "Takes 15 minutes.",
    ]


def read_yaml_services() -> list[dict[str, Any]]:
    data: Any = read_yaml(BASE)
    return list(data["services"])


def test_prices_faq_and_other_scripts_are_not_seeded() -> None:
    recorder = RecordingExtractor()
    state(extract=recorder)
    for text in (
        SEEDED.ai_identity_reply,
        SEEDED.reminder_line,
        SEEDED.callback_promise,
        SEEDED.faq[0].say,
    ):
        assert text not in recorder.texts
    s = state()
    assert Decimal("20.50") not in s.allowed_amounts  # a service price
    assert Decimal(25) not in s.allowed_amounts  # a deposit amount


@pytest.mark.parametrize(
    ("kind", "in_amounts"), [("amount", True), ("price_context", True), ("plain", False)]
)
def test_admit_numbers_by_kind(kind: NumberKind, in_amounts: bool) -> None:
    s = state(extract=RecordingExtractor())

    def extract(text: str) -> Iterable[Found]:
        return [Found(Decimal("14"), kind)]

    admit_numbers(s, "anything", extract)
    assert Decimal(14) in s.allowed_numbers
    assert (Decimal(14) in s.allowed_amounts) is in_amounts


def test_decimal_formatting_does_not_matter() -> None:
    s = state()
    assert Decimal("40.00") in s.allowed_amounts


def test_started_at_must_be_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        CallState(
            call_id="c",
            tenant_id="t",
            started_at=datetime(2026, 11, 3, 16, 0),  # noqa: DTZ001
            data_key_id="k",
        )


def test_call_id_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="call_id"):
        CallState(call_id="", tenant_id="t", started_at=NOW, data_key_id="k")


# --------------------------------------------------------------------------- escalation


def test_arming_accumulates_and_never_commits() -> None:
    s = state()
    s.record_arm("hostility")
    s.record_arm("hostility")
    assert s.escalation == Armed(frozenset({"hostility"}))
    s.record_arm("prompt_injection")
    assert s.escalation == Armed(frozenset({"hostility", "prompt_injection"}))


@pytest.mark.parametrize("armed_first", [False, True])
def test_commit_from_not_escalated_or_armed(armed_first: bool) -> None:
    s = state()
    if armed_first:
        s.record_arm("hostility")
    s.record_commit("legal_or_complaint", Callback())
    assert s.escalation == Committed("legal_or_complaint", Callback())


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.record_arm("hostility"),
        lambda s: s.record_commit("medical_emergency", EmergencyEnd()),
    ],
)
def test_committed_is_terminal(change: Any) -> None:
    s = state()
    s.record_commit("legal_or_complaint", Transfer(1, "+15555550101"))
    before = s.escalation
    with pytest.raises(EscalationCommittedError):
        change(s)
    assert s.escalation == before


def test_armed_needs_a_trip() -> None:
    with pytest.raises(ValueError):
        Armed(frozenset())


def test_escalation_cannot_be_assigned_directly() -> None:
    s = state()
    with pytest.raises(AttributeError):
        s.escalation = Committed("x", Callback())  # type: ignore[misc]


@pytest.mark.parametrize(
    ("path", "number"),
    [
        (Transfer(1, "+15555550101"), "+15555550101"),
        (Callback(), None),
        (EmergencyEnd(), None),
    ],
)
def test_pending_transfer_number_is_derived(path: Any, number: str | None) -> None:
    s = state()
    s.record_commit("legal_or_complaint", path)
    assert s.pending_transfer_number == number


def test_recorder_accepts_any_path_it_is_given() -> None:
    # Choosing a path (and checking it against the tenant) is the executor's job.
    s = state()
    s.record_commit("x", Transfer(9, "+15555550100"))
    assert s.pending_transfer_number == "+15555550100"


# --------------------------------------------------------------------------- session store


async def test_store_create_get_save_delete() -> None:
    store = InMemorySessionStore()
    s = state()
    await store.create(s)
    with pytest.raises(CallExistsError):
        await store.create(s)

    loaded = await store.get("call_1")
    assert loaded == s
    assert loaded is not s

    loaded.turn_count = 3
    loaded.transcript.append(CallerLine("[redacted:clinical_question]", NOW))
    loaded.record_arm("hostility")
    await store.save(loaded)
    again = await store.get("call_1")
    assert again is not None
    assert again.turn_count == 3
    assert again.transcript == [CallerLine("[redacted:clinical_question]", NOW)]
    assert again.escalation == Armed(frozenset({"hostility"}))

    await store.delete("call_1")
    await store.delete("call_1")
    assert await store.get("call_1") is None


async def test_store_missing_call() -> None:
    store = InMemorySessionStore()
    assert await store.get("nope") is None
    with pytest.raises(UnknownCallError):
        await store.save(state(call_id="nope"))


async def test_store_mutation_without_save_is_lost() -> None:
    store = InMemorySessionStore()
    s = state()
    await store.create(s)
    s.turn_count = 5
    s.allowed_amounts.add(Decimal(999))
    loaded = await store.get("call_1")
    assert loaded is not None
    loaded.offered_slots.append(
        OfferedSlot(
            service_id="alpha",
            start=NOW,
            end=NOW + timedelta(minutes=30),
            provider_id="p1",
            room_id="r1",
            version=0,
        )
    )
    again = await store.get("call_1")
    assert again is not None
    assert again.turn_count == 0
    assert Decimal(999) not in again.allowed_amounts
    assert again.offered_slots == []


def test_version_is_not_a_constructor_argument() -> None:
    assert "version" not in {f.name for f in dataclasses.fields(CallState) if f.init}
    assert state().version == 0


async def test_create_stores_version_zero() -> None:
    store = InMemorySessionStore()
    s = state()
    s.version = 7
    await store.create(s)
    loaded = await store.get("call_1")
    assert loaded is not None
    assert (s.version, loaded.version) == (0, 0)


async def test_second_save_from_a_stale_copy_is_rejected() -> None:
    store = InMemorySessionStore()
    await store.create(state())
    first = await store.get("call_1")
    second = await store.get("call_1")
    assert first is not None and second is not None
    await store.save(first)
    with pytest.raises(StaleStateError):
        await store.save(second)
    stored = await store.get("call_1")
    assert stored is not None
    assert stored.version == 1


async def test_stale_save_cannot_erase_a_commit() -> None:
    store = InMemorySessionStore()
    await store.create(state())
    a = await store.get("call_1")
    b = await store.get("call_1")
    assert a is not None and b is not None
    a.record_commit("legal_or_complaint", Transfer(1, "+15555550101"))
    await store.save(a)
    b.turn_count += 1
    with pytest.raises(StaleStateError):
        await store.save(b)
    stored = await store.get("call_1")
    assert stored is not None
    assert stored.escalation == Committed("legal_or_complaint", Transfer(1, "+15555550101"))
    assert stored.turn_count == 0


async def test_repeated_saves_from_one_object_advance_the_version() -> None:
    store = InMemorySessionStore()
    s = state()
    await store.create(s)
    await store.save(s)
    assert s.version == 1
    s.turn_count = 2
    await store.save(s)
    assert s.version == 2
    stored = await store.get("call_1")
    assert stored is not None
    assert (stored.version, stored.turn_count) == (2, 2)


async def test_failed_save_leaves_caller_version_unchanged() -> None:
    store = InMemorySessionStore()
    await store.create(state())
    a = await store.get("call_1")
    b = await store.get("call_1")
    assert a is not None and b is not None
    await store.save(a)
    with pytest.raises(StaleStateError):
        await store.save(b)
    assert b.version == 0


# --------------------------------------------------------------------------- no decision logic

FIELDS = {
    "call_id",
    "tenant_id",
    "started_at",
    "data_key_id",
    "transcript",
    "allowed_amounts",
    "allowed_numbers",
    "offered_slots",
    "turn_count",
    "language",
    "caller_name",
    "caller_number",
    "caller_number_confirmed",
    "idempotency_keys",
    "version",
}


def test_call_state_public_surface_is_data_and_recorders_only() -> None:
    public = {name for name in dir(CallState) if not name.startswith("_")}
    assert public == FIELDS | {
        "escalation",
        "pending_transfer_number",
        "record_arm",
        "record_commit",
    }


def test_module_has_no_helper_that_records_caller_text() -> None:
    # Caller text enters the transcript only via the turn loop, after the gate.
    functions = {
        name
        for name, value in vars(core.state).items()
        if callable(value)
        and getattr(value, "__module__", None) == "core.state"
        and not isinstance(value, type)
    }
    assert functions == {"admit_numbers", "new_call_state"}


def test_state_module_imports_no_decision_code() -> None:
    tree = ast.parse(Path(core.state.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections.abc",
        "copy",
        "core.tenant",
        "dataclasses",
        "datetime",
        "decimal",
        "typing",
    }


def test_agent_lines_keep_origin() -> None:
    line = AgentLine("Hello.", "backend", NOW)
    assert line.origin == "backend"
