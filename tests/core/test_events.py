import ast
import json
import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import pytest

import core.crypto
import core.events
from core.crypto import (
    Ciphertext,
    InMemoryKeyStore,
    InvalidCiphertextError,
    Shredded,
    bind_aad,
)
from core.events import (
    EVENT_KINDS,
    OPTIONAL_FIELDS,
    PAYLOAD_FIELDS,
    SENSITIVE_FIELDS,
    CallerUtteranceMustUseHookError,
    EventDraft,
    EventKind,
    InMemoryEventLog,
    JsonlEventLog,
    MissingPayloadFieldError,
    TruncatedEventLogError,
    UnknownEventKindError,
    UnknownPayloadFieldError,
    record_error,
    redact_if_clinical,
    shred_call,
)

NOW = datetime(2026, 11, 3, 16, 0, tzinfo=UTC)
SECRET = "CLINICAL_SECRET_PHRASE_DO_NOT_STORE"


@pytest.fixture
def keys() -> InMemoryKeyStore:
    return InMemoryKeyStore()


async def ready(keys: InMemoryKeyStore, call_id: str = "call_1") -> str:
    return await keys.create(call_id)


def draft(
    *,
    kind: EventKind = "call_ended",
    payload: dict[str, str | int | bool | list[str]] | None = None,
    call_id: str = "call_1",
    tenant_id: str = "tenant_a",
) -> EventDraft:
    return EventDraft(
        call_id=call_id,
        tenant_id=tenant_id,
        at=NOW,
        kind=kind,
        payload=dict({"reason": "goodbye"} if payload is None else payload),
    )


async def test_field_tables_cover_every_kind_exactly() -> None:
    kinds = set(get_args(EventKind))
    assert kinds == EVENT_KINDS
    assert set(PAYLOAD_FIELDS) == kinds
    assert set(SENSITIVE_FIELDS) == kinds
    assert set(OPTIONAL_FIELDS) == kinds
    for kind in kinds:
        assert SENSITIVE_FIELDS[kind] <= PAYLOAD_FIELDS[kind]
        assert OPTIONAL_FIELDS[kind] <= PAYLOAD_FIELDS[kind]


async def test_in_memory_and_jsonl_read_back_in_order(
    keys: InMemoryKeyStore, tmp_path: Path
) -> None:
    await ready(keys)
    memory = InMemoryEventLog(keys)
    jsonl = JsonlEventLog(tmp_path / "events.jsonl", keys)
    first = draft(kind="faq_served", payload={"faq_id": "tox_duration"})
    second = draft(kind="call_ended", payload={"reason": "goodbye"})
    for log in (memory, jsonl):
        await log.append(first)
        await log.append(second)
        events = await log.read()
        assert [event.kind for event in events] == ["faq_served", "call_ended"]
        assert events[0].payload["faq_id"] == "tox_duration"
        filtered = await log.read("call_1")
        assert len(filtered) == 2
        assert await log.read("other") == ()


async def test_shred_leaves_kinds_and_timings_readable(keys: InMemoryKeyStore) -> None:
    await ready(keys)
    log = InMemoryEventLog(keys)
    stored = await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text="hello there",
        clinical_class=False,
    )
    assert stored == "hello there"
    await shred_call(keys, log, call_id="call_1", tenant_id="tenant_a", at=NOW, reason="retention")
    events = await log.read()
    assert [event.kind for event in events] == ["caller_utterance", "call_data_shredded"]
    assert events[0].at == NOW
    assert events[0].payload["turn"] == 1
    assert events[0].payload["text"] == Shredded()
    assert events[1].payload["reason"] == "retention"


async def test_unknown_field_fails_closed_and_writes_nothing(
    keys: InMemoryKeyStore, tmp_path: Path
) -> None:
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    with pytest.raises(UnknownPayloadFieldError) as info:
        await log.append(
            draft(
                kind="guardrail_blocked",
                payload={"check": "number", "pattern_ids": ["p1"], "text": "nope"},
            )
        )
    assert info.value.field == "text"
    assert not path.exists()


async def test_unknown_kind_fails(keys: InMemoryKeyStore) -> None:
    log = InMemoryEventLog(keys)
    with pytest.raises(UnknownEventKindError):
        await log.append(draft(kind="not_a_kind"))  # type: ignore[arg-type]


async def test_missing_required_field_fails(keys: InMemoryKeyStore) -> None:
    log = InMemoryEventLog(keys)
    with pytest.raises(MissingPayloadFieldError):
        await log.append(draft(kind="faq_served", payload={}))


async def test_naive_datetime_rejected(keys: InMemoryKeyStore) -> None:
    log = InMemoryEventLog(keys)
    with pytest.raises(ValueError, match="timezone-aware"):
        await log.append(
            EventDraft(
                call_id="call_1",
                tenant_id="tenant_a",
                at=datetime(2026, 11, 3, 16, 0),  # noqa: DTZ001
                kind="call_ended",
                payload={"reason": "goodbye"},
            )
        )


async def test_append_caller_utterance_must_use_hook(keys: InMemoryKeyStore) -> None:
    await ready(keys)
    log = InMemoryEventLog(keys)
    with pytest.raises(CallerUtteranceMustUseHookError):
        await log.append(draft(kind="caller_utterance", payload={"text": "hi", "turn": 1}))


async def test_jsonl_open_is_append_only(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    modes: list[str] = []
    real_open = Path.open

    def spy(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        modes.append(mode)
        return real_open(self, mode, *args, **kwargs)  # type: ignore[call-overload]

    with patch.object(Path, "open", spy):
        await log.append(draft())
    assert modes == ["ab"]


async def test_jsonl_fsyncs_each_append(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    await ready(keys)
    log = JsonlEventLog(tmp_path / "events.jsonl", keys)
    calls: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    with patch.object(os, "fsync", spy):
        await log.append(draft())
    assert calls


async def test_truncated_jsonl_raises(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"call_id":"call_1"')
    log = JsonlEventLog(path, keys)
    with pytest.raises(TruncatedEventLogError):
        await log.read()


async def test_public_surface_is_append_read_and_hook() -> None:
    allowed = {"append", "read", "record_caller_utterance"}
    for cls in (InMemoryEventLog, JsonlEventLog):
        public = {name for name in dir(cls) if not name.startswith("_")}
        assert public == allowed


def test_events_module_has_no_update_or_delete_path() -> None:
    tree = ast.parse(Path(core.events.__file__).read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert names.isdisjoint({"update", "delete", "remove", "overwrite", "truncate"})


def test_import_layers() -> None:
    def imported(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module)
        return found

    crypto_imports = imported(Path(core.crypto.__file__))
    assert "core.events" not in crypto_imports
    assert "core.state" not in crypto_imports
    assert "core.policy" not in crypto_imports
    assert "core.turn" not in crypto_imports
    event_imports = imported(Path(core.events.__file__))
    assert event_imports >= {"core.crypto"}
    assert "core.policy" not in event_imports
    assert "core.turn" not in event_imports
    assert "core.state" not in event_imports


async def test_sensitive_text_is_ciphertext_on_disk(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text="hello there",
        clinical_class=False,
    )
    raw = path.read_text(encoding="utf-8")
    assert "hello there" not in raw
    assert '"kind":"caller_utterance"' in raw
    assert '"call_id":"call_1"' in raw
    line = json.loads(raw)
    blob = line["payload"]["text"]
    assert blob["_enc"] == 1
    assert "n" in blob and "c" in blob


class SpyStore:
    def __init__(self) -> None:
        self._inner = InMemoryKeyStore()
        self.plaintexts: list[bytes] = []

    async def create(self, call_id: str) -> str:
        return await self._inner.create(call_id)

    async def shred(self, call_id: str) -> None:
        await self._inner.shred(call_id)

    async def encrypt(self, call_id: str, plaintext: bytes, *, aad: bytes) -> Ciphertext:
        self.plaintexts.append(plaintext)
        return await self._inner.encrypt(call_id, plaintext, aad=aad)

    async def decrypt(
        self, call_id: str, ciphertext: Ciphertext, *, aad: bytes
    ) -> bytes | Shredded:
        return await self._inner.decrypt(call_id, ciphertext, aad=aad)


def _walk_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from _walk_strings(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk_strings(inner)


async def test_clinical_text_never_reaches_encryptor_file_or_memory(tmp_path: Path) -> None:
    keys = SpyStore()
    await keys.create("call_1")
    path = tmp_path / "events.jsonl"
    jsonl = JsonlEventLog(path, keys)
    memory = InMemoryEventLog(keys)
    for log in (jsonl, memory):
        stored = await log.record_caller_utterance(
            call_id="call_1",
            tenant_id="tenant_a",
            at=NOW,
            turn=1,
            text=SECRET,
            clinical_class=True,
            trip="trip_alpha",
        )
        assert stored == "[redacted:trip_alpha]"
    assert SECRET.encode("utf-8") not in keys.plaintexts
    assert b"[redacted:trip_alpha]" in keys.plaintexts
    raw = path.read_bytes()
    assert SECRET.encode("utf-8") not in raw
    assert b"[redacted:trip_alpha]" not in raw
    for event in await jsonl.read():
        assert SECRET not in str(event.payload)
        assert event.payload["text"] == "[redacted:trip_alpha]"
    for record in memory._stored:
        assert SECRET not in "".join(_walk_strings(record))
    for event in await memory.read():
        assert event.payload["text"] == "[redacted:trip_alpha]"


async def test_non_clinical_text_does_reach_encryptor_not_file(tmp_path: Path) -> None:
    keys = SpyStore()
    await keys.create("call_1")
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text=SECRET,
        clinical_class=False,
    )
    assert SECRET.encode("utf-8") in keys.plaintexts
    assert SECRET.encode("utf-8") not in path.read_bytes()


def test_redact_format() -> None:
    redacted = redact_if_clinical("hi", clinical_class=True, trip="trip_alpha")
    assert redacted == "[redacted:trip_alpha]"
    assert redact_if_clinical("hi", clinical_class=False, trip="trip_alpha") == "hi"
    with pytest.raises(ValueError, match="trip name"):
        redact_if_clinical("hi", clinical_class=True, trip=None)


async def test_ciphertext_moved_to_another_tenant_fails_on_read(keys: InMemoryKeyStore) -> None:
    await ready(keys)
    log = InMemoryEventLog(keys)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text="hello",
        clinical_class=False,
    )
    log._stored[0]["tenant_id"] = "tenant_b"
    with pytest.raises(InvalidCiphertextError):
        await log.read()


async def test_wrong_field_aad_fails_on_read(keys: InMemoryKeyStore) -> None:
    await ready(keys)
    log = InMemoryEventLog(keys)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text="hello",
        clinical_class=False,
    )
    payload = log._stored[0]["payload"]
    assert isinstance(payload, dict)
    blob = payload["text"]
    forged = {
        "call_id": "call_1",
        "tenant_id": "tenant_a",
        "at": NOW.isoformat(),
        "kind": "agent_sentence",
        "payload": {"text": blob, "origin": "backend", "turn": 1},
    }
    with pytest.raises(InvalidCiphertextError):
        await core.events.unseal(keys, forged)


async def test_tampered_jsonl_ciphertext_raises(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="tenant_a",
        at=NOW,
        turn=1,
        text="hello",
        clinical_class=False,
    )
    line = json.loads(path.read_text(encoding="utf-8"))
    blob = line["payload"]["text"]["c"]
    line["payload"]["text"]["c"] = blob[:-2] + ("A" if blob[-1] != "A" else "B") + blob[-1]
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(InvalidCiphertextError):
        await log.read()


async def test_agent_sentence_text_is_encrypted(keys: InMemoryKeyStore, tmp_path: Path) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    await log.append(
        draft(
            kind="agent_sentence",
            payload={"text": "You're all set.", "origin": "backend", "turn": 2},
        )
    )
    raw = path.read_text(encoding="utf-8")
    assert "You're all set." not in raw
    events = await log.read()
    assert events[0].payload["text"] == "You're all set."
    assert events[0].payload["origin"] == "backend"


async def test_tool_args_encrypted_as_one_json_string(
    keys: InMemoryKeyStore, tmp_path: Path
) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    args = json.dumps({"patient_name": "Dana", "slot": "2026-11-04T14:00:00"})
    await log.append(draft(kind="tool_invoked", payload={"tool": "book", "args": args}))
    raw = path.read_text(encoding="utf-8")
    assert "Dana" not in raw
    assert '"tool":"book"' in raw
    events = await log.read()
    assert events[0].payload["tool"] == "book"
    assert events[0].payload["args"] == args


async def test_tool_result_encrypted_as_one_json_string(keys: InMemoryKeyStore) -> None:
    await ready(keys)
    log = InMemoryEventLog(keys)
    result = json.dumps({"booked": True, "booking_id": "bk_1"})
    await log.append(draft(kind="tool_result", payload={"tool": "book", "result": result}))
    events = await log.read()
    assert events[0].payload["result"] == result


async def test_booking_kinds_encrypt_only_name_and_phone(
    keys: InMemoryKeyStore, tmp_path: Path
) -> None:
    await ready(keys)
    path = tmp_path / "events.jsonl"
    log = JsonlEventLog(path, keys)
    await log.append(
        draft(
            kind="booking_attempted",
            payload={
                "patient_name": "Dana Reyes",
                "patient_phone": "+12035551234",
                "service_id": "filler",
                "slot_start": "2026-11-04T14:00:00",
                "confirmed": True,
            },
        )
    )
    raw = path.read_text(encoding="utf-8")
    assert "Dana Reyes" not in raw
    assert "+12035551234" not in raw
    assert '"service_id":"filler"' in raw
    events = await log.read()
    assert events[0].payload["patient_name"] == "Dana Reyes"
    assert events[0].payload["service_id"] == "filler"


async def test_tripwire_and_escalation_names_stay_clear(keys: InMemoryKeyStore) -> None:
    log = InMemoryEventLog(keys)
    await log.append(draft(kind="tripwire_fired", payload={"trip": "trip_alpha"}))
    await log.append(
        draft(
            kind="escalation_committed",
            payload={"trip": "trip_alpha", "path": "callback"},
        )
    )
    events = await log.read()
    assert events[0].payload["trip"] == "trip_alpha"
    assert events[1].payload["path"] == "callback"


async def test_optional_caller_number_on_call_started(
    keys: InMemoryKeyStore, tmp_path: Path
) -> None:
    await keys.create("call_1")
    await keys.create("call_2")
    path = tmp_path / "started.jsonl"
    log = JsonlEventLog(path, keys)
    await log.append(draft(kind="call_started", payload={}))
    await log.append(
        draft(
            kind="call_started",
            payload={"caller_number": "+12035550000"},
            call_id="call_2",
        )
    )
    raw = path.read_text(encoding="utf-8")
    assert "+12035550000" not in raw
    events = await log.read()
    assert events[0].payload == {}
    assert events[1].payload["caller_number"] == "+12035550000"


async def test_record_error_logs_ids_only(
    keys: InMemoryKeyStore, caplog: pytest.LogCaptureFixture
) -> None:
    log = InMemoryEventLog(keys)
    caplog.set_level(logging.INFO, logger="core.events")
    await record_error(
        log, call_id="call_1", tenant_id="tenant_a", at=NOW, code="hedge", where="turn"
    )
    events = await log.read()
    assert events[0].kind == "error"
    assert events[0].payload == {"code": "hedge", "where": "turn"}
    assert "hedge" in caplog.text
    assert "call_1" in caplog.text
    assert "tenant_a" in caplog.text


async def test_turn_latency_schema_accepts_optional_timings(keys: InMemoryKeyStore) -> None:
    log = InMemoryEventLog(keys)
    await log.append(
        draft(
            kind="turn_latency",
            payload={"response_id": "r1", "t_first_chunk_ms": 120},
        )
    )
    events = await log.read()
    assert events[0].payload["t_first_chunk_ms"] == 120


async def test_bind_aad_includes_tenant() -> None:
    bound = bind_aad(tenant_id="t", call_id="c", kind="k", field="f")
    assert bound.startswith(b"v1\x1f")
    assert bound == b"v1\x1ft\x1fc\x1fk\x1ff"
