"""Append-only event log with per-field payload encryption."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Protocol, cast, final, get_args

from core.crypto import (
    Ciphertext,
    InvalidCiphertextError,
    KeyStore,
    Shredded,
    bind_aad,
)

logger = logging.getLogger("core.events")

EventKind = Literal[
    "call_started",
    "caller_utterance",
    "agent_sentence",
    "tool_invoked",
    "tool_result",
    "tripwire_fired",
    "escalation_committed",
    "guardrail_blocked",
    "booking_attempted",
    "booking_confirmed",
    "booking_failed",
    "booking_needs_review",
    "turn_latency",
    "call_ended",
    "error",
    "sms_logged",
    "faq_served",
    "escalation_path_chosen",
    "call_data_shredded",
]

EVENT_KINDS: Final[frozenset[str]] = frozenset(get_args(EventKind))

ClearValue = str | int | bool | list[str]
PayloadValue = ClearValue | Shredded
ShredReason = Literal["retention", "caller_request"]

PAYLOAD_FIELDS: Final[dict[EventKind, frozenset[str]]] = {
    "call_started": frozenset({"caller_number"}),
    "caller_utterance": frozenset({"text", "turn"}),
    "agent_sentence": frozenset({"text", "origin", "turn"}),
    "tool_invoked": frozenset({"tool", "args"}),
    "tool_result": frozenset({"tool", "result"}),
    "tripwire_fired": frozenset({"trip", "path"}),
    "escalation_committed": frozenset({"trip", "path", "tier"}),
    "guardrail_blocked": frozenset({"check", "pattern_ids"}),
    "booking_attempted": frozenset(
        {"patient_name", "patient_phone", "service_id", "slot_start", "confirmed"}
    ),
    "booking_confirmed": frozenset(
        {"patient_name", "patient_phone", "booking_id", "service_id", "slot_start", "confirmed"}
    ),
    "booking_failed": frozenset(
        {"patient_name", "patient_phone", "reason", "service_id", "slot_start"}
    ),
    "booking_needs_review": frozenset(
        {"patient_name", "patient_phone", "reason", "service_id", "slot_start"}
    ),
    "turn_latency": frozenset(
        {"response_id", "t_first_chunk_ms", "t_model_first_token_ms", "t_tool_ms", "t_guardrail_ms"}
    ),
    "call_ended": frozenset({"reason"}),
    "error": frozenset({"code", "where"}),
    "sms_logged": frozenset(
        {
            "category",
            "patient_name",
            "callback_number",
            "service_id",
            "slot_window",
            "new_or_existing",
        }
    ),
    "faq_served": frozenset({"faq_id"}),
    "escalation_path_chosen": frozenset({"trip", "path", "tier"}),
    "call_data_shredded": frozenset({"reason"}),
}

SENSITIVE_FIELDS: Final[dict[EventKind, frozenset[str]]] = {
    "call_started": frozenset({"caller_number"}),
    "caller_utterance": frozenset({"text"}),
    "agent_sentence": frozenset({"text"}),
    "tool_invoked": frozenset({"args"}),
    "tool_result": frozenset({"result"}),
    "tripwire_fired": frozenset(),
    "escalation_committed": frozenset(),
    "guardrail_blocked": frozenset(),
    "booking_attempted": frozenset({"patient_name", "patient_phone"}),
    "booking_confirmed": frozenset({"patient_name", "patient_phone"}),
    "booking_failed": frozenset({"patient_name", "patient_phone"}),
    "booking_needs_review": frozenset({"patient_name", "patient_phone"}),
    "turn_latency": frozenset(),
    "call_ended": frozenset(),
    "error": frozenset(),
    "sms_logged": frozenset({"patient_name", "callback_number"}),
    "faq_served": frozenset(),
    "escalation_path_chosen": frozenset(),
    "call_data_shredded": frozenset(),
}

OPTIONAL_FIELDS: Final[dict[EventKind, frozenset[str]]] = {
    "call_started": frozenset({"caller_number"}),
    "caller_utterance": frozenset(),
    "agent_sentence": frozenset(),
    "tool_invoked": frozenset(),
    "tool_result": frozenset(),
    "tripwire_fired": frozenset({"path"}),
    "escalation_committed": frozenset({"tier"}),
    "guardrail_blocked": frozenset(),
    "booking_attempted": frozenset(),
    "booking_confirmed": frozenset(),
    "booking_failed": frozenset(),
    "booking_needs_review": frozenset(),
    "turn_latency": frozenset({"t_model_first_token_ms", "t_tool_ms", "t_guardrail_ms"}),
    "call_ended": frozenset(),
    "error": frozenset(),
    "sms_logged": frozenset(
        {"patient_name", "callback_number", "service_id", "slot_window", "new_or_existing"}
    ),
    "faq_served": frozenset(),
    "escalation_path_chosen": frozenset({"tier"}),
    "call_data_shredded": frozenset(),
}


class UnknownEventKindError(Exception):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"unknown event kind {kind!r}")


class UnknownPayloadFieldError(Exception):
    def __init__(self, kind: str, field: str) -> None:
        self.kind = kind
        self.field = field
        super().__init__(f"unknown field {field!r} on {kind}")


class MissingPayloadFieldError(Exception):
    def __init__(self, kind: str, field: str) -> None:
        self.kind = kind
        self.field = field
        super().__init__(f"missing field {field!r} on {kind}")


class CallerUtteranceMustUseHookError(Exception):
    """caller_utterance must go through record_caller_utterance so redaction cannot be skipped."""


class TruncatedEventLogError(Exception):
    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"truncated event log {path}")


@dataclass(frozen=True)
class EventDraft:
    call_id: str
    tenant_id: str
    at: datetime
    kind: EventKind
    payload: dict[str, ClearValue]


@dataclass(frozen=True)
class Event:
    call_id: str
    tenant_id: str
    at: datetime
    kind: EventKind
    payload: dict[str, PayloadValue]


class EventLog(Protocol):
    async def append(self, draft: EventDraft) -> None: ...

    async def record_caller_utterance(
        self,
        *,
        call_id: str,
        tenant_id: str,
        at: datetime,
        turn: int,
        text: str,
        clinical_class: bool,
        trip: str | None = None,
    ) -> str: ...

    async def read(self, call_id: str | None = None) -> tuple[Event, ...]: ...


def redact_if_clinical(text: str, *, clinical_class: bool, trip: str | None) -> str:
    if not clinical_class:
        return text
    if not trip:
        raise ValueError("clinical_class requires a trip name")
    return f"[redacted:{trip}]"


def log_ids(message: str, **ids: str | int) -> None:
    """Application log: ids and timings only."""
    detail = " ".join(f"{key}={value}" for key, value in ids.items())
    logger.info("%s %s", message, detail)


class EventLogBase(ABC):
    def __init__(self, keys: KeyStore) -> None:
        self._keys = keys

    async def append(self, draft: EventDraft) -> None:
        if draft.kind == "caller_utterance":
            raise CallerUtteranceMustUseHookError
        await self._commit(draft)

    async def record_caller_utterance(
        self,
        *,
        call_id: str,
        tenant_id: str,
        at: datetime,
        turn: int,
        text: str,
        clinical_class: bool,
        trip: str | None = None,
    ) -> str:
        stored = redact_if_clinical(text, clinical_class=clinical_class, trip=trip)
        await self._commit(
            EventDraft(
                call_id=call_id,
                tenant_id=tenant_id,
                at=at,
                kind="caller_utterance",
                payload={"text": stored, "turn": turn},
            )
        )
        return stored

    async def read(self, call_id: str | None = None) -> tuple[Event, ...]:
        events: list[Event] = []
        for record in await self._load():
            event = await unseal(self._keys, record)
            if call_id is None or event.call_id == call_id:
                events.append(event)
        return tuple(events)

    async def _commit(self, draft: EventDraft) -> None:
        validate_draft(draft)
        await self._persist(await seal(self._keys, draft))

    @abstractmethod
    async def _persist(self, record: dict[str, object]) -> None: ...

    @abstractmethod
    async def _load(self) -> Sequence[Mapping[str, object]]: ...


@final
class InMemoryEventLog(EventLogBase):
    def __init__(self, keys: KeyStore) -> None:
        super().__init__(keys)
        self._stored: list[dict[str, object]] = []

    async def _persist(self, record: dict[str, object]) -> None:
        self._stored.append(record)

    async def _load(self) -> Sequence[Mapping[str, object]]:
        return self._stored


@final
class JsonlEventLog(EventLogBase):
    def __init__(self, path: Path, keys: KeyStore) -> None:
        super().__init__(keys)
        self._path = path

    async def _persist(self, record: dict[str, object]) -> None:
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
        with self._path.open("ab") as handle:
            handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    async def _load(self) -> Sequence[Mapping[str, object]]:
        if not self._path.exists():
            return []
        data = self._path.read_bytes()
        if not data:
            return []
        if not data.endswith(b"\n"):
            raise TruncatedEventLogError(str(self._path))
        records: list[dict[str, object]] = []
        for raw in data.split(b"\n")[:-1]:
            if not raw:
                raise TruncatedEventLogError(str(self._path))
            loaded = json.loads(raw)
            if not isinstance(loaded, dict):
                raise TruncatedEventLogError(str(self._path))
            records.append(cast(dict[str, object], loaded))
        return records


async def record_error(
    log: EventLog,
    *,
    call_id: str,
    tenant_id: str,
    at: datetime,
    code: str,
    where: str,
) -> None:
    log_ids("error", call_id=call_id, tenant_id=tenant_id, code=code, where=where)
    await log.append(
        EventDraft(
            call_id=call_id,
            tenant_id=tenant_id,
            at=at,
            kind="error",
            payload={"code": code, "where": where},
        )
    )


async def shred_call(
    keys: KeyStore,
    log: EventLog,
    *,
    call_id: str,
    tenant_id: str,
    at: datetime,
    reason: ShredReason,
) -> None:
    await log.append(
        EventDraft(
            call_id=call_id,
            tenant_id=tenant_id,
            at=at,
            kind="call_data_shredded",
            payload={"reason": reason},
        )
    )
    await keys.shred(call_id)


def validate_draft(draft: EventDraft) -> None:
    if not draft.call_id:
        raise ValueError("call_id must not be empty")
    if not draft.tenant_id:
        raise ValueError("tenant_id must not be empty")
    if draft.at.tzinfo is None or draft.at.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    if draft.kind not in EVENT_KINDS:
        raise UnknownEventKindError(draft.kind)
    allowed = PAYLOAD_FIELDS[draft.kind]
    sensitive = SENSITIVE_FIELDS[draft.kind]
    optional = OPTIONAL_FIELDS[draft.kind]
    required = allowed - optional
    extra = set(draft.payload) - allowed
    if extra:
        raise UnknownPayloadFieldError(draft.kind, sorted(extra)[0])
    missing = required - set(draft.payload)
    if missing:
        raise MissingPayloadFieldError(draft.kind, sorted(missing)[0])
    for field, value in draft.payload.items():
        if field in sensitive:
            if not isinstance(value, str):
                raise ValueError(f"{draft.kind}.{field} must be a string")
            if field in ("args", "result"):
                _require_json_object_string(value, field)
            continue
        if not _is_clear_value(value):
            raise ValueError(f"{draft.kind}.{field} has a disallowed type")
        _check_closed_string(draft.kind, field, value)


async def seal(keys: KeyStore, draft: EventDraft) -> dict[str, object]:
    kind = draft.kind
    sealed: dict[str, object] = {}
    for field, value in draft.payload.items():
        if field in SENSITIVE_FIELDS[kind]:
            if not isinstance(value, str):
                raise ValueError(f"{kind}.{field} must be a string")
            aad = bind_aad(tenant_id=draft.tenant_id, call_id=draft.call_id, kind=kind, field=field)
            blob = await keys.encrypt(draft.call_id, value.encode("utf-8"), aad=aad)
            sealed[field] = _ciphertext_json(blob)
        else:
            sealed[field] = value
    return {
        "call_id": draft.call_id,
        "tenant_id": draft.tenant_id,
        "at": draft.at.isoformat(),
        "kind": kind,
        "payload": sealed,
    }


async def unseal(keys: KeyStore, record: Mapping[str, object]) -> Event:
    kind = _kind_of(record.get("kind"))
    call_id = _require_str(record.get("call_id"), "call_id")
    tenant_id = _require_str(record.get("tenant_id"), "tenant_id")
    at = _parse_at(record.get("at"))
    raw_payload = record.get("payload")
    if not isinstance(raw_payload, dict):
        raise InvalidCiphertextError(call_id)
    payload: dict[str, PayloadValue] = {}
    for field, value in raw_payload.items():
        if not isinstance(field, str):
            raise InvalidCiphertextError(call_id)
        if field in SENSITIVE_FIELDS[kind]:
            blob = _parse_ciphertext(value, call_id)
            aad = bind_aad(tenant_id=tenant_id, call_id=call_id, kind=kind, field=field)
            decrypted = await keys.decrypt(call_id, blob, aad=aad)
            if isinstance(decrypted, Shredded):
                payload[field] = decrypted
            else:
                payload[field] = decrypted.decode("utf-8")
        else:
            if not _is_clear_value(value):
                raise InvalidCiphertextError(call_id)
            payload[field] = value
    return Event(call_id=call_id, tenant_id=tenant_id, at=at, kind=kind, payload=payload)


def _ciphertext_json(blob: Ciphertext) -> dict[str, object]:
    return {
        "_enc": 1,
        "kid": blob.key_id,
        "n": base64.b64encode(blob.nonce).decode("ascii"),
        "c": base64.b64encode(blob.data).decode("ascii"),
    }


def _parse_ciphertext(value: object, call_id: str) -> Ciphertext:
    if not isinstance(value, dict) or value.get("_enc") != 1:
        raise InvalidCiphertextError(call_id)
    kid = value.get("kid")
    nonce_b64 = value.get("n")
    data_b64 = value.get("c")
    if not isinstance(kid, str) or not isinstance(nonce_b64, str) or not isinstance(data_b64, str):
        raise InvalidCiphertextError(call_id)
    try:
        nonce = base64.b64decode(nonce_b64, validate=True)
        data = base64.b64decode(data_b64, validate=True)
    except (ValueError, binascii.Error):
        raise InvalidCiphertextError(call_id) from None
    try:
        return Ciphertext(key_id=kid, nonce=nonce, data=data)
    except ValueError as exc:
        raise InvalidCiphertextError(call_id) from exc


def _kind_of(value: object) -> EventKind:
    if not isinstance(value, str) or value not in EVENT_KINDS:
        raise UnknownEventKindError(str(value))
    return cast(EventKind, value)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _parse_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("at must be an ISO datetime")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    return parsed


def _is_clear_value(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        return True
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _require_json_object_string(value: str, field: str) -> None:
    try:
        json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be a JSON string") from exc


def _check_closed_string(kind: EventKind, field: str, value: ClearValue) -> None:
    if field == "origin" and value not in ("backend", "model"):
        raise ValueError("origin must be backend or model")
    if field == "path" and value not in ("emergency_end", "transfer", "callback"):
        raise ValueError("path must be emergency_end, transfer or callback")
    if (
        kind == "call_data_shredded"
        and field == "reason"
        and value not in ("retention", "caller_request")
    ):
        raise ValueError("reason must be retention or caller_request")
