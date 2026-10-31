"""check_availability and book: spoken lines for the caller, structured results for the model."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from zoneinfo import ZoneInfo

from core.state import CallState, NumberExtractor, OfferedSlot, admit_numbers
from core.tenant import (
    CONSULT_ID,
    Ambiguous,
    Consult,
    Match,
    NoMatch,
    Service,
    Tenant,
    resolve_service,
)
from integrations.booking import (
    BookingAdapter,
    IdempotencyConflict,
    Patient,
    Slot,
    SlotTaken,
    Unavailable,
    UnknownBooking,
)
from tools.pricing import format_money

HAVE_ONE = "I have {a}."
HAVE_TWO = "I have {a} or {b}."
HAVE_THREE = "I have {a}, {b}, or {c}."
EMPTY = "I don't have anything in the next two weeks."
UNKNOWN_PREFIX = "I don't have that on the list. I can book a consult. "
UNKNOWN_EMPTY = "I don't have that on the list. I don't have anything in the next two weeks."
AMBIGUOUS = "Which were you asking about, {names}?"
CONFIRM = "Just to confirm: {what} on {time}. Should I go ahead and book that?"
CONFIRM_BARE = "Just to confirm — should I go ahead and book that?"
NEED_NAME = "I need a name for the booking."
NEED_PHONE = "Can I get a good number to reach you at?"
NO_OFFER = "Let me find you a time. What would you like to book?"
NOT_HELD = "I don't have that time held. "
TAKEN = "That time was just taken. "
UNAVAILABLE = "That time isn't open anymore. "
DST_COLLISION = "That clock time matched more than one opening. "
REVIEW = "I've got your request in and someone from the team will confirm with you."
SUCCESS = "You're all set for {what} on {time}."
DEPOSIT = " Just so you know, there's a {amount} deposit for this treatment."
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}
_E164 = re.compile(r"\+[1-9]\d{1,14}")
_MIN_SECRET = 32


@dataclass(frozen=True)
class CheckAvailabilityResult:
    say: str
    model: Mapping[str, object]


@dataclass(frozen=True)
class BookResult:
    say: str
    model: Mapping[str, object]
    booking_id: str | None = None
    service_id: str | None = None

    def __repr__(self) -> str:
        return (
            f"BookResult(booked={self.model.get('booked')}, "
            f"needs_review={self.model.get('needs_review')}, "
            f"has_booking_id={self.booking_id is not None})"
        )

    def __str__(self) -> str:
        return repr(self)


def format_slot_time(start: datetime, tz: ZoneInfo) -> str:
    """Tenant-local spoken time: 'Wednesday the 4th at 2:00 PM'."""
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start must be timezone-aware")
    local = start.astimezone(tz)
    hour12 = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    return f"{local.strftime('%A')} the {_ordinal(local.day)} at {hour12}:{local.minute:02d} {ampm}"


async def check_availability(
    tenant: Tenant,
    state: CallState,
    adapter: BookingAdapter,
    service_text: str,
    *,
    earliest_iso: str | None,
    extract: NumberExtractor,
    now: datetime,
    limit: int = 3,
) -> CheckAvailabilityResult:
    """List bookable times. Admit the say. Model payload has times, not provider or room ids."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    cap = _clamp_limit(limit)
    earliest = _parse_earliest(earliest_iso, tz=tenant.tz, now=now)
    resolution = resolve_service(tenant, service_text)
    match resolution:
        case Ambiguous(services):
            say = AMBIGUOUS.format(names=_join_names(tuple(item.name for item in services)))
            model: Mapping[str, object] = MappingProxyType(
                {
                    "found": False,
                    "already_spoken": True,
                    "matches": tuple(item.id for item in services),
                }
            )
        case Match(service) if _books_as_consult(service):
            slots = await _list(adapter, tenant.consult, service.id, earliest, cap)
            _store_offers(state, service.id, slots)
            say = _have_line(tuple(format_slot_time(slot.start, tenant.tz) for slot in slots))
            model = _listed_model(
                found=True, service_id=service.id, consult=True, slots=slots, tz=tenant.tz
            )
        case Match(service):
            slots = await _list(adapter, service, service.id, earliest, cap)
            _store_offers(state, service.id, slots)
            say = _have_line(tuple(format_slot_time(slot.start, tenant.tz) for slot in slots))
            model = _listed_model(
                found=True, service_id=service.id, consult=False, slots=slots, tz=tenant.tz
            )
        case NoMatch():
            slots = await _list(adapter, tenant.consult, CONSULT_ID, earliest, cap)
            _store_offers(state, CONSULT_ID, slots)
            displays = tuple(format_slot_time(slot.start, tenant.tz) for slot in slots)
            say = UNKNOWN_EMPTY if not displays else UNKNOWN_PREFIX + _have_line(displays)
            model = _listed_model(
                found=False, service_id=CONSULT_ID, consult=True, slots=slots, tz=tenant.tz
            )
    admit_numbers(state, say, extract)
    return CheckAvailabilityResult(say=say, model=model)


def _books_as_consult(service: Service) -> bool:
    return service.consult_required or not service.quotable


def _clamp_limit(limit: int) -> int:
    if limit < 1:
        return 1
    if limit > 3:
        return 3
    return limit


def _parse_earliest(earliest_iso: str | None, *, tz: ZoneInfo, now: datetime) -> datetime:
    if earliest_iso is None or not earliest_iso.strip():
        return now
    text = earliest_iso.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return now
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        utc = _naive_local_to_utc(parsed, tz)
        if utc is None:
            return now
        parsed = utc
    if parsed < now:
        return now
    return parsed


def _naive_local_to_utc(naive: datetime, tz: ZoneInfo) -> datetime | None:
    attached = naive.replace(tzinfo=tz)
    utc = attached.astimezone(UTC)
    back = utc.astimezone(tz)
    if (back.year, back.month, back.day, back.hour, back.minute, back.second) != (
        naive.year,
        naive.month,
        naive.day,
        naive.hour,
        naive.minute,
        naive.second,
    ):
        return None
    return utc


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = _ORDINAL_SUFFIX.get(day % 10, "th")
    return f"{day}{suffix}"


def _join_names(names: Sequence[str]) -> str:
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return f"{', '.join(names[:-1])}, or {names[-1]}"


def _have_line(displays: Sequence[str]) -> str:
    if not displays:
        return EMPTY
    if len(displays) == 1:
        return HAVE_ONE.format(a=displays[0])
    if len(displays) == 2:
        return HAVE_TWO.format(a=displays[0], b=displays[1])
    return HAVE_THREE.format(a=displays[0], b=displays[1], c=displays[2])


def _local_naive_iso(start: datetime, tz: ZoneInfo) -> str:
    local = start.astimezone(tz).replace(microsecond=0)
    return local.strftime("%Y-%m-%dT%H:%M:%S")


def _listed_model(
    *,
    found: bool,
    service_id: str,
    consult: bool,
    slots: Sequence[Slot],
    tz: ZoneInfo,
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "found": found,
            "service_id": service_id,
            "already_spoken": True,
            "consult": consult,
            "slots": tuple(
                MappingProxyType(
                    {
                        "index": index,
                        "start": _local_naive_iso(slot.start, tz),
                        "say": format_slot_time(slot.start, tz),
                    }
                )
                for index, slot in enumerate(slots, start=1)
            ),
        }
    )


def _store_offers(state: CallState, service_id: str, slots: Sequence[Slot]) -> None:
    state.offered_slots = [
        OfferedSlot(
            service_id=service_id,
            start=slot.start,
            end=slot.end,
            provider_id=slot.provider_id,
            room_id=slot.room_id,
            version=slot.version,
        )
        for slot in slots
    ]


async def _list(
    adapter: BookingAdapter,
    bookable: Service | Consult,
    service_id: str,
    earliest: datetime,
    limit: int,
) -> tuple[Slot, ...]:
    duration_min = bookable.duration_min
    room_type = bookable.room_type
    if duration_min is None or room_type is None or not bookable.provider_types:
        raise RuntimeError(f"bookable {service_id} is missing duration, room, or providers")
    return await adapter.list_availability(
        service_id=service_id,
        duration=timedelta(minutes=duration_min),
        room_type=room_type,
        provider_types=bookable.provider_types,
        earliest=earliest,
        limit=limit,
    )


def normalize_phone(raw: str) -> str | None:
    """Return E.164, or None if the input cannot be a reachable number."""
    text = raw.strip()
    if _E164.fullmatch(text):
        if text.startswith("+1"):
            return _nanp_from_national(text[2:])
        return text
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10:
        return _nanp_from_national(digits)
    if len(digits) == 11 and digits.startswith("1"):
        return _nanp_from_national(digits[1:])
    return None


def _nanp_from_national(national: str) -> str | None:
    if len(national) != 10 or national[0] in "01":
        return None
    return f"+1{national}"


async def book(
    tenant: Tenant,
    state: CallState,
    adapter: BookingAdapter,
    *,
    slot_start_iso: str,
    patient_name: str,
    patient_phone: str,
    confirmed: bool,
    extract: NumberExtractor,
    now: datetime,
    secret: bytes,
) -> BookResult:
    """Book one offered slot. Admit the say. Model payload is flags only."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if len(secret) < _MIN_SECRET:
        raise ValueError("secret must be at least 32 bytes")
    hits = _matching_offers(state, slot_start_iso, tenant.tz)
    if confirmed is not True:
        if len(hits) == 1:
            say = CONFIRM.format(
                what=_spoken_what(tenant, hits[0].service_id),
                time=format_slot_time(hits[0].start, tenant.tz),
            )
        else:
            say = CONFIRM_BARE
        return _finish(
            state,
            say,
            _flags(booked=False, needs_review=False, retry=False),
            extract,
            service_id=hits[0].service_id if len(hits) == 1 else None,
        )
    name = patient_name.strip()
    if not name:
        return _finish(
            state,
            NEED_NAME,
            _flags(booked=False, needs_review=False, retry=False),
            extract,
            service_id=hits[0].service_id if len(hits) == 1 else None,
        )
    phone = normalize_phone(patient_phone)
    if phone is None:
        return _finish(
            state,
            NEED_PHONE,
            _flags(booked=False, needs_review=False, retry=False),
            extract,
            service_id=hits[0].service_id if len(hits) == 1 else None,
        )
    if len(hits) != 1:
        if not state.offered_slots:
            return _finish(
                state, NO_OFFER, _flags(booked=False, needs_review=False, retry=True), extract
            )
        prefix = DST_COLLISION if len(hits) > 1 else NOT_HELD
        times = await _relist(tenant, state, adapter, state.offered_slots[0].service_id, now)
        return _finish(
            state,
            prefix + times,
            _flags(booked=False, needs_review=False, retry=True),
            extract,
            service_id=state.offered_slots[0].service_id,
        )
    offered = hits[0]
    slot = Slot(
        start=offered.start,
        end=offered.end,
        provider_id=offered.provider_id,
        room_id=offered.room_id,
        version=offered.version,
    )
    key = _idempotency_key(secret, tenant.id, phone, offered.service_id, offered.start)
    patient = Patient(name=name, phone=phone)
    try:
        created = await adapter.create_booking(idempotency_key=key, slot=slot, patient=patient)
    except SlotTaken:
        times = await _relist(tenant, state, adapter, offered.service_id, now)
        return _finish(
            state,
            TAKEN + times,
            _flags(booked=False, needs_review=False, retry=True),
            extract,
            service_id=offered.service_id,
        )
    except Unavailable:
        times = await _relist(tenant, state, adapter, offered.service_id, now)
        return _finish(
            state,
            UNAVAILABLE + times,
            _flags(booked=False, needs_review=False, retry=True),
            extract,
            service_id=offered.service_id,
        )
    except IdempotencyConflict:
        state.idempotency_keys.add(key)
        return _finish(
            state,
            REVIEW,
            _flags(booked=False, needs_review=True, retry=False),
            extract,
            service_id=offered.service_id,
        )
    state.idempotency_keys.add(key)
    try:
        returned = await adapter.get_booking(created.id)
    except UnknownBooking:
        return _finish(
            state,
            REVIEW,
            _flags(booked=False, needs_review=True, retry=False),
            extract,
            service_id=offered.service_id,
        )
    if returned.slot.identity != created.slot.identity or returned.id != created.id:
        return _finish(
            state,
            REVIEW,
            _flags(booked=False, needs_review=True, retry=False),
            extract,
            service_id=offered.service_id,
        )
    what = _spoken_what(tenant, offered.service_id)
    time = format_slot_time(returned.slot.start, tenant.tz)
    say = SUCCESS.format(what=what, time=time) + _deposit_clause(tenant, offered.service_id)
    return _finish(
        state,
        say,
        _flags(booked=True, needs_review=False, retry=False),
        extract,
        booking_id=created.id,
        service_id=offered.service_id,
    )


def _finish(
    state: CallState,
    say: str,
    model: Mapping[str, object],
    extract: NumberExtractor,
    *,
    booking_id: str | None = None,
    service_id: str | None = None,
) -> BookResult:
    admit_numbers(state, say, extract)
    return BookResult(say=say, model=model, booking_id=booking_id, service_id=service_id)


def _flags(*, booked: bool, needs_review: bool, retry: bool) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "booked": booked,
            "needs_review": needs_review,
            "retry": retry,
            "already_spoken": True,
        }
    )


def _matching_offers(state: CallState, slot_start_iso: str, tz: ZoneInfo) -> list[OfferedSlot]:
    canonical = _canonical_local_iso(slot_start_iso, tz)
    if canonical is None:
        return []
    return [item for item in state.offered_slots if _local_naive_iso(item.start, tz) == canonical]


def _canonical_local_iso(slot_start_iso: str, tz: ZoneInfo) -> str | None:
    text = slot_start_iso.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        utc = _naive_local_to_utc(parsed, tz)
        if utc is None:
            return None
        parsed = utc
    return _local_naive_iso(parsed, tz)


def _idempotency_key(
    secret: bytes, tenant_id: str, phone: str, service_id: str, start: datetime
) -> str:
    payload = b"\x1f".join(
        (
            tenant_id.encode(),
            phone.encode(),
            service_id.encode(),
            start.astimezone(UTC).isoformat().encode(),
        )
    )
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def _spoken_what(tenant: Tenant, service_id: str) -> str:
    if service_id == CONSULT_ID:
        return "a consult"
    for service in tenant.services:
        if service.id == service_id:
            if _books_as_consult(service):
                return "a consult"
            return service.name
    return "a consult"


def _deposit_clause(tenant: Tenant, service_id: str) -> str:
    for service in tenant.services:
        if (
            service.id == service_id
            and service.deposit_required
            and service.deposit_amount is not None
        ):
            return DEPOSIT.format(amount=format_money(service.deposit_amount))
    return ""


def _bookable_for(tenant: Tenant, service_id: str) -> Service | Consult:
    if service_id == CONSULT_ID:
        return tenant.consult
    for service in tenant.services:
        if service.id == service_id:
            if _books_as_consult(service):
                return tenant.consult
            return service
    return tenant.consult


async def _relist(
    tenant: Tenant,
    state: CallState,
    adapter: BookingAdapter,
    service_id: str,
    now: datetime,
) -> str:
    slots = await _list(adapter, _bookable_for(tenant, service_id), service_id, now, 3)
    _store_offers(state, service_id, slots)
    return _have_line(tuple(format_slot_time(slot.start, tenant.tz) for slot in slots))
