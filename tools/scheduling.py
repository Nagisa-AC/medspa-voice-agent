"""check_availability: spoken times for the caller, structured slots for the model."""

from __future__ import annotations

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
from integrations.booking import BookingAdapter, Slot

HAVE_ONE = "I have {a}."
HAVE_TWO = "I have {a} or {b}."
HAVE_THREE = "I have {a}, {b}, or {c}."
EMPTY = "I don't have anything in the next two weeks."
UNKNOWN_PREFIX = "I don't have that on the list. I can book a consult. "
UNKNOWN_EMPTY = "I don't have that on the list. I don't have anything in the next two weeks."
AMBIGUOUS = "Which were you asking about, {names}?"
_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


@dataclass(frozen=True)
class CheckAvailabilityResult:
    say: str
    model: Mapping[str, object]


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
