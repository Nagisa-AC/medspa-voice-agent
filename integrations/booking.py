"""Booking adapter protocol and an in-memory mock driven by tenants/<slug>.mock.yaml."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import yaml

from core.tenant import LiveHours

GRID_MINUTES = 15
HORIZON_DAYS = 14
_SLUG = re.compile(r"[a-z][a-z0-9_]*")
_ROOT_KEYS = frozenset({"providers", "rooms", "hours"})
_NAMED_KEYS = frozenset({"id", "types"})


class BookingError(Exception):
    """Typed adapter failure. Messages carry ids and times, never patient fields."""


class SlotTaken(BookingError):
    def __init__(self, slot: Slot) -> None:
        self.start = slot.start
        self.provider_id = slot.provider_id
        self.room_id = slot.room_id
        self.version = slot.version
        super().__init__(
            f"slot taken {slot.provider_id}/{slot.room_id} "
            f"start={slot.start.isoformat()} version={slot.version}"
        )


class UnknownBooking(BookingError):
    def __init__(self, booking_id: str) -> None:
        self.booking_id = booking_id
        super().__init__(f"unknown booking {booking_id}")


class Unavailable(BookingError):
    def __init__(self, slot: Slot) -> None:
        self.start = slot.start
        self.provider_id = slot.provider_id
        self.room_id = slot.room_id
        super().__init__(
            f"unavailable {slot.provider_id}/{slot.room_id} start={slot.start.isoformat()}"
        )


class IdempotencyConflict(BookingError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__("idempotency conflict")


class MockScheduleError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _require_utc(value: datetime, name: str) -> datetime:
    _require_aware(value, name)
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    return value


@dataclass(frozen=True)
class Slot:
    start: datetime
    end: datetime
    provider_id: str
    room_id: str
    version: int

    def __post_init__(self) -> None:
        _require_utc(self.start, "start")
        _require_utc(self.end, "end")
        if self.end <= self.start:
            raise ValueError("end must be after start")

    @property
    def identity(self) -> tuple[datetime, str, str]:
        return (self.start, self.provider_id, self.room_id)


@dataclass(frozen=True)
class Patient:
    name: str
    phone: str

    def __repr__(self) -> str:
        return "Patient(<redacted>)"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class Booking:
    id: str
    slot: Slot


@dataclass(frozen=True)
class Provider:
    id: str
    types: frozenset[str]


@dataclass(frozen=True)
class Room:
    id: str
    types: frozenset[str]


@dataclass(frozen=True)
class MockSchedule:
    providers: tuple[Provider, ...]
    rooms: tuple[Room, ...]
    hours: LiveHours


class BookingAdapter(Protocol):
    async def list_availability(
        self,
        *,
        service_id: str,
        duration: timedelta,
        room_type: str,
        provider_types: Sequence[str],
        earliest: datetime,
        limit: int,
    ) -> tuple[Slot, ...]: ...

    async def create_booking(
        self,
        *,
        idempotency_key: str,
        slot: Slot,
        patient: Patient,
    ) -> Booking: ...

    async def get_booking(self, booking_id: str) -> Booking: ...


def load_mock_schedule(path: Path) -> MockSchedule:
    source = str(path)
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MockScheduleError(source, f"invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise MockScheduleError(source, "schedule must be a mapping")
    extra = set(raw) - _ROOT_KEYS
    if extra:
        raise MockScheduleError(source, f"unknown fields {sorted(extra)}")
    hours_text = raw.get("hours")
    if not isinstance(hours_text, str) or not hours_text.strip():
        raise MockScheduleError(source, "hours is required")
    try:
        hours = LiveHours.parse(hours_text)
    except ValueError as exc:
        raise MockScheduleError(source, f"hours: {exc}") from None
    providers = _load_named(source, "providers", raw.get("providers"), Provider)
    rooms = _load_named(source, "rooms", raw.get("rooms"), Room)
    return MockSchedule(providers=providers, rooms=rooms, hours=hours)


class MockBookingAdapter:
    def __init__(
        self,
        schedule: MockSchedule,
        *,
        tz: ZoneInfo,
        clock: Callable[[], datetime],
    ) -> None:
        self._schedule = schedule
        self._tz = tz
        self._clock = clock
        self._lock = asyncio.Lock()
        self._bookings: dict[str, Booking] = {}
        self._patients: dict[str, str] = {}
        self._keys: dict[str, str] = {}
        self._versions: dict[tuple[datetime, str, str], int] = {}
        self._fail_next: set[tuple[datetime, str, str]] = set()
        self._hide_next: set[str] = set()
        self._next_id = 1

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        tz: ZoneInfo,
        clock: Callable[[], datetime],
    ) -> MockBookingAdapter:
        return cls(load_mock_schedule(path), tz=tz, clock=clock)

    def fail_next_create(self, slot: Slot) -> None:
        """The next create_booking for this identity raises SlotTaken."""
        self._fail_next.add(slot.identity)

    def hide_on_next_get(self, booking_id: str) -> None:
        """The next get_booking(booking_id) raises UnknownBooking once."""
        self._hide_next.add(booking_id)

    async def list_availability(
        self,
        *,
        service_id: str,
        duration: timedelta,
        room_type: str,
        provider_types: Sequence[str],
        earliest: datetime,
        limit: int,
    ) -> tuple[Slot, ...]:
        _ = service_id
        if duration <= timedelta(0):
            raise ValueError("duration must be positive")
        if limit <= 0:
            return ()
        floor = max(
            _require_aware(earliest, "earliest").astimezone(UTC),
            _require_aware(self._clock(), "clock").astimezone(UTC),
        )
        wanted = frozenset(provider_types)
        providers = tuple(p for p in self._schedule.providers if p.types & wanted)
        rooms = tuple(r for r in self._schedule.rooms if room_type in r.types)
        async with self._lock:
            found: list[Slot] = []
            local_day = floor.astimezone(self._tz).date()
            stop = local_day + timedelta(days=HORIZON_DAYS)
            while local_day < stop and len(found) < limit:
                for start in self._starts_on(local_day, duration):
                    if start < floor:
                        continue
                    end = start + duration
                    slot = self._first_free(providers, rooms, start, end)
                    if slot is None:
                        continue
                    found.append(slot)
                    if len(found) >= limit:
                        break
                local_day += timedelta(days=1)
            return tuple(found)

    async def create_booking(
        self,
        *,
        idempotency_key: str,
        slot: Slot,
        patient: Patient,
    ) -> Booking:
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        async with self._lock:
            existing_id = self._keys.get(idempotency_key)
            if existing_id is not None:
                existing = self._bookings[existing_id]
                if existing.slot.identity != slot.identity:
                    raise IdempotencyConflict(idempotency_key)
                if self._patients[existing_id] != patient.phone:
                    raise IdempotencyConflict(idempotency_key)
                return existing
            if slot.identity in self._fail_next:
                self._fail_next.discard(slot.identity)
                self._bump(slot.identity)
                raise SlotTaken(slot)
            if not self._slot_on_schedule(slot):
                raise Unavailable(slot)
            current = self._versions.get(slot.identity, 0)
            if slot.version != current:
                raise SlotTaken(slot)
            if self._occupied(slot.provider_id, slot.room_id, slot.start, slot.end):
                raise SlotTaken(slot)
            booking = Booking(id=f"bk_{self._next_id:05d}", slot=slot)
            self._next_id += 1
            self._bookings[booking.id] = booking
            self._patients[booking.id] = patient.phone
            self._keys[idempotency_key] = booking.id
            self._bump(slot.identity)
            return booking

    async def get_booking(self, booking_id: str) -> Booking:
        async with self._lock:
            if booking_id in self._hide_next:
                self._hide_next.discard(booking_id)
                raise UnknownBooking(booking_id)
            booking = self._bookings.get(booking_id)
            if booking is None:
                raise UnknownBooking(booking_id)
            return booking

    def _bump(self, identity: tuple[datetime, str, str]) -> None:
        self._versions[identity] = self._versions.get(identity, 0) + 1

    def _first_free(
        self,
        providers: Sequence[Provider],
        rooms: Sequence[Room],
        start: datetime,
        end: datetime,
    ) -> Slot | None:
        for provider in providers:
            for room in rooms:
                if self._occupied(provider.id, room.id, start, end):
                    continue
                identity = (start, provider.id, room.id)
                return Slot(
                    start=start,
                    end=end,
                    provider_id=provider.id,
                    room_id=room.id,
                    version=self._versions.get(identity, 0),
                )
        return None

    def _occupied(self, provider_id: str, room_id: str, start: datetime, end: datetime) -> bool:
        for booking in self._bookings.values():
            held = booking.slot
            if not (start < held.end and held.start < end):
                continue
            if held.provider_id == provider_id or held.room_id == room_id:
                return True
        return False

    def _slot_on_schedule(self, slot: Slot) -> bool:
        if slot.provider_id not in {p.id for p in self._schedule.providers}:
            return False
        if slot.room_id not in {r.id for r in self._schedule.rooms}:
            return False
        if slot.start < _require_aware(self._clock(), "clock").astimezone(UTC):
            return False
        local = slot.start.astimezone(self._tz)
        duration = slot.end - slot.start
        minute = local.hour * 60 + local.minute
        if minute % GRID_MINUTES:
            return False
        dur_min = int(duration.total_seconds() // 60)
        if timedelta(minutes=dur_min) != duration:
            return False
        weekday = local.weekday()
        for window in self._schedule.hours.windows:
            if window.weekday != weekday:
                continue
            if window.start_min <= minute and minute + dur_min <= window.end_min:
                expected = self._wall_to_utc(local.date(), minute)
                if slot.start in expected:
                    return True
        return False

    def _starts_on(self, day: date, duration: timedelta) -> list[datetime]:
        dur_min = int(duration.total_seconds() // 60)
        if timedelta(minutes=dur_min) != duration or dur_min <= 0:
            return []
        starts: list[datetime] = []
        weekday = day.weekday()
        for window in self._schedule.hours.windows:
            if window.weekday != weekday:
                continue
            minute = window.start_min
            if minute % GRID_MINUTES:
                minute += GRID_MINUTES - (minute % GRID_MINUTES)
            while minute + dur_min <= window.end_min:
                starts.extend(self._wall_to_utc(day, minute))
                minute += GRID_MINUTES
        starts.sort()
        return starts

    def _wall_to_utc(self, day: date, minute: int) -> list[datetime]:
        hour, remainder = divmod(minute, 60)
        if hour >= 24:
            return []
        instants: list[datetime] = []
        for fold in (0, 1):
            local = datetime(
                day.year, day.month, day.day, hour, remainder, tzinfo=self._tz, fold=fold
            )
            utc = local.astimezone(UTC)
            roundtrip = utc.astimezone(self._tz)
            wall = (day.year, day.month, day.day, hour, remainder)
            got = (
                roundtrip.year,
                roundtrip.month,
                roundtrip.day,
                roundtrip.hour,
                roundtrip.minute,
            )
            if got != wall:
                continue
            if utc not in instants:
                instants.append(utc)
        return instants


def _load_named(
    source: str,
    field: str,
    raw: object,
    cls: type[Provider] | type[Room],
) -> tuple[Any, ...]:
    if not isinstance(raw, list) or not raw:
        raise MockScheduleError(source, f"{field} must be a non-empty list")
    seen: set[str] = set()
    items: list[Any] = []
    for i, item in enumerate(raw):
        loc = f"{field}[{i}]"
        if not isinstance(item, dict):
            raise MockScheduleError(source, f"{loc} must be a mapping")
        extra = set(item) - _NAMED_KEYS
        if extra:
            raise MockScheduleError(source, f"{loc} unknown fields {sorted(extra)}")
        ident = item.get("id")
        if not isinstance(ident, str) or not _SLUG.fullmatch(ident):
            raise MockScheduleError(source, f"{loc}.id must be a slug")
        if ident in seen:
            raise MockScheduleError(source, f"duplicate id {ident!r}")
        seen.add(ident)
        types_raw = item.get("types")
        if not isinstance(types_raw, list) or not types_raw:
            raise MockScheduleError(source, f"{loc}.types must be a non-empty list")
        types: set[str] = set()
        for j, label in enumerate(types_raw):
            if not isinstance(label, str) or not label.strip():
                raise MockScheduleError(source, f"{loc}.types[{j}] must be a non-empty string")
            types.add(label.strip())
        items.append(cls(id=ident, types=frozenset(types)))
    items.sort(key=lambda row: row.id)
    return tuple(items)
