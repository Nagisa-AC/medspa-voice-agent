"""Adapter contract. Add a factory param to run these against Square in M4."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.tenant import load_tenant_file
from integrations.booking import (
    Booking,
    BookingAdapter,
    IdempotencyConflict,
    MockBookingAdapter,
    Patient,
    Slot,
    SlotTaken,
    Unavailable,
    UnknownBooking,
)

ROOT = Path(__file__).resolve().parents[2]
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=date(2026, 10, 29))
NY = ZoneInfo("America/New_York")
THU = date(2026, 10, 29)
CLOCK = datetime(2026, 10, 29, 8, 0, tzinfo=NY)
NAME = "Dana Reyes"
PHONE = "+12035551234"
OTHER_PHONE = "+12035559999"
PATIENT = Patient(NAME, PHONE)
TOX_DURATION = timedelta(minutes=30)
FILLER_DURATION = timedelta(minutes=45)

FACTORIES: dict[str, Callable[[], BookingAdapter]] = {
    "mock": lambda: MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml",
        tz=GLOW.tz,
        clock=lambda: CLOCK,
    ),
}


@pytest.fixture(params=list(FACTORIES))
def adapter(request: pytest.FixtureRequest) -> BookingAdapter:
    return FACTORIES[request.param]()


def local(hour: int, minute: int = 0, day: date = THU) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=NY)


def as_local(slot: Slot) -> datetime:
    return slot.start.astimezone(NY)


async def list_tox(adapter: BookingAdapter, *, earliest: datetime, limit: int) -> tuple[Slot, ...]:
    return await adapter.list_availability(
        service_id="tox",
        duration=TOX_DURATION,
        room_type="injectable",
        provider_types=("NP", "RN"),
        earliest=earliest,
        limit=limit,
    )


async def list_filler(
    adapter: BookingAdapter, *, earliest: datetime, limit: int
) -> tuple[Slot, ...]:
    return await adapter.list_availability(
        service_id="filler",
        duration=FILLER_DURATION,
        room_type="injectable",
        provider_types=("NP",),
        earliest=earliest,
        limit=limit,
    )


async def book(
    adapter: BookingAdapter,
    slot: Slot,
    *,
    key: str = "key-1",
    patient: Patient = PATIENT,
) -> Booking:
    return await adapter.create_booking(idempotency_key=key, slot=slot, patient=patient)


def assert_utc_slot(slot: Slot, duration: timedelta) -> None:
    assert slot.start.tzinfo is not None and slot.start.utcoffset() == timedelta(0)
    assert slot.end.tzinfo is not None and slot.end.utcoffset() == timedelta(0)
    assert slot.end - slot.start == duration


async def test_create_then_get_round_trips(adapter: BookingAdapter) -> None:
    slots = await list_tox(adapter, earliest=local(9), limit=1)
    assert slots
    created = await book(adapter, slots[0])
    fetched = await adapter.get_booking(created.id)
    assert fetched.id == created.id
    assert fetched.slot.start == slots[0].start


async def test_idempotent_create_returns_original(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    first = await book(adapter, slot, key="same")
    second = await book(adapter, slot, key="same")
    assert first.id == second.id
    later = await list_tox(adapter, earliest=local(9), limit=1)
    assert later[0].start != slot.start or later[0].identity != slot.identity


async def test_same_key_typoed_name_returns_original(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    first = await book(adapter, slot, key="callback")
    second = await book(adapter, slot, key="callback", patient=Patient("Dana Reyez", PHONE))
    assert second.id == first.id


async def test_same_key_different_phone_conflicts(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    first = await book(adapter, slot, key="callback")
    with pytest.raises(IdempotencyConflict):
        await book(adapter, slot, key="callback", patient=Patient(NAME, OTHER_PHONE))
    fetched = await adapter.get_booking(first.id)
    assert fetched.slot.start == slot.start


async def test_same_key_different_slot_conflicts(adapter: BookingAdapter) -> None:
    first_slot, second_slot = await list_tox(adapter, earliest=local(9), limit=2)
    first = await book(adapter, first_slot, key="callback")
    with pytest.raises(IdempotencyConflict):
        await book(adapter, second_slot, key="callback")
    fetched = await adapter.get_booking(first.id)
    assert fetched.slot.identity == first_slot.identity


async def test_stale_version_is_slot_taken(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    await book(adapter, slot, key="a")
    with pytest.raises(SlotTaken):
        await book(adapter, slot, key="b")


async def test_concurrent_create_one_wins(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    results = await asyncio.gather(
        book(adapter, slot, key="racer-a"),
        book(adapter, slot, key="racer-b"),
        return_exceptions=True,
    )
    wins = [item for item in results if not isinstance(item, BaseException)]
    losses = [item for item in results if isinstance(item, SlotTaken)]
    assert len(wins) == 1
    assert len(losses) == 1
    fetched = await adapter.get_booking(wins[0].id)
    assert fetched.slot.identity == slot.identity


async def test_unknown_booking(adapter: BookingAdapter) -> None:
    with pytest.raises(UnknownBooking):
        await adapter.get_booking("bk_missing")


async def test_unavailable_unknown_provider(adapter: BookingAdapter) -> None:
    listed = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    ghost = Slot(
        start=listed.start,
        end=listed.end,
        provider_id="ghost",
        room_id=listed.room_id,
        version=0,
    )
    with pytest.raises(Unavailable):
        await book(adapter, ghost)


async def test_filler_never_returns_rn(adapter: BookingAdapter) -> None:
    slots = await list_filler(adapter, earliest=local(9), limit=8)
    assert slots
    assert all(slot.provider_id != "rn_1" for slot in slots)


async def test_one_slot_per_start_and_relist_after_book(adapter: BookingAdapter) -> None:
    slots = await list_tox(adapter, earliest=local(14), limit=3)
    assert len(slots) == 3
    starts = [slot.start for slot in slots]
    assert len(set(starts)) == 3
    assert as_local(slots[0]) == local(14)
    first = slots[0]
    await book(adapter, first, key="two-pm")
    again = await list_tox(adapter, earliest=local(14), limit=3)
    assert again[0].start == first.start
    assert again[0].identity != first.identity


async def test_occupying_both_pairs_clears_that_start(adapter: BookingAdapter) -> None:
    first = (await list_tox(adapter, earliest=local(14), limit=1))[0]
    await book(adapter, first, key="pair-a")
    second = (await list_tox(adapter, earliest=local(14), limit=1))[0]
    assert second.start == first.start
    await book(adapter, second, key="pair-b")
    third = await list_tox(adapter, earliest=local(14), limit=1)
    assert third[0].start != first.start


async def test_sunday_is_closed(adapter: BookingAdapter) -> None:
    sunday = local(8, day=date(2026, 11, 1))
    slots = await list_tox(adapter, earliest=sunday, limit=4)
    assert slots
    assert all(as_local(slot).weekday() != 6 for slot in slots)


async def test_slots_are_utc_with_duration(adapter: BookingAdapter) -> None:
    tox = await list_tox(adapter, earliest=local(9), limit=2)
    filler = await list_filler(adapter, earliest=local(9), limit=2)
    for slot in tox:
        assert_utc_slot(slot, TOX_DURATION)
    for slot in filler:
        assert_utc_slot(slot, FILLER_DURATION)


async def test_dst_weekend_slots_are_unique_utc(adapter: BookingAdapter) -> None:
    earliest = local(8, day=date(2026, 3, 6))
    slots = await list_tox(adapter, earliest=earliest, limit=20)
    starts = [slot.start for slot in slots]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    for slot in slots:
        assert_utc_slot(slot, TOX_DURATION)


async def test_limit_is_three_distinct_times(adapter: BookingAdapter) -> None:
    slots = await list_tox(adapter, earliest=local(9), limit=3)
    assert len(slots) == 3
    assert len({slot.start for slot in slots}) == 3
    assert [as_local(slot) for slot in slots] == [local(9), local(9, 15), local(9, 30)]


async def test_filler_uses_fifteen_minute_grid(adapter: BookingAdapter) -> None:
    slots = await list_filler(adapter, earliest=local(9), limit=4)
    got = [as_local(slot) for slot in slots]
    assert got == [local(9), local(9, 15), local(9, 30), local(9, 45)]
    assert got[:2] != [local(9), local(9, 45)]


async def test_clock_skips_past_slots(adapter: BookingAdapter) -> None:
    if not isinstance(adapter, MockBookingAdapter):
        pytest.skip("clock control is mock-only")
    late = MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml",
        tz=GLOW.tz,
        clock=lambda: local(14, 7),
    )
    slots = await list_tox(late, earliest=local(9), limit=1)
    assert as_local(slots[0]) >= local(14, 15)


async def test_privacy_reprs_omit_name_and_phone(adapter: BookingAdapter) -> None:
    slot = (await list_tox(adapter, earliest=local(9), limit=1))[0]
    created = await book(adapter, slot)
    shown = " ".join(
        [
            repr(PATIENT),
            str(PATIENT),
            repr(created),
            str(created),
            repr(slot),
            str(slot),
        ]
    )
    assert NAME not in shown
    assert PHONE not in shown
    assert not hasattr(created, "patient")
    with pytest.raises(UnknownBooking) as unknown:
        await adapter.get_booking("bk_missing")
    with pytest.raises(Unavailable) as unavailable:
        await book(
            adapter,
            Slot(
                start=slot.start,
                end=slot.end,
                provider_id="ghost",
                room_id=slot.room_id,
                version=0,
            ),
            key="ghost",
        )
    with pytest.raises(IdempotencyConflict) as conflict:
        await book(adapter, slot, key="key-1", patient=Patient(NAME, OTHER_PHONE))
    for exc in (unknown.value, unavailable.value, conflict.value):
        text = repr(exc) + str(exc)
        assert NAME not in text
        assert PHONE not in text
        assert OTHER_PHONE not in text
