import ast
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import integrations.booking
from core.tenant import LiveHours
from integrations.booking import (
    MockBookingAdapter,
    MockSchedule,
    MockScheduleError,
    Patient,
    Provider,
    Room,
    Slot,
    SlotTaken,
    UnknownBooking,
    load_mock_schedule,
)

NY = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[2]
NAME = "Dana Reyes"
PHONE = "+12035551234"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "spa.mock.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _night_adapter(clock: datetime) -> MockBookingAdapter:
    schedule = MockSchedule(
        providers=(Provider(id="np_1", types=frozenset({"NP"})),),
        rooms=(Room(id="inject_1", types=frozenset({"injectable"})),),
        hours=LiveHours.parse("Sun 01:00-04:00"),
    )
    return MockBookingAdapter(schedule, tz=NY, clock=lambda: clock)


async def _list_night(
    adapter: MockBookingAdapter, earliest: datetime, limit: int = 20
) -> tuple[Slot, ...]:
    return await adapter.list_availability(
        service_id="tox",
        duration=timedelta(minutes=15),
        room_type="injectable",
        provider_types=("NP",),
        earliest=earliest,
        limit=limit,
    )


def test_glow_mock_loads() -> None:
    schedule = load_mock_schedule(ROOT / "tenants" / "glow_aesthetics.mock.yaml")
    assert [p.id for p in schedule.providers] == ["np_1", "rn_1"]
    assert [r.id for r in schedule.rooms] == ["inject_1", "inject_2"]


def test_load_missing_hours(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "providers:\n  - id: np_1\n    types: [NP]\nrooms:\n  - id: r1\n    types: [a]\n",
    )
    with pytest.raises(MockScheduleError, match="hours is required"):
        load_mock_schedule(path)


def test_load_duplicate_provider_id(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "hours: 'Mon 09:00-10:00'\n"
        "providers:\n  - id: np_1\n    types: [NP]\n  - id: np_1\n    types: [RN]\n"
        "rooms:\n  - id: r1\n    types: [a]\n",
    )
    with pytest.raises(MockScheduleError, match="duplicate id 'np_1'"):
        load_mock_schedule(path)


def test_load_unknown_field(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "hours: 'Mon 09:00-10:00'\nproviders:\n  - id: np_1\n    types: [NP]\n"
        "rooms:\n  - id: r1\n    types: [a]\ntimezone: America/New_York\n",
    )
    with pytest.raises(MockScheduleError, match="unknown fields"):
        load_mock_schedule(path)


def test_load_invalid_hours(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "hours: 'whenever'\nproviders:\n  - id: np_1\n    types: [NP]\n"
        "rooms:\n  - id: r1\n    types: [a]\n",
    )
    with pytest.raises(MockScheduleError, match="hours:"):
        load_mock_schedule(path)


async def test_race_hook_slot_taken_without_booking() -> None:
    adapter = MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml",
        tz=NY,
        clock=lambda: datetime(2026, 10, 29, 8, 0, tzinfo=NY),
    )
    slots = await adapter.list_availability(
        service_id="tox",
        duration=timedelta(minutes=30),
        room_type="injectable",
        provider_types=("NP", "RN"),
        earliest=datetime(2026, 10, 29, 9, 0, tzinfo=NY),
        limit=1,
    )
    slot = slots[0]
    adapter.fail_next_create(slot)
    with pytest.raises(SlotTaken):
        await adapter.create_booking(
            idempotency_key="k",
            slot=slot,
            patient=Patient(NAME, PHONE),
        )
    with pytest.raises(UnknownBooking):
        await adapter.get_booking("bk_00001")


async def test_hide_on_next_get_then_recovers() -> None:
    adapter = MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml",
        tz=NY,
        clock=lambda: datetime(2026, 10, 29, 8, 0, tzinfo=NY),
    )
    slot = (
        await adapter.list_availability(
            service_id="tox",
            duration=timedelta(minutes=30),
            room_type="injectable",
            provider_types=("NP", "RN"),
            earliest=datetime(2026, 10, 29, 9, 0, tzinfo=NY),
            limit=1,
        )
    )[0]
    created = await adapter.create_booking(
        idempotency_key="k",
        slot=slot,
        patient=Patient(NAME, PHONE),
    )
    adapter.hide_on_next_get(created.id)
    with pytest.raises(UnknownBooking):
        await adapter.get_booking(created.id)
    again = await adapter.get_booking(created.id)
    assert again.id == created.id


async def test_spring_forward_skips_the_gap() -> None:
    adapter = _night_adapter(datetime(2026, 3, 8, 0, 30, tzinfo=NY))
    day = date(2026, 3, 8)
    slots = [
        slot
        for slot in await _list_night(adapter, datetime(2026, 3, 8, 1, 0, tzinfo=NY))
        if slot.start.astimezone(NY).date() == day
    ]
    hours = [slot.start.astimezone(NY).hour for slot in slots]
    assert 1 in hours
    assert 3 in hours
    assert 2 not in hours
    assert len({slot.start for slot in slots}) == len(slots)


async def test_fall_back_emits_both_folds() -> None:
    adapter = _night_adapter(datetime(2026, 11, 1, 0, 30, tzinfo=NY))
    day = date(2026, 11, 1)
    slots = [
        slot
        for slot in await _list_night(adapter, datetime(2026, 11, 1, 1, 0, tzinfo=NY))
        if slot.start.astimezone(NY).date() == day
    ]
    ones = [
        slot.start
        for slot in slots
        if slot.start.astimezone(NY).hour == 1 and slot.start.astimezone(NY).minute == 0
    ]
    assert len(ones) == 2
    assert ones[0] != ones[1]
    ordered = sorted(slots, key=lambda item: (item.start, item.provider_id, item.room_id))
    assert slots == ordered


def test_booking_import_layers() -> None:
    tree = ast.parse(Path(integrations.booking.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "asyncio",
        "collections.abc",
        "core.tenant",
        "dataclasses",
        "datetime",
        "pathlib",
        "re",
        "typing",
        "yaml",
        "zoneinfo",
    }
    assert "core.state" not in imported
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "retell" not in imported
    assert "logging" not in imported
