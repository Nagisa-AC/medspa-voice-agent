import ast
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import tools.scheduling
from core.guardrail import Blocked, Pass, check_numbers
from core.numbers import find_numbers
from core.state import CallState, new_call_state
from core.tenant import CONSULT_ID, LiveHours, extra_price_words, load_tenant_file
from integrations.booking import MockBookingAdapter, MockSchedule, Provider, Room, Slot
from tools.scheduling import CheckAvailabilityResult, check_availability, format_slot_time

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NY = ZoneInfo("America/New_York")
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
_GLOW_WORDS = extra_price_words(GLOW)
glow_extract = partial(
    find_numbers,
    extra_price_words=_GLOW_WORDS.extra_price_words,
    service_words=_GLOW_WORDS.service_words,
)
FRIDAY_13 = datetime(2026, 11, 13, 17, 0, tzinfo=NY)


def glow_state(now: datetime = FRIDAY_13) -> CallState:
    return new_call_state(
        tenant=GLOW, call_id="call_1", data_key_id="key_1", now=now, extract=glow_extract
    )


def glow_adapter(clock: datetime) -> MockBookingAdapter:
    return MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml",
        tz=GLOW.tz,
        clock=lambda: clock,
    )


async def ask(
    service_text: str,
    *,
    now: datetime = FRIDAY_13,
    earliest_iso: str | None = None,
    limit: int = 3,
    state: CallState | None = None,
    adapter: MockBookingAdapter | None = None,
) -> tuple[CheckAvailabilityResult, CallState]:
    state = state or glow_state(now)
    adapter = adapter or glow_adapter(now)
    result = await check_availability(
        GLOW,
        state,
        adapter,
        service_text,
        earliest_iso=earliest_iso,
        extract=glow_extract,
        now=now,
        limit=limit,
    )
    return result, state


def model_slots(result: CheckAvailabilityResult) -> tuple[MappingProxyType[str, object], ...]:
    slots = result.model["slots"]
    assert isinstance(slots, tuple)
    return slots


def assert_say_passes(state: CallState, say: str) -> None:
    result = check_numbers(
        say,
        allowed_amounts=state.allowed_amounts,
        allowed_numbers=state.allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(result, Pass)


def ny(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY)


@pytest.mark.parametrize(
    ("when", "spoken"),
    [
        (ny(2026, 1, 1, 14, 0), "Thursday the 1st at 2:00 PM"),
        (ny(2026, 1, 2, 14, 0), "Friday the 2nd at 2:00 PM"),
        (ny(2026, 1, 3, 14, 0), "Saturday the 3rd at 2:00 PM"),
        (ny(2026, 1, 4, 14, 0), "Sunday the 4th at 2:00 PM"),
        (ny(2026, 1, 11, 14, 0), "Sunday the 11th at 2:00 PM"),
        (ny(2026, 1, 12, 14, 0), "Monday the 12th at 2:00 PM"),
        (ny(2026, 1, 13, 14, 0), "Tuesday the 13th at 2:00 PM"),
        (ny(2026, 1, 21, 14, 0), "Wednesday the 21st at 2:00 PM"),
        (ny(2026, 1, 22, 14, 0), "Thursday the 22nd at 2:00 PM"),
        (ny(2026, 1, 23, 14, 0), "Friday the 23rd at 2:00 PM"),
        (ny(2026, 1, 31, 14, 0), "Saturday the 31st at 2:00 PM"),
        (ny(2026, 1, 12, 12, 0), "Monday the 12th at 12:00 PM"),
        (ny(2026, 1, 12, 0, 0), "Monday the 12th at 12:00 AM"),
        (ny(2026, 1, 12, 14, 5), "Monday the 12th at 2:05 PM"),
    ],
)
def test_format_slot_time_ordinals_and_clock(when: datetime, spoken: str) -> None:
    assert format_slot_time(when.astimezone(UTC), NY) == spoken
    assert "14:00" not in spoken


async def test_friday_the_13th_readback_needs_admission() -> None:
    readback = "Friday the 13th at 5:00 PM"
    before = glow_state()
    blocked = check_numbers(
        readback,
        allowed_amounts=before.allowed_amounts,
        allowed_numbers=before.allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(blocked, Blocked)
    assert D(13) not in before.allowed_numbers
    result, state = await ask("botox")
    assert result.say.startswith("I have Friday the 13th at 5:00 PM")
    assert D(13) in state.allowed_numbers
    assert_say_passes(state, readback)
    assert_say_passes(state, result.say)
    assert result.model["found"] is True
    assert result.model["service_id"] == "tox"
    assert result.model["consult"] is False
    slots = result.model["slots"]
    assert isinstance(slots, tuple)
    assert slots[0]["index"] == 1
    assert slots[0]["start"] == "2026-11-13T17:00:00"
    assert slots[0]["say"] == "Friday the 13th at 5:00 PM"


async def test_offered_local_times_stay_inside_glow_hours() -> None:
    result, state = await ask("botox")
    assert len(state.offered_slots) == 3
    for offered in state.offered_slots:
        local = offered.start.astimezone(NY)
        minute = local.hour * 60 + local.minute
        assert local.weekday() != 6
        if local.weekday() == 5:
            assert 10 * 60 <= minute < 16 * 60
        else:
            assert 9 * 60 <= minute < 18 * 60
    assert "Sunday" not in result.say


async def test_spoken_hour_follows_eastern_across_spring_forward() -> None:
    now = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
    result, state = await ask("botox", now=now)
    first = state.offered_slots[0].start
    local = first.astimezone(NY)
    assert local.date() == date(2026, 3, 9)
    assert local.hour == 9
    assert first == datetime(2026, 3, 9, 13, 0, tzinfo=UTC)
    assert "Monday the 9th at 9:00 AM" in result.say
    assert "1:00 PM" not in result.say


async def test_spoken_hour_follows_eastern_across_fall_back() -> None:
    now = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    result, state = await ask("botox", now=now)
    first = state.offered_slots[0].start
    local = first.astimezone(NY)
    assert local.date() == date(2026, 11, 2)
    assert local.hour == 9
    assert first == datetime(2026, 11, 2, 14, 0, tzinfo=UTC)
    assert "Monday the 2nd at 9:00 AM" in result.say
    assert "2:00 PM" not in result.say


async def test_weight_loss_lists_consult_times() -> None:
    result, state = await ask("ozempic")
    assert result.model["found"] is True
    assert result.model["service_id"] == "weightloss"
    assert result.model["consult"] is True
    assert result.say.startswith("I have ")
    assert "don't have that on the list" not in result.say
    assert state.offered_slots[0].service_id == "weightloss"
    assert_say_passes(state, result.say)


async def test_unknown_service_lists_consult() -> None:
    result, state = await ask("something private")
    assert result.say.startswith("I don't have that on the list. I can book a consult. I have ")
    assert result.model["found"] is False
    assert result.model["service_id"] == CONSULT_ID
    assert result.model["consult"] is True
    assert state.offered_slots[0].service_id == CONSULT_ID
    assert_say_passes(state, result.say)


async def test_empty_result_wording() -> None:
    adapter = MockBookingAdapter(
        MockSchedule(
            providers=(Provider(id="np_1", types=frozenset({"NP", "RN"})),),
            rooms=(Room(id="inject_1", types=frozenset({"injectable"})),),
            hours=LiveHours.parse("Mon 09:00-09:15"),
        ),
        tz=GLOW.tz,
        clock=lambda: FRIDAY_13,
    )
    result, state = await ask("botox", adapter=adapter)
    assert result.say == "I don't have anything in the next two weeks."
    assert state.offered_slots == []
    assert result.model["slots"] == ()
    assert_say_passes(state, result.say)


async def test_unknown_empty_keeps_the_list_prefix() -> None:
    adapter = MockBookingAdapter(
        MockSchedule(
            providers=(Provider(id="np_1", types=frozenset({"NP", "RN"})),),
            rooms=(Room(id="inject_1", types=frozenset({"injectable"})),),
            hours=LiveHours.parse("Mon 09:00-09:15"),
        ),
        tz=GLOW.tz,
        clock=lambda: FRIDAY_13,
    )
    result, _state = await ask("something private", adapter=adapter)
    assert result.say == (
        "I don't have that on the list. I don't have anything in the next two weeks."
    )


async def test_ambiguous_does_not_list_or_replace() -> None:
    _first, state = await ask("lip filler")
    offered = list(state.offered_slots)
    assert offered[0].service_id == "filler"

    class Boom:
        called = False

        async def list_availability(self, **_kwargs: object) -> tuple[Slot, ...]:
            self.called = True
            raise AssertionError("ambiguous must not list")

        async def create_booking(self, **_kwargs: object) -> object:
            raise AssertionError("must not book")

        async def get_booking(self, booking_id: str) -> object:
            raise AssertionError("must not get")

    boom = Boom()
    second = await check_availability(
        GLOW,
        state,
        boom,  # type: ignore[arg-type]
        "botox and filler",
        earliest_iso=None,
        extract=glow_extract,
        now=FRIDAY_13,
    )
    assert boom.called is False
    assert second.say == "Which were you asking about, Botox or Dermal Filler?"
    assert second.model["found"] is False
    assert second.model["matches"] == ("tox", "filler")
    assert "service_id" not in second.model
    assert state.offered_slots == offered
    assert_say_passes(state, second.say)


async def test_a_new_offer_replaces_the_previous_menu() -> None:
    _first, state = await ask("lip filler")
    assert {slot.service_id for slot in state.offered_slots} == {"filler"}
    result, state = await ask("botox", state=state)
    assert {slot.service_id for slot in state.offered_slots} == {"tox"}
    assert result.model["service_id"] == "tox"


async def test_two_and_three_slot_joiners() -> None:
    two, _state = await ask("botox", limit=2)
    assert two.say == ("I have Friday the 13th at 5:00 PM or Friday the 13th at 5:15 PM.")
    three, _state = await ask("botox", limit=3)
    assert three.say == (
        "I have Friday the 13th at 5:00 PM, Friday the 13th at 5:15 PM, "
        "or Friday the 13th at 5:30 PM."
    )


async def test_past_or_garbage_earliest_falls_back_to_now() -> None:
    past, _state = await ask("botox", earliest_iso="2020-01-01T00:00:00")
    garbage, _state = await ask("botox", earliest_iso="not a date")
    current, _state = await ask("botox", earliest_iso=None)
    assert past.model["slots"] == current.model["slots"] == garbage.model["slots"]


async def test_naive_future_earliest_is_tenant_local() -> None:
    result, state = await ask("botox", earliest_iso="2026-11-16T10:00:00")
    first = state.offered_slots[0].start.astimezone(NY)
    assert first.date() == date(2026, 11, 16)
    assert first.hour == 10
    assert model_slots(result)[0]["start"] == "2026-11-16T10:00:00"
    assert "Monday the 16th at 10:00 AM" in result.say


async def test_spring_gap_earliest_falls_back_to_now() -> None:
    now = datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
    gap, _state = await ask("botox", now=now, earliest_iso="2026-03-08T02:30:00")
    current, _state = await ask("botox", now=now)
    assert gap.model["slots"] == current.model["slots"]


async def test_limit_is_clamped_to_three() -> None:
    result, state = await ask("botox", limit=50)
    assert len(state.offered_slots) == 3
    assert len(model_slots(result)) == 3


async def test_model_payload_has_no_staffing_ids() -> None:
    result, _state = await ask("botox")
    payload = dict(result.model)
    payload["slots"] = [dict(slot) for slot in model_slots(result)]
    dumped = json.dumps(payload)
    for fragment in ("np_1", "rn_1", "inject_1", "inject_2", "provider", "room_id", "version"):
        assert fragment not in dumped
    assert "say" not in result.model
    assert isinstance(result.model, MappingProxyType)
    slot = model_slots(result)[0]
    assert set(slot) == {"index", "start", "say"}


async def test_result_is_frozen() -> None:
    result, _state = await ask("botox")
    assert isinstance(result, CheckAvailabilityResult)
    with pytest.raises(TypeError):
        result.model["found"] = False  # type: ignore[index]


async def test_naive_now_is_rejected() -> None:
    state = glow_state()
    with pytest.raises(ValueError, match="timezone-aware"):
        await check_availability(
            GLOW,
            state,
            glow_adapter(FRIDAY_13),
            "botox",
            earliest_iso=None,
            extract=glow_extract,
            now=datetime(2026, 11, 13, 17, 0),  # noqa: DTZ001
        )


def test_scheduling_import_layers() -> None:
    tree = ast.parse(Path(tools.scheduling.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections.abc",
        "core.state",
        "core.tenant",
        "dataclasses",
        "datetime",
        "integrations.booking",
        "types",
        "zoneinfo",
    }
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "retell" not in imported
