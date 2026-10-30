import ast
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import tools.scheduling
from core.guardrail import Blocked, Pass, check_numbers
from core.numbers import find_numbers
from core.state import CallState, OfferedSlot, new_call_state
from core.tenant import CONSULT_ID, LiveHours, extra_price_words, load_tenant_file
from integrations.booking import MockBookingAdapter, MockSchedule, Provider, Room, Slot
from tools.scheduling import (
    BookResult,
    CheckAvailabilityResult,
    book,
    check_availability,
    format_slot_time,
    normalize_phone,
)

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
SECRET = b"n" * 32
NAME = "Dana Reyes"
PHONE = "+12035551234"
REVIEW = "I've got your request in and someone from the team will confirm with you."


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


def slot_iso(result: CheckAvailabilityResult, index: int = 0) -> str:
    start = model_slots(result)[index]["start"]
    assert isinstance(start, str)
    return start


def assert_say_passes(state: CallState, say: str) -> None:
    result = check_numbers(
        say,
        allowed_amounts=state.allowed_amounts,
        allowed_numbers=state.allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(result, Pass)


async def seed(
    service_text: str,
    *,
    earliest_iso: str | None = None,
    now: datetime = FRIDAY_13,
    adapter: MockBookingAdapter | None = None,
    state: CallState | None = None,
) -> tuple[CheckAvailabilityResult, CallState, MockBookingAdapter]:
    adapter = adapter or glow_adapter(now)
    listed, state = await ask(
        service_text, now=now, earliest_iso=earliest_iso, state=state, adapter=adapter
    )
    return listed, state, adapter


async def book_it(
    state: CallState,
    adapter: MockBookingAdapter,
    slot_start_iso: str,
    *,
    name: str = NAME,
    phone: str = PHONE,
    confirmed: bool = True,
    now: datetime = FRIDAY_13,
    secret: bytes = SECRET,
) -> BookResult:
    return await book(
        GLOW,
        state,
        adapter,
        slot_start_iso=slot_start_iso,
        patient_name=name,
        patient_phone=phone,
        confirmed=confirmed,
        extract=glow_extract,
        now=now,
        secret=secret,
    )


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
        "hashlib",
        "hmac",
        "integrations.booking",
        "re",
        "tools.pricing",
        "types",
        "zoneinfo",
    }
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "retell" not in imported


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("(203) 555-1234", PHONE),
        ("203.555.1234", PHONE),
        ("1 203 555 1234", PHONE),
        (PHONE, PHONE),
        ("  +12035551234  ", PHONE),
    ],
)
def test_normalize_phone_accepts_nanp(raw: str, expected: str) -> None:
    assert normalize_phone(raw) == expected


def test_normalize_phone_rejects_local_seven() -> None:
    assert normalize_phone("555-1234") is None


def test_normalize_phone_rejects_short_plus_one() -> None:
    assert normalize_phone("+12") is None


def test_normalize_phone_rejects_area_code_starting_with_zero() -> None:
    assert normalize_phone("(023) 555-1234") is None


def test_normalize_phone_accepts_non_nanp_e164() -> None:
    assert normalize_phone("+442071234567") == "+442071234567"


async def test_book_filler_success_and_deposit() -> None:
    listed, state, adapter = await seed("lip filler", earliest_iso="2026-11-16T14:00:00")
    iso = slot_iso(listed)
    assert iso == "2026-11-16T14:00:00"
    result = await book_it(state, adapter, iso)
    assert result.say == (
        "You're all set for Dermal Filler on Monday the 16th at 2:00 PM. "
        "Just so you know, there's a $100 deposit for this treatment."
    )
    assert "text" not in result.say.lower()
    assert result.model == {
        "booked": True,
        "needs_review": False,
        "retry": False,
        "already_spoken": True,
    }
    assert D(100) in state.allowed_amounts
    assert_say_passes(state, result.say)
    assert iso in {_local(s.start) for s in state.offered_slots}
    assert len(adapter._bookings) == 1


def _local(start: datetime) -> str:
    return start.astimezone(NY).strftime("%Y-%m-%dT%H:%M:%S")


async def test_book_consult_wording() -> None:
    listed, state, adapter = await seed("ozempic", earliest_iso="2026-11-16T14:00:00")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso)
    assert result.say == "You're all set for a consult on Monday the 16th at 2:00 PM."
    assert "deposit" not in result.say
    assert result.model["booked"] is True
    assert_say_passes(state, result.say)


async def test_book_unknown_is_a_consult() -> None:
    listed, state, adapter = await seed("something private", earliest_iso="2026-11-16T14:00:00")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso)
    assert result.say == "You're all set for a consult on Monday the 16th at 2:00 PM."
    assert result.model["booked"] is True


async def test_book_refuses_without_confirmation() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    offered = list(state.offered_slots)
    result = await book_it(state, adapter, iso, confirmed=False)
    assert result.say == (
        "Just to confirm: Botox on Friday the 13th at 5:00 PM. Should I go ahead and book that?"
    )
    assert result.model["booked"] is False
    assert adapter._bookings == {}
    assert state.offered_slots == offered


async def test_book_refuses_blank_name() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso, name="  ")
    assert result.say == "I need a name for the booking."
    assert adapter._bookings == {}


async def test_book_reasks_short_phone() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    offered = list(state.offered_slots)
    result = await book_it(state, adapter, iso, phone="555-1234")
    assert result.say == "Can I get a good number to reach you at?"
    assert result.model["booked"] is False
    assert adapter._bookings == {}
    assert state.offered_slots == offered


@pytest.mark.parametrize("phone", ["(203) 555-1234", "203.555.1234", "1 203 555 1234", PHONE])
async def test_book_normalizes_nanp_phone(phone: str) -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso, phone=phone)
    assert result.model["booked"] is True
    booking_id = next(iter(adapter._bookings))
    assert adapter._patients[booking_id] == PHONE


@pytest.mark.parametrize("phone", ["+12", "(023) 555-1234"])
async def test_book_reasks_invalid_plus_one(phone: str) -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    offered = list(state.offered_slots)
    result = await book_it(state, adapter, iso, phone=phone)
    assert result.say == "Can I get a good number to reach you at?"
    assert result.model["booked"] is False
    assert adapter._bookings == {}
    assert state.offered_slots == offered


async def test_book_accepts_non_nanp_e164() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso, phone="+442071234567")
    assert result.model["booked"] is True
    booking_id = next(iter(adapter._bookings))
    assert adapter._patients[booking_id] == "+442071234567"


async def test_book_with_no_offer_does_not_guess() -> None:
    state = glow_state()
    assert state.offered_slots == []

    class Boom:
        listed = False

        async def list_availability(self, **_kwargs: object) -> tuple[Slot, ...]:
            self.listed = True
            raise AssertionError("empty offer must not list")

        async def create_booking(self, **_kwargs: object) -> object:
            raise AssertionError("must not book")

        async def get_booking(self, booking_id: str) -> object:
            raise AssertionError("must not get")

    boom = Boom()
    result = await book(
        GLOW,
        state,
        boom,  # type: ignore[arg-type]
        slot_start_iso="2026-11-13T17:00:00",
        patient_name=NAME,
        patient_phone=PHONE,
        confirmed=True,
        extract=glow_extract,
        now=FRIDAY_13,
        secret=SECRET,
    )
    assert result.say == "Let me find you a time. What would you like to book?"
    assert result.model["retry"] is True
    assert result.model["booked"] is False
    assert boom.listed is False
    assert state.offered_slots == []
    assert "consult" not in result.say.lower()
    assert "I have" not in result.say
    assert_say_passes(state, result.say)


async def test_book_not_in_offer_relists() -> None:
    _listed, state, adapter = await seed("botox")
    result = await book_it(state, adapter, "2026-11-16T14:00:00")
    assert result.say.startswith("I don't have that time held. I have ")
    assert result.model["retry"] is True
    assert result.model["booked"] is False
    assert adapter._bookings == {}
    assert state.offered_slots
    assert_say_passes(state, result.say)


async def test_book_dst_collision_relists_without_guessing() -> None:
    _listed, state, adapter = await seed("botox")
    first = state.offered_slots[0]
    fold0 = datetime(2026, 11, 1, 1, 0, tzinfo=NY)
    fold1 = fold0.replace(fold=1)
    state.offered_slots = [
        OfferedSlot(
            service_id=first.service_id,
            start=fold0.astimezone(UTC),
            end=fold0.astimezone(UTC) + timedelta(minutes=30),
            provider_id="np_1",
            room_id="inject_1",
            version=0,
        ),
        OfferedSlot(
            service_id=first.service_id,
            start=fold1.astimezone(UTC),
            end=fold1.astimezone(UTC) + timedelta(minutes=30),
            provider_id="rn_1",
            room_id="inject_2",
            version=0,
        ),
    ]
    result = await book_it(state, adapter, "2026-11-01T01:00:00")
    assert result.say.startswith("That clock time matched more than one opening. I have ")
    assert result.model["retry"] is True
    assert adapter._bookings == {}


async def test_book_slot_taken_relists() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    adapter.fail_next_create(
        Slot(
            start=state.offered_slots[0].start,
            end=state.offered_slots[0].end,
            provider_id=state.offered_slots[0].provider_id,
            room_id=state.offered_slots[0].room_id,
            version=state.offered_slots[0].version,
        )
    )
    result = await book_it(state, adapter, iso)
    assert result.say.startswith("That time was just taken. I have ")
    assert result.model["retry"] is True
    assert adapter._bookings == {}


async def test_book_unavailable_relists() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    original = state.offered_slots[0]
    state.offered_slots = [
        OfferedSlot(
            service_id=original.service_id,
            start=original.start,
            end=original.end,
            provider_id="no_such_provider",
            room_id="no_such_room",
            version=0,
        )
    ]
    result = await book_it(state, adapter, iso)
    assert result.say.startswith("That time isn't open anymore. I have ")
    assert result.model["retry"] is True
    assert adapter._bookings == {}


async def test_book_failed_readback_never_says_booked() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    adapter.hide_on_next_get("bk_00001")
    result = await book_it(state, adapter, iso)
    assert result.say == REVIEW
    assert "booked" not in result.say.lower()
    assert result.model["booked"] is False
    assert result.model["needs_review"] is True
    assert len(adapter._bookings) == 1


async def test_idempotent_recall_returns_same_confirmation() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    first = await book_it(state, adapter, iso)
    second = await book_it(state, adapter, iso)
    assert first.say == second.say
    assert first.model["booked"] is True
    assert second.model["booked"] is True
    assert len(adapter._bookings) == 1


async def test_idempotency_conflict_is_needs_review() -> None:
    listed, state, adapter = await seed("lip filler", earliest_iso="2026-11-16T14:00:00")
    iso = slot_iso(listed)
    assert iso == "2026-11-16T14:00:00"
    first = await book_it(state, adapter, iso)
    assert first.model["booked"] is True
    original = state.offered_slots[0]
    other_provider = "rn_1" if original.provider_id == "np_1" else "np_1"
    other_room = "inject_2" if original.room_id == "inject_1" else "inject_1"
    state.offered_slots = [
        OfferedSlot(
            service_id=original.service_id,
            start=original.start,
            end=original.end,
            provider_id=other_provider,
            room_id=other_room,
            version=original.version,
        )
    ]
    second = await book_it(state, adapter, iso)
    assert second.say == REVIEW
    assert second.model["needs_review"] is True
    assert second.model["booked"] is False
    assert "booking" not in second.say.lower()
    assert "under that number" not in second.say
    assert len(adapter._bookings) == 1


async def test_book_privacy_omits_name_phone_key_and_secret() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    result = await book_it(state, adapter, iso)
    dumped = repr(result) + str(result) + json.dumps(dict(result.model))
    key = next(iter(state.idempotency_keys))
    for fragment in (NAME, PHONE, "2035551234", SECRET.decode(), key, "bk_"):
        assert fragment not in dumped
    assert "say" not in result.model
    assert set(result.model) == {"booked", "needs_review", "retry", "already_spoken"}


async def test_short_secret_fails_closed() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    with pytest.raises(ValueError, match="32 bytes"):
        await book_it(state, adapter, iso, secret=b"too-short")
    assert adapter._bookings == {}


async def test_book_naive_now_is_rejected() -> None:
    listed, state, adapter = await seed("botox")
    iso = slot_iso(listed)
    with pytest.raises(ValueError, match="timezone-aware"):
        await book_it(
            state,
            adapter,
            iso,
            now=datetime(2026, 11, 13, 17, 0),  # noqa: DTZ001
        )


async def test_book_confirm_without_unique_slot() -> None:
    _listed, state, adapter = await seed("botox")
    offered = list(state.offered_slots)
    result = await book_it(state, adapter, "2026-11-16T14:00:00", confirmed=False)
    assert result.say == "Just to confirm — should I go ahead and book that?"
    assert result.model["booked"] is False
    assert adapter._bookings == {}
    assert state.offered_slots == offered
