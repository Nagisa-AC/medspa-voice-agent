import ast
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

import core.escalation
from core.escalation import (
    CATEGORY_LABELS,
    EMERGENCY_LINE,
    TRANSFER_LINE,
    EscalationPlan,
    category_label,
    plan,
)
from core.policy import PACKS_DIR, Commit, load_input_pack
from core.state import (
    Callback,
    CallerLine,
    CallState,
    Committed,
    EmergencyEnd,
    EscalationCommittedError,
    Transfer,
    new_call_state,
)
from core.tenant import Tenant, TransferTier, load_tenant_file

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NY = ZoneInfo("America/New_York")
CHI = ZoneInfo("America/Chicago")
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
SPA = load_tenant_file(
    Path(__file__).parent / "fixtures" / "tenants" / "test_spa.yaml", today=TODAY
)
DESK = "+12035550101"
MANAGER = "+12035550102"
SPA_DESK = "+15555550101"
CALLER = "+12035550148"
CLINICAL_UTTERANCE = "I'm pregnant, can I get Botox if I have a cold"
HUMAN_UTTERANCE = "get me a real person"
CLINICAL_PLANTS = (CLINICAL_UTTERANCE, "pregnant", "Botox", "drooping", "ptosis")
CLINICAL = Commit("clinical_question", "HARD", True)
HUMAN = Commit("explicit_human_request", "HARD", False)
EMERGENCY = Commit("medical_emergency", "HARD", True)
TUE = datetime(2026, 11, 3, 11, 0, tzinfo=NY)
SUN = datetime(2026, 11, 8, 14, 0, tzinfo=NY)
FRI_END = datetime(2026, 11, 6, 18, 0, tzinfo=NY)
FRI_LAST = datetime(2026, 11, 6, 17, 59, tzinfo=NY)
SAT_OPEN = datetime(2026, 11, 7, 10, 0, tzinfo=NY)
SAT_END = datetime(2026, 11, 7, 16, 0, tzinfo=NY)
SAT_BEFORE = datetime(2026, 11, 7, 9, 59, tzinfo=NY)
CHI_OPEN = datetime(2026, 11, 2, 9, 0, tzinfo=CHI)
CHI_END = datetime(2026, 11, 2, 17, 0, tzinfo=CHI)

CLINICAL_CALLBACK_BODY = """\
clinical question
caller ID: +12035550148
name: Maria
time: Sunday 2026-11-08 14:00 America/New_York
transfer attempted: no"""

HUMAN_TRANSFER_BODY = """\
asked for a person
callback: +12035550148
time: Tuesday 2026-11-03 11:00 America/New_York
transfer attempted: yes"""

EMERGENCY_BODY = """\
medical emergency
callback: not available
time: Tuesday 2026-11-03 11:00 America/New_York
transfer attempted: no"""


def utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def fresh(tenant: Tenant = GLOW, **fields: Any) -> CallState:
    state = new_call_state(
        tenant=tenant,
        call_id="call_1",
        data_key_id="key_1",
        now=datetime(2026, 11, 3, 16, 0, tzinfo=UTC),
        extract=lambda _text: (),
    )
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def glow_with(*tiers: TransferTier) -> Tenant:
    sms = GLOW.transfer_tiers[-1]
    return GLOW.model_copy(update={"transfer_tiers": (*tiers, sms)})


# --------------------------------------------------------------------------- path picker


def test_clinical_tuesday_transfers_to_live_desk() -> None:
    state = fresh()
    result = plan(GLOW, state, CLINICAL, TUE)
    assert result.path == Transfer(1, DESK)
    assert result.script == TRANSFER_LINE
    assert result.end_call is False
    assert state.escalation == Committed("clinical_question", Transfer(1, DESK))
    assert state.pending_transfer_number == DESK
    assert result.summary.transfer_attempted is True


def test_clinical_sunday_is_callback() -> None:
    state = fresh()
    result = plan(GLOW, state, CLINICAL, SUN)
    assert result.path == Callback()
    assert result.script == GLOW.callback_promise
    assert result.end_call is True
    assert state.pending_transfer_number is None
    assert result.summary.transfer_attempted is False


def test_friday_end_is_exclusive_callback() -> None:
    assert plan(GLOW, fresh(), CLINICAL, FRI_END).path == Callback()
    assert plan(GLOW, fresh(), CLINICAL, FRI_LAST).path == Transfer(1, DESK)


def test_saturday_window() -> None:
    assert plan(GLOW, fresh(), CLINICAL, SAT_OPEN).path == Transfer(1, DESK)
    assert plan(GLOW, fresh(), CLINICAL, SAT_END).path == Callback()
    assert plan(GLOW, fresh(), CLINICAL, SAT_BEFORE).path == Callback()


def test_no_live_tier_is_callback() -> None:
    desk = GLOW.transfer_tiers[0].model_copy(update={"answers_live": False, "live_hours": None})
    tenant = glow_with(desk, GLOW.transfer_tiers[1])
    result = plan(tenant, fresh(tenant=tenant), CLINICAL, TUE)
    assert result.path == Callback()
    assert result.script == GLOW.callback_promise


def test_emergency_inside_live_hours_never_transfers() -> None:
    state = fresh()
    result = plan(GLOW, state, EMERGENCY, TUE)
    assert result.path == EmergencyEnd()
    assert result.script == EMERGENCY_LINE
    assert result.end_call is True
    assert state.pending_transfer_number is None
    assert result.summary.transfer_attempted is False


def test_emergency_outside_hours_still_ends() -> None:
    result = plan(GLOW, fresh(), EMERGENCY, SUN)
    assert result.path == EmergencyEnd()
    assert result.script == EMERGENCY_LINE
    assert result.end_call is True


def test_manager_cell_and_sms_tier_are_never_transfer_targets() -> None:
    result = plan(GLOW, fresh(), CLINICAL, TUE)
    assert result.path == Transfer(1, DESK)
    assert result.path != Transfer(2, MANAGER)


def test_transfer_target_equal_to_main_number_raises() -> None:
    desk = GLOW.transfer_tiers[0].model_copy(update={"target": GLOW.main_number})
    tenant = glow_with(desk, GLOW.transfer_tiers[1])
    with pytest.raises(ValueError, match="main_number"):
        plan(tenant, fresh(tenant=tenant), CLINICAL, TUE)


def test_chicago_hours_use_tenant_zone() -> None:
    assert plan(SPA, fresh(tenant=SPA), CLINICAL, CHI_OPEN).path == Transfer(1, SPA_DESK)
    assert plan(SPA, fresh(tenant=SPA), CLINICAL, CHI_END).path == Callback()


def test_same_utc_instant_uses_each_tenant_zone() -> None:
    # 14:30 UTC Tuesday = 09:30 NY (open) and 08:30 Chicago (closed).
    now = utc(2026, 11, 3, 14, 30)
    assert plan(GLOW, fresh(), CLINICAL, now).path == Transfer(1, DESK)
    assert plan(SPA, fresh(tenant=SPA), CLINICAL, now).path == Callback()


def test_first_open_live_tier_not_the_closed_one() -> None:
    evening = TransferTier.model_validate(
        {
            "tier": 2,
            "kind": "transfer",
            "target": "+12035550199",
            "label": "evening_desk",
            "answers_live": True,
            "live_hours": "Mon-Fri 18:00-21:00",
        }
    )
    tenant = glow_with(GLOW.transfer_tiers[0], evening)
    result = plan(tenant, fresh(tenant=tenant), CLINICAL, FRI_END)
    assert result.path == Transfer(2, "+12035550199")


def test_lower_tier_wins_when_two_live_tiers_are_open() -> None:
    evening = TransferTier.model_validate(
        {
            "tier": 2,
            "kind": "transfer",
            "target": "+12035550199",
            "label": "evening_desk",
            "answers_live": True,
            "live_hours": "Mon-Fri 09:00-21:00",
        }
    )
    tenant = glow_with(GLOW.transfer_tiers[0], evening)
    result = plan(tenant, fresh(tenant=tenant), CLINICAL, TUE)
    assert result.path == Transfer(1, DESK)


def test_soft_commit_in_hours_still_transfers() -> None:
    result = plan(GLOW, fresh(), Commit("over_turn_budget", "SOFT", False), TUE)
    assert result.path == Transfer(1, DESK)
    assert result.summary.category == "long call"


# --------------------------------------------------------------------------- DST


def test_spring_dst_same_utc_flips_from_callback_to_transfer() -> None:
    # US DST starts Sunday 2026-03-08. 13:30 UTC is 08:30 EST before, 09:30 EDT after.
    assert plan(GLOW, fresh(), CLINICAL, utc(2026, 3, 6, 13, 30)).path == Callback()
    assert plan(GLOW, fresh(), CLINICAL, utc(2026, 3, 9, 13, 30)).path == Transfer(1, DESK)


def test_autumn_dst_same_utc_flips_from_transfer_to_callback() -> None:
    # US DST ends Sunday 2026-11-01. 13:30 UTC is 09:30 EDT before, 08:30 EST after.
    assert plan(GLOW, fresh(), CLINICAL, utc(2026, 10, 30, 13, 30)).path == Transfer(1, DESK)
    assert plan(GLOW, fresh(), CLINICAL, utc(2026, 11, 2, 13, 30)).path == Callback()


def test_naive_now_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        plan(GLOW, fresh(), CLINICAL, datetime(2026, 11, 3, 11, 0))  # noqa: DTZ001


# --------------------------------------------------------------------------- spoken lines


def test_transfer_line_does_not_promise_pickup() -> None:
    result = plan(GLOW, fresh(), HUMAN, TUE)
    assert result.script == TRANSFER_LINE
    lowered = result.script.lower()
    assert "pick up" not in lowered
    assert "will answer" not in lowered
    assert "someone will" not in lowered
    assert result.script != GLOW.callback_promise
    assert GLOW.callback_promise not in result.script


def test_callback_line_is_the_tenant_promise() -> None:
    glow = plan(GLOW, fresh(), CLINICAL, SUN)
    spa = plan(SPA, fresh(tenant=SPA), CLINICAL, CHI_END)
    assert glow.script == GLOW.callback_promise
    assert spa.script == SPA.callback_promise


def test_emergency_line_matches_pack() -> None:
    result = plan(GLOW, fresh(), EMERGENCY, TUE)
    assert result.script == EMERGENCY_LINE
    assert result.script == "If this is an emergency, please hang up and call 911 now."


# --------------------------------------------------------------------------- state / determinism


def test_same_inputs_on_fresh_state_are_equal() -> None:
    first = plan(GLOW, fresh(), CLINICAL, TUE)
    second = plan(GLOW, fresh(), CLINICAL, TUE)
    assert first == second


def test_second_plan_keeps_committed_transfer_on_sunday() -> None:
    state = fresh()
    first = plan(GLOW, state, CLINICAL, TUE)
    second = plan(GLOW, state, CLINICAL, SUN)
    assert first.path == second.path == Transfer(1, DESK)
    assert state.pending_transfer_number == DESK
    assert second.summary.category == first.summary.category
    assert second.summary.transfer_attempted is True


def test_second_plan_with_a_different_trip_raises() -> None:
    state = fresh()
    plan(GLOW, state, CLINICAL, TUE)
    with pytest.raises(EscalationCommittedError):
        plan(GLOW, state, HUMAN, TUE)


def test_failed_transfer_does_not_compose_a_second_path() -> None:
    """M2 ring-out must not call plan again; replay here keeps the first path."""
    state = fresh()
    first = plan(GLOW, state, HUMAN, TUE)
    replay = plan(GLOW, state, HUMAN, TUE)
    assert replay.path == first.path
    assert replay.summary.transfer_attempted is True
    assert replay.summary.category == first.summary.category


# --------------------------------------------------------------------------- summary


def test_clinical_callback_summary_uses_caller_id_when_unconfirmed() -> None:
    state = fresh(caller_name="Maria", caller_number=CALLER)
    state.transcript.append(CallerLine(CLINICAL_UTTERANCE, SUN))
    result = plan(GLOW, state, CLINICAL, SUN)
    assert result.summary.text() == CLINICAL_CALLBACK_BODY
    shown = result.summary.text() + repr(result.summary) + str(result.summary) + repr(result)
    for plant in CLINICAL_PLANTS:
        assert plant not in shown


def test_nonclinical_transfer_summary_uses_callback_when_confirmed() -> None:
    state = fresh(caller_number=CALLER, caller_number_confirmed=True)
    state.transcript.append(CallerLine(HUMAN_UTTERANCE, TUE))
    result = plan(GLOW, state, HUMAN, TUE)
    assert result.summary.text() == HUMAN_TRANSFER_BODY
    assert HUMAN_UTTERANCE not in result.summary.text()
    assert HUMAN_UTTERANCE not in repr(result)
    assert result.summary.category == "asked for a person"


def test_emergency_summary_without_a_number() -> None:
    result = plan(GLOW, fresh(), EMERGENCY, TUE)
    assert result.summary.text() == EMERGENCY_BODY
    assert result.summary.transfer_attempted is False


def test_summary_repr_hides_the_planted_phone() -> None:
    result = plan(GLOW, fresh(caller_name="Maria", caller_number=CALLER), CLINICAL, SUN)
    shown = repr(result.summary) + str(result.summary) + repr(result) + str(result)
    assert CALLER not in shown
    assert "2035550148" not in shown
    assert "203-555-0148" not in shown
    assert CALLER in result.summary.text()
    assert "Maria" not in shown


def test_every_pack_trip_has_a_mapped_or_fallback_label() -> None:
    names = ["over_turn_budget", "over_duration_budget"]
    for path in (PACKS_DIR / "universal.yaml", PACKS_DIR / "medspa.yaml"):
        names.extend(trip.name for trip in load_input_pack(path).tripwires)
    assert names
    for name in names:
        label = category_label(name)
        if name in CATEGORY_LABELS:
            assert label == CATEGORY_LABELS[name]
        else:
            assert label == name.replace("_", " ")
        assert label
        assert "_" not in label


def test_mapped_category_labels() -> None:
    assert category_label("explicit_human_request") == "asked for a person"
    assert category_label("over_turn_budget") == "long call"
    assert category_label("over_duration_budget") == "long call"
    assert category_label("hostility") == "difficult call"
    assert category_label("prompt_injection") == "difficult call"
    assert category_label("legal_or_complaint") == "complaint or legal"
    assert category_label("clinical_question") == "clinical question"
    assert category_label("medical_emergency") == "medical emergency"


# --------------------------------------------------------------------------- layers


def test_escalation_module_imports_are_closed() -> None:
    tree = ast.parse(Path(core.escalation.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "core.policy",
        "core.state",
        "core.tenant",
        "dataclasses",
        "datetime",
    }
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "retell" not in imported
    assert "tools" not in imported


def test_plan_returns_exactly_one_summary_on_every_path() -> None:
    plans = (
        plan(GLOW, fresh(), EMERGENCY, TUE),
        plan(GLOW, fresh(), CLINICAL, TUE),
        plan(GLOW, fresh(), CLINICAL, SUN),
    )
    for result in plans:
        assert isinstance(result, EscalationPlan)
        assert result.summary.text()
        assert result.summary.text().count("transfer attempted:") == 1
