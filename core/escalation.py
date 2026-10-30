"""Choose transfer vs callback vs emergency end, and compose the spa SMS summary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from core.policy import Commit
from core.state import (
    Callback,
    CallState,
    Committed,
    EmergencyEnd,
    EscalationCommittedError,
    EscalationPath,
    Transfer,
)
from core.tenant import Tenant

EMERGENCY_LINE = "If this is an emergency, please hang up and call 911 now."
TRANSFER_LINE = "Let me connect you with the team."
_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
CATEGORY_LABELS: dict[str, str] = {
    "explicit_human_request": "asked for a person",
    "over_turn_budget": "long call",
    "over_duration_budget": "long call",
    "hostility": "difficult call",
    "prompt_injection": "difficult call",
    "legal_or_complaint": "complaint or legal",
}


def category_label(trip: str) -> str:
    """Human label for the SMS. Never taken from caller words."""
    return CATEGORY_LABELS.get(trip, trip.replace("_", " "))


@dataclass(frozen=True)
class SpaSummary:
    category: str
    callback_number: str | None
    number_confirmed: bool
    caller_name: str | None
    local_time: str
    transfer_attempted: bool

    def text(self) -> str:
        lines = [self.category, _number_line(self.callback_number, self.number_confirmed)]
        if self.caller_name is not None:
            lines.append(f"name: {self.caller_name}")
        lines.append(f"time: {self.local_time}")
        lines.append(f"transfer attempted: {'yes' if self.transfer_attempted else 'no'}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SpaSummary(category={self.category!r}, "
            f"has_name={self.caller_name is not None}, "
            f"has_callback={self.callback_number is not None}, "
            f"number_confirmed={self.number_confirmed}, "
            f"local_time={self.local_time!r}, "
            f"transfer_attempted={self.transfer_attempted})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class EscalationPlan:
    path: EscalationPath
    script: str
    end_call: bool
    summary: SpaSummary

    def __repr__(self) -> str:
        return (
            f"EscalationPlan(path={self.path!r}, end_call={self.end_call}, "
            f"summary={self.summary!r})"
        )

    def __str__(self) -> str:
        return repr(self)


def plan(tenant: Tenant, state: CallState, commit: Commit, now: datetime) -> EscalationPlan:
    """Pick one path, record it on first call, and always return one SMS summary."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if isinstance(state.escalation, Committed):
        if state.escalation.trip != commit.trip:
            raise EscalationCommittedError(state.escalation.trip)
        path = state.escalation.path
    else:
        path = _pick(tenant, commit, now)
        state.record_commit(commit.trip, path)
    return _build(path, tenant, state, commit, now)


def _pick(tenant: Tenant, commit: Commit, now: datetime) -> EscalationPath:
    if commit.trip == "medical_emergency":
        return EmergencyEnd()
    live = _first_live_transfer(tenant, now)
    if live is not None:
        return live
    return Callback()


def _first_live_transfer(tenant: Tenant, now: datetime) -> Transfer | None:
    for row in sorted(tenant.transfer_tiers, key=lambda tier: tier.tier):
        if row.kind != "transfer" or not row.answers_live or row.live_hours is None:
            continue
        if not row.live_hours.is_open(now, tenant.tz):
            continue
        if row.target == tenant.main_number:
            raise ValueError("transfer target equals main_number")
        return Transfer(tier=row.tier, target=row.target)
    return None


def _build(
    path: EscalationPath,
    tenant: Tenant,
    state: CallState,
    commit: Commit,
    now: datetime,
) -> EscalationPlan:
    script, end_call = _spoken(path, tenant)
    return EscalationPlan(
        path=path,
        script=script,
        end_call=end_call,
        summary=SpaSummary(
            category=category_label(commit.trip),
            callback_number=state.caller_number,
            number_confirmed=state.caller_number_confirmed,
            caller_name=state.caller_name,
            local_time=_local_time(now, tenant),
            transfer_attempted=isinstance(path, Transfer),
        ),
    )


def _spoken(path: EscalationPath, tenant: Tenant) -> tuple[str, bool]:
    match path:
        case EmergencyEnd():
            return EMERGENCY_LINE, True
        case Transfer():
            return TRANSFER_LINE, False
        case Callback():
            return tenant.callback_promise, True


def _local_time(now: datetime, tenant: Tenant) -> str:
    local = now.astimezone(tenant.tz)
    weekday = _WEEKDAYS[local.weekday()]
    return f"{weekday} {local:%Y-%m-%d %H:%M} {tenant.timezone}"


def _number_line(number: str | None, confirmed: bool) -> str:
    if number is None:
        return "callback: not available"
    if confirmed:
        return f"callback: {number}"
    return f"caller ID: {number}"
