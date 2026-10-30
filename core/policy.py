"""Input tripwire packs and the policy engine that gates every caller utterance."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml

from core.state import Armed, CallState, Committed
from core.tenant import Tenant

_SLUG = re.compile(r"[a-z][a-z0-9_]*")
_PACK_KEYS = frozenset({"tripwires"})
_TRIP_KEYS = frozenset({"name", "outcome", "clinical_class", "script", "patterns"})
_PATTERN_KEYS = frozenset({"id", "regex"})
_APOSTROPHES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)
_KEEP = frozenset("abcdefghijklmnopqrstuvwxyz0123456789'")
PACKS_DIR = Path(__file__).resolve().parents[1] / "packs"
Outcome = Literal["HARD", "SOFT", "REPLY"]


class InputPackError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class InputPattern:
    id: str
    regex: re.Pattern[str]


@dataclass(frozen=True)
class Tripwire:
    name: str
    outcome: Outcome
    clinical_class: bool
    script: str
    patterns: tuple[InputPattern, ...]


@dataclass(frozen=True)
class InputPack:
    tripwires: tuple[Tripwire, ...]


@dataclass(frozen=True)
class Continue:
    def __repr__(self) -> str:
        return "Continue()"


@dataclass(frozen=True)
class Reply:
    trip: str
    outcome: Literal["REPLY"]
    clinical_class: bool
    script: str

    def __repr__(self) -> str:
        return f"Reply(trip={self.trip!r}, outcome='REPLY', clinical_class={self.clinical_class})"


@dataclass(frozen=True)
class Arm:
    trip: str
    outcome: Literal["SOFT"]
    clinical_class: bool

    def __repr__(self) -> str:
        return f"Arm(trip={self.trip!r}, outcome='SOFT', clinical_class={self.clinical_class})"


@dataclass(frozen=True)
class Commit:
    trip: str
    outcome: Literal["HARD", "SOFT"]
    clinical_class: bool

    def __repr__(self) -> str:
        return (
            f"Commit(trip={self.trip!r}, outcome={self.outcome!r}, "
            f"clinical_class={self.clinical_class})"
        )


GateResult = Continue | Reply | Arm | Commit

_TURN_BUDGET = Tripwire(
    name="over_turn_budget",
    outcome="SOFT",
    clinical_class=False,
    script="",
    patterns=(),
)
_DURATION_BUDGET = Tripwire(
    name="over_duration_budget",
    outcome="SOFT",
    clinical_class=False,
    script="",
    patterns=(),
)


def normalize_utterance(text: str) -> str:
    """Fold case, curly quotes, and punctuation so pack regexes see a stable string."""
    folded = text.translate(_APOSTROPHES).lower()
    chars: list[str] = []
    for char in folded:
        if char in _KEEP or char.isspace():
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def load_input_pack(path: Path) -> InputPack:
    """Load one input pack. Invalid regex, duplicates, or missing fields fail closed."""
    source = str(path)
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputPackError(source, f"invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise InputPackError(source, "pack must be a mapping")
    extra = set(raw) - _PACK_KEYS
    if extra:
        raise InputPackError(source, f"unknown fields {sorted(extra)}")
    rows = raw.get("tripwires")
    if not isinstance(rows, list) or not rows:
        raise InputPackError(source, "tripwires must be a non-empty list")
    tripwires: list[Tripwire] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        tripwires.append(_load_tripwire(source, index, item, seen))
    return InputPack(tripwires=tuple(tripwires))


def load_policy_engine(tenant: Tenant, paths: Sequence[Path] | None = None) -> PolicyEngine:
    files = paths or (PACKS_DIR / "universal.yaml", PACKS_DIR / "medspa.yaml")
    return PolicyEngine(tuple(load_input_pack(path) for path in files), tenant)


class PolicyEngine:
    def __init__(self, packs: Sequence[InputPack], tenant: Tenant) -> None:
        trips: list[Tripwire] = []
        seen: set[str] = set()
        for pack in packs:
            for trip in pack.tripwires:
                if trip.name in seen:
                    raise InputPackError("<engine>", f"duplicate trip {trip.name!r}")
                seen.add(trip.name)
                trips.append(trip)
        self._trips = tuple(trips)
        self._by_name = {trip.name: trip for trip in trips}
        self._tenant = tenant

    def evaluate(self, utterance: str, state: CallState, *, now: datetime) -> GateResult:
        """Return one gate result. Does not mutate state."""
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        if isinstance(state.escalation, Committed):
            return self._commit_named(state.escalation.trip)
        hits = self._hits(utterance)
        if state.turn_count >= self._tenant.max_turns:
            hits.append(_TURN_BUDGET)
        elapsed = now - state.started_at
        if elapsed >= timedelta(seconds=self._tenant.max_seconds):
            hits.append(_DURATION_BUDGET)
        if not hits:
            return Continue()
        emergency = [trip for trip in hits if trip.name == "medical_emergency"]
        if emergency:
            return self._commit(emergency[0])
        hards = [trip for trip in hits if trip.outcome == "HARD"]
        if hards:
            return self._commit(hards[0])
        replies = [trip for trip in hits if trip.outcome == "REPLY"]
        if replies:
            return self._reply(replies[0])
        softs = [trip for trip in hits if trip.outcome == "SOFT"]
        return self._soft(softs, state)

    def matched_patterns(self, utterance: str) -> tuple[tuple[str, str], ...]:
        """(trip name, pattern id) pairs for tests. Never includes the utterance."""
        folded = normalize_utterance(utterance)
        found: list[tuple[str, str]] = []
        for trip in self._trips:
            for pattern in trip.patterns:
                if pattern.regex.search(folded):
                    found.append((trip.name, pattern.id))
        return tuple(found)

    def _hits(self, utterance: str) -> list[Tripwire]:
        folded = normalize_utterance(utterance)
        hits: list[Tripwire] = []
        for trip in self._trips:
            if any(pattern.regex.search(folded) for pattern in trip.patterns):
                hits.append(trip)
        return hits

    def _reply(self, trip: Tripwire) -> Reply:
        script = trip.script
        if trip.name == "ai_identity_question":
            script = self._tenant.ai_identity_reply
        return Reply(
            trip=trip.name, outcome="REPLY", clinical_class=trip.clinical_class, script=script
        )

    def _commit(self, trip: Tripwire) -> Commit:
        return Commit(trip=trip.name, outcome="HARD", clinical_class=trip.clinical_class)

    def _commit_named(self, name: str) -> Commit:
        trip = self._by_name.get(name)
        if trip is None:
            return Commit(trip=name, outcome="SOFT", clinical_class=False)
        outcome: Literal["HARD", "SOFT"] = "HARD" if trip.outcome == "HARD" else "SOFT"
        return Commit(trip=name, outcome=outcome, clinical_class=trip.clinical_class)

    def _soft(self, softs: Sequence[Tripwire], state: CallState) -> Arm | Commit:
        armed: frozenset[str] = frozenset()
        if isinstance(state.escalation, Armed):
            armed = state.escalation.trips
        new: list[Tripwire] = []
        seen: set[str] = set()
        for trip in softs:
            if trip.name in armed or trip.name in seen:
                continue
            seen.add(trip.name)
            new.append(trip)
        if not new:
            first = softs[0]
            return Arm(trip=first.name, outcome="SOFT", clinical_class=first.clinical_class)
        if armed:
            trip = new[0]
            return Commit(trip=trip.name, outcome="SOFT", clinical_class=trip.clinical_class)
        if len(new) >= 2:
            trip = new[1]
            return Commit(trip=trip.name, outcome="SOFT", clinical_class=trip.clinical_class)
        trip = new[0]
        return Arm(trip=trip.name, outcome="SOFT", clinical_class=trip.clinical_class)


def _load_tripwire(source: str, index: int, item: object, seen: set[str]) -> Tripwire:
    loc = f"tripwires[{index}]"
    if not isinstance(item, dict):
        raise InputPackError(source, f"{loc} must be a mapping")
    extra = set(item) - _TRIP_KEYS
    if extra:
        raise InputPackError(source, f"{loc} unknown fields {sorted(extra)}")
    name = item.get("name")
    if not isinstance(name, str) or not _SLUG.fullmatch(name):
        raise InputPackError(source, f"{loc}.name must be a slug")
    if name in seen:
        raise InputPackError(source, f"duplicate trip {name!r}")
    seen.add(name)
    outcome_raw = item.get("outcome")
    if outcome_raw == "HARD":
        outcome: Outcome = "HARD"
    elif outcome_raw == "SOFT":
        outcome = "SOFT"
    elif outcome_raw == "REPLY":
        outcome = "REPLY"
    else:
        raise InputPackError(source, f"{loc}.outcome must be HARD, SOFT, or REPLY")
    if "clinical_class" not in item:
        raise InputPackError(source, f"{loc}.clinical_class is required")
    clinical_class = item.get("clinical_class")
    if not isinstance(clinical_class, bool):
        raise InputPackError(source, f"{loc}.clinical_class must be a bool")
    script_raw = item.get("script", "")
    if not isinstance(script_raw, str):
        raise InputPackError(source, f"{loc}.script must be a string")
    script = script_raw.strip()
    if outcome in ("HARD", "REPLY") and not script:
        raise InputPackError(source, f"{loc} {outcome} requires a script")
    patterns_raw = item.get("patterns")
    if not isinstance(patterns_raw, list) or not patterns_raw:
        raise InputPackError(source, f"{loc}.patterns must be a non-empty list")
    patterns: list[InputPattern] = []
    pattern_ids: set[str] = set()
    for pattern_index, pattern_item in enumerate(patterns_raw):
        patterns.append(_load_pattern(source, loc, pattern_index, pattern_item, pattern_ids))
    return Tripwire(
        name=name,
        outcome=outcome,
        clinical_class=clinical_class,
        script=script,
        patterns=tuple(patterns),
    )


def _load_pattern(
    source: str,
    loc: str,
    index: int,
    item: object,
    seen: set[str],
) -> InputPattern:
    ploc = f"{loc}.patterns[{index}]"
    if not isinstance(item, dict):
        raise InputPackError(source, f"{ploc} must be a mapping")
    extra = set(item) - _PATTERN_KEYS
    if extra:
        raise InputPackError(source, f"{ploc} unknown fields {sorted(extra)}")
    pattern_id = item.get("id")
    if not isinstance(pattern_id, str) or not _SLUG.fullmatch(pattern_id):
        raise InputPackError(source, f"{ploc}.id must be a slug")
    if pattern_id in seen:
        raise InputPackError(source, f"{loc} duplicate id {pattern_id!r}")
    seen.add(pattern_id)
    regex_text = item.get("regex")
    if not isinstance(regex_text, str) or not regex_text:
        raise InputPackError(source, f"{ploc}.regex is required")
    try:
        compiled = re.compile(regex_text, re.IGNORECASE)
    except re.error as exc:
        raise InputPackError(source, f"{ploc}.regex invalid: {exc}") from None
    return InputPattern(id=pattern_id, regex=compiled)
