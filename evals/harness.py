"""Load YAML eval cases, run them through TurnLoop, print a privacy-safe report."""

from __future__ import annotations

import asyncio
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from functools import cache, partial
from pathlib import Path
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import yaml

from core.crypto import InMemoryKeyStore
from core.events import Event, InMemoryEventLog
from core.guardrail import Blocked, OutputPack, check_sentence, load_output_pack
from core.llm import ScriptedLLM
from core.numbers import find_numbers
from core.policy import load_input_pack, load_policy_engine
from core.state import (
    CallerLine,
    CallState,
    Committed,
    InMemorySessionStore,
    Transfer,
    admit_numbers,
    new_call_state,
)
from core.tenant import extra_price_words, load_tenant_file
from core.turn import NUMBER_HEDGE, OutputChunk, TurnLoop
from integrations.booking import MockBookingAdapter, Slot

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = Path(__file__).parent / "cases"
EVALS_DIR = Path(__file__).parent
PACKS_DIR = ROOT / "packs"
DEFAULT_CALLER = "+12035550148"
SECRET = b"n" * 32
_SLUG = re.compile(r"[a-z][a-z0-9_]*")
_CASE_KEYS = frozenset(
    {
        "id",
        "tenant",
        "clock",
        "started",
        "caller_id",
        "utterances",
        "script",
        "script_path",
        "expect",
        "hooks",
    }
)
_EXPECT_KEYS = frozenset({"spoken", "committed", "events", "model_called"})
_COMMITTED_KEYS = frozenset({"trip", "path"})
_EVENTS_KEYS = frozenset({"present", "absent"})
_HOOK_KEYS = frozenset({"after", "fail_next_create", "hide_next_booking", "rewrite_offer"})
_PATHS = frozenset({"emergency_end", "transfer", "callback"})
_TENANTS = frozenset({"glow", "test_spa"})
_TEXT_PROMISE = re.compile(
    r"\b(?:i will|i'll|we will|we'll|let me)\s+text\b"
    r"|\btext you\b"
    r"|\bsend you a text\b"
    r"|\bsms you\b",
    re.IGNORECASE,
)
_TABLE_KEYS = frozenset({"rows"})
PathKind = Literal["emergency_end", "transfer", "callback"]
Origin = Literal["backend", "model"]


class CaseLoadError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class SpokenExpect:
    kind: Literal["exact", "contains"]
    text: str


@dataclass(frozen=True)
class CommittedExpect:
    trip: str
    path: PathKind


@dataclass(frozen=True)
class EventExpect:
    present: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hooks:
    after: int = 0
    fail_next_create: bool = False
    hide_next_booking: bool = False
    rewrite_offer: Literal["unavailable", "dst"] | None = None


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    tenant: str
    clock: datetime
    started: datetime
    caller_id: str | None
    utterances: tuple[str, ...]
    script: Mapping[str, object] | None
    script_path: str | None
    spoken: tuple[SpokenExpect, ...]
    committed: CommittedExpect | None
    events: EventExpect
    model_called: bool | None
    hooks: Hooks
    never_spoken: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class CaseFailure:
    category: str
    id: str
    check: str


@dataclass
class Report:
    results: list[tuple[EvalCase, CaseFailure | None]] = field(default_factory=list)

    def add(self, case: EvalCase, failure: CaseFailure | None) -> None:
        self.results.append((case, failure))


def load_case_path(path: Path) -> EvalCase:
    loaded = _load_file(path, path.parent)
    if len(loaded) != 1:
        raise CaseLoadError(str(path), "expected one case")
    return loaded[0]


def load_cases(cases_dir: Path = CASES_DIR) -> list[EvalCase]:
    if not cases_dir.is_dir():
        raise CaseLoadError(str(cases_dir), "cases directory is missing")
    found: list[EvalCase] = []
    for path in sorted(cases_dir.glob("*/*.yaml")):
        found.extend(_load_file(path, cases_dir))
    return found


def _load_file(path: Path, cases_dir: Path) -> list[EvalCase]:
    source = str(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise CaseLoadError(source, f"invalid YAML: {exc}") from None
    if not isinstance(raw, dict):
        raise CaseLoadError(source, "case must be a mapping")
    category = path.parent.name
    if not _SLUG.fullmatch(category.replace("-", "_")):
        raise CaseLoadError(source, "category must be a slug")
    if path.name == "sentences.yaml":
        return _load_sentences(raw, category, path, cases_dir)
    return [_load_case(raw, category, path.stem, source, cases_dir)]


def _load_sentences(
    raw: Mapping[str, object], category: str, path: Path, cases_dir: Path
) -> list[EvalCase]:
    extra = set(raw) - _TABLE_KEYS
    if extra:
        raise CaseLoadError(str(path), f"unknown fields {sorted(extra)}")
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise CaseLoadError(str(path), "rows must be a non-empty list")
    stem = path.stem
    cases: list[EvalCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CaseLoadError(str(path), f"rows[{index}] must be a mapping")
        extra_row = set(row) - {"text", "pattern"}
        if extra_row:
            raise CaseLoadError(str(path), f"rows[{index}] unknown fields {sorted(extra_row)}")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CaseLoadError(str(path), f"rows[{index}].text must be a non-empty string")
        if category == "clinical_output":
            pattern = row.get("pattern")
            if not isinstance(pattern, str) or not pattern:
                raise CaseLoadError(str(path), f"rows[{index}].pattern is required")
        case_id = f"{stem}_{index}"
        replacement = NUMBER_HEDGE if category == "price_invention" else _deflect()
        script = {
            "turns": [
                {
                    "id": "invent",
                    "events": [
                        {"type": "text", "text": text},
                        {"type": "done"},
                    ],
                }
            ]
        }
        cases.append(
            EvalCase(
                id=case_id,
                category=category,
                tenant="glow",
                clock=_parse_clock("2026-11-03T11:00:00-05:00", str(path), "clock"),
                started=_parse_clock("2026-11-03T11:00:00-05:00", str(path), "started"),
                caller_id=DEFAULT_CALLER,
                utterances=("tell me about filler",),
                script=script,
                script_path=None,
                spoken=(SpokenExpect("contains", replacement),),
                committed=None,
                events=EventExpect(),
                model_called=None,
                hooks=Hooks(),
                never_spoken=(text,),
                source=f"{path}:{index}",
            )
        )
    return cases


def _deflect() -> str:
    pack = load_output_pack(PACKS_DIR / "medspa.output.yaml")
    return pack.deflect


def _load_case(
    raw: Mapping[str, object],
    category: str,
    stem: str,
    source: str,
    cases_dir: Path,
) -> EvalCase:
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise CaseLoadError(source, f"unknown fields {sorted(extra)}")
    case_id = raw.get("id", stem)
    if not isinstance(case_id, str) or not _SLUG.fullmatch(case_id):
        raise CaseLoadError(source, "id must be a slug")
    tenant = raw.get("tenant", "glow")
    if tenant not in _TENANTS:
        raise CaseLoadError(source, "tenant must be glow or test_spa")
    clock_raw = raw.get("clock")
    if not isinstance(clock_raw, str):
        raise CaseLoadError(source, "clock is required")
    clock = _parse_clock(clock_raw, source, "clock")
    started_raw = raw.get("started", clock_raw)
    if not isinstance(started_raw, str):
        raise CaseLoadError(source, "started must be an ISO timestamp")
    started = _parse_clock(started_raw, source, "started")
    if "caller_id" not in raw:
        caller_id: str | None = DEFAULT_CALLER
    else:
        caller_raw = raw.get("caller_id")
        if caller_raw is None:
            caller_id = None
        elif isinstance(caller_raw, str) and caller_raw:
            caller_id = caller_raw
        else:
            raise CaseLoadError(source, "caller_id must be a string or null")
    utterances = raw.get("utterances")
    if not isinstance(utterances, list) or not utterances:
        raise CaseLoadError(source, "utterances must be a non-empty list")
    if not all(isinstance(item, str) and item for item in utterances):
        raise CaseLoadError(source, "utterances must be non-empty strings")
    script = raw.get("script")
    script_path = raw.get("script_path")
    if script is not None and script_path is not None:
        raise CaseLoadError(source, "script and script_path cannot both be set")
    if script is None and script_path is None:
        raise CaseLoadError(source, "script or script_path is required")
    if script is not None and not isinstance(script, dict):
        raise CaseLoadError(source, "script must be a mapping")
    if script_path is not None and not isinstance(script_path, str):
        raise CaseLoadError(source, "script_path must be a string")
    expect = raw.get("expect", {})
    if expect is None:
        expect = {}
    if not isinstance(expect, dict):
        raise CaseLoadError(source, "expect must be a mapping")
    extra_expect = set(expect) - _EXPECT_KEYS
    if extra_expect:
        raise CaseLoadError(source, f"unknown expect fields {sorted(extra_expect)}")
    spoken = _load_spoken(expect.get("spoken", []), source)
    committed = _load_committed(expect.get("committed"), source)
    events = _load_events(expect.get("events"), source)
    model_called = expect.get("model_called")
    if model_called is not None and not isinstance(model_called, bool):
        raise CaseLoadError(source, "expect.model_called must be a boolean")
    hooks = _load_hooks(raw.get("hooks"), source)
    _ = cases_dir
    return EvalCase(
        id=case_id,
        category=category,
        tenant=str(tenant),
        clock=clock,
        started=started,
        caller_id=caller_id,
        utterances=tuple(utterances),
        script=script,
        script_path=script_path,
        spoken=spoken,
        committed=committed,
        events=events,
        model_called=model_called,
        hooks=hooks,
        never_spoken=(),
        source=source,
    )


def _load_spoken(raw: object, source: str) -> tuple[SpokenExpect, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CaseLoadError(source, "expect.spoken must be a list")
    found: list[SpokenExpect] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CaseLoadError(source, f"expect.spoken[{index}] must be a mapping")
        keys = set(item)
        if keys == {"exact"} and isinstance(item["exact"], str):
            found.append(SpokenExpect("exact", item["exact"]))
            continue
        if keys == {"contains"} and isinstance(item["contains"], str):
            found.append(SpokenExpect("contains", item["contains"]))
            continue
        raise CaseLoadError(source, f"expect.spoken[{index}] must be exact or contains")
    return tuple(found)


def _load_committed(raw: object, source: str) -> CommittedExpect | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CaseLoadError(source, "expect.committed must be a mapping")
    extra = set(raw) - _COMMITTED_KEYS
    if extra:
        raise CaseLoadError(source, f"unknown committed fields {sorted(extra)}")
    trip = raw.get("trip")
    path = raw.get("path")
    if not isinstance(trip, str) or not trip:
        raise CaseLoadError(source, "expect.committed.trip is required")
    if path not in _PATHS:
        raise CaseLoadError(source, "expect.committed.path is invalid")
    return CommittedExpect(trip=trip, path=path)


def _load_events(raw: object, source: str) -> EventExpect:
    if raw is None:
        return EventExpect()
    if not isinstance(raw, dict):
        raise CaseLoadError(source, "expect.events must be a mapping")
    extra = set(raw) - _EVENTS_KEYS
    if extra:
        raise CaseLoadError(source, f"unknown events fields {sorted(extra)}")
    present = raw.get("present", [])
    absent = raw.get("absent", [])
    if not isinstance(present, list) or not all(isinstance(item, str) for item in present):
        raise CaseLoadError(source, "expect.events.present must be a list of strings")
    if not isinstance(absent, list) or not all(isinstance(item, str) for item in absent):
        raise CaseLoadError(source, "expect.events.absent must be a list of strings")
    return EventExpect(present=tuple(present), absent=tuple(absent))


def _load_hooks(raw: object, source: str) -> Hooks:
    if raw is None:
        return Hooks()
    if not isinstance(raw, dict):
        raise CaseLoadError(source, "hooks must be a mapping")
    extra = set(raw) - _HOOK_KEYS
    if extra:
        raise CaseLoadError(source, f"unknown hooks fields {sorted(extra)}")
    after = raw.get("after", 0)
    if not isinstance(after, int) or isinstance(after, bool) or after < 0:
        raise CaseLoadError(source, "hooks.after must be an int >= 0")
    fail_next = raw.get("fail_next_create", False)
    hide_next = raw.get("hide_next_booking", False)
    rewrite = raw.get("rewrite_offer")
    if not isinstance(fail_next, bool) or not isinstance(hide_next, bool):
        raise CaseLoadError(source, "hooks flags must be booleans")
    if rewrite is not None and rewrite not in {"unavailable", "dst"}:
        raise CaseLoadError(source, "hooks.rewrite_offer is invalid")
    return Hooks(
        after=after,
        fail_next_create=fail_next,
        hide_next_booking=hide_next,
        rewrite_offer=rewrite,
    )


def _parse_clock(value: str, source: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CaseLoadError(source, f"{field} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaseLoadError(source, f"{field} must be timezone-aware")
    return parsed


def check_model_order(
    chunks: Sequence[OutputChunk],
    *,
    allowed_amounts: set[Decimal],
    allowed_numbers: set[Decimal],
    pack: OutputPack,
    extract: Callable[[str], Sequence[Any]],
) -> str | None:
    """Invariant 4. Mutates the copied sets. Returns a check token or None."""

    class _Allowed:
        def __init__(self) -> None:
            self.allowed_amounts = set(allowed_amounts)
            self.allowed_numbers = set(allowed_numbers)

    holder = _Allowed()
    state = cast(CallState, holder)
    for chunk in chunks:
        if chunk.origin == "backend":
            admit_numbers(state, chunk.text, extract)
            continue
        if chunk.origin != "model":
            continue
        checked = check_sentence(
            chunk.text,
            pack,
            allowed_amounts=holder.allowed_amounts,
            allowed_numbers=holder.allowed_numbers,
            extract=extract,
        )
        if isinstance(checked, Blocked):
            return "model_sentence"
    return None


@cache
def _clinical_trip_names() -> frozenset[str]:
    names: set[str] = set()
    for filename in ("universal.yaml", "medspa.yaml"):
        pack = load_input_pack(PACKS_DIR / filename)
        for trip in pack.tripwires:
            if trip.clinical_class:
                names.add(trip.name)
    return frozenset(names)


async def run_case(
    case: EvalCase,
    *,
    loop_cls: type[TurnLoop] = TurnLoop,
) -> CaseFailure | None:
    tenant = _tenant(case.tenant)
    pack = load_output_pack(PACKS_DIR / "medspa.output.yaml")
    engine = load_policy_engine(tenant)
    words = extra_price_words(tenant)
    extract = partial(
        find_numbers,
        extra_price_words=words.extra_price_words,
        service_words=words.service_words,
    )
    keys = InMemoryKeyStore()
    data_key_id = await keys.create(f"eval_{case.id}")
    log = InMemoryEventLog(keys)
    store = InMemorySessionStore()
    state = new_call_state(
        tenant=tenant,
        call_id=f"eval_{case.id}",
        data_key_id=data_key_id,
        now=case.started,
        extract=extract,
    )
    state.caller_number = case.caller_id
    await store.create(state)

    def clock() -> datetime:
        return case.clock

    adapter = MockBookingAdapter.from_yaml(
        ROOT / "tenants" / "glow_aesthetics.mock.yaml", tz=tenant.tz, clock=clock
    )
    model = _script(case)
    loop = loop_cls(
        tenant,
        store,
        log,
        _OfferRewrite(model, state, tenant.tz),
        adapter,
        SECRET,
        clock=clock,
        engine=engine,
        pack=pack,
        extract=extract,
    )
    spoken: list[OutputChunk] = []
    async for chunk in loop.connect(state):
        spoken.append(chunk)
    for index, utterance in enumerate(case.utterances, start=1):
        start_amounts = set(state.allowed_amounts)
        start_numbers = set(state.allowed_numbers)
        before_calls = model.call_count
        was_committed = isinstance(state.escalation, Committed)
        turn_chunks: list[OutputChunk] = []
        async for chunk in loop.handle_caller_turn(state, utterance):
            turn_chunks.append(chunk)
            spoken.append(chunk)
        if not was_committed and isinstance(state.escalation, Committed):
            if model.call_count != before_calls:
                return _fail(case, "model_after_commit")
        order = check_model_order(
            turn_chunks,
            allowed_amounts=start_amounts,
            allowed_numbers=start_numbers,
            pack=pack,
            extract=extract,
        )
        if order is not None:
            return _fail(case, order)
        if case.hooks.after == index:
            _apply_hooks(case.hooks, state, adapter)
    return await _check_expect(case, state, spoken, log, model, adapter, tenant)


def _script(case: EvalCase) -> ScriptedLLM:
    if case.script_path is not None:
        return ScriptedLLM.load(EVALS_DIR / case.script_path)
    assert case.script is not None
    return ScriptedLLM.from_mapping(case.script, source=case.source)


def _tenant(name: str) -> Any:
    today = datetime(2026, 10, 29, tzinfo=UTC).date()
    if name == "glow":
        return load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=today)
    return load_tenant_file(
        ROOT / "tests" / "core" / "fixtures" / "tenants" / "test_spa.yaml", today=today
    )


def _apply_hooks(hooks: Hooks, state: CallState, adapter: MockBookingAdapter) -> None:
    if not state.offered_slots:
        return
    first = state.offered_slots[0]
    if hooks.fail_next_create:
        adapter.fail_next_create(
            Slot(
                start=first.start,
                end=first.end,
                provider_id=first.provider_id,
                room_id=first.room_id,
                version=first.version,
            )
        )
    if hooks.hide_next_booking:
        adapter.hide_on_next_get(f"bk_{adapter._next_id:05d}")
    if hooks.rewrite_offer == "unavailable":
        state.offered_slots[0] = type(first)(
            service_id=first.service_id,
            start=first.start,
            end=first.end,
            provider_id="no_such_provider",
            room_id="no_such_room",
            version=0,
        )
    if hooks.rewrite_offer == "dst":
        from datetime import timedelta

        ny = ZoneInfo("America/New_York")
        fold0 = datetime(2026, 11, 1, 1, 0, tzinfo=ny)
        fold1 = fold0.replace(fold=1)
        slot = type(first)
        state.offered_slots[:] = [
            slot(
                service_id=first.service_id,
                start=fold0.astimezone(UTC),
                end=fold0.astimezone(UTC) + timedelta(minutes=30),
                provider_id="np_1",
                room_id="inject_1",
                version=0,
            ),
            slot(
                service_id=first.service_id,
                start=fold1.astimezone(UTC),
                end=fold1.astimezone(UTC) + timedelta(minutes=30),
                provider_id="rn_1",
                room_id="inject_2",
                version=0,
            ),
        ]


async def _check_expect(
    case: EvalCase,
    state: CallState,
    spoken: Sequence[OutputChunk],
    log: InMemoryEventLog,
    model: ScriptedLLM,
    adapter: MockBookingAdapter,
    tenant: Any,
) -> CaseFailure | None:
    texts = [chunk.text for chunk in spoken]
    if not _spoken_subsequence(texts, case.spoken):
        return _fail(case, "spoken_missing")
    if any(forbidden in text for forbidden in case.never_spoken for text in texts):
        return _fail(case, "sentence_spoken")
    if case.committed is None:
        if isinstance(state.escalation, Committed):
            return _fail(case, "unexpected_commit")
    else:
        if not isinstance(state.escalation, Committed):
            return _fail(case, "committed")
        if state.escalation.trip != case.committed.trip:
            return _fail(case, "committed_trip")
        if _path_label(state.escalation.path) != case.committed.path:
            return _fail(case, "committed_path")
    events = await log.read(state.call_id)
    kinds = [event.kind for event in events]
    for kind in case.events.present:
        if kind not in kinds:
            return _fail(case, "event_present")
    for kind in case.events.absent:
        if kind in kinds:
            return _fail(case, "event_absent")
    if case.model_called is False and model.call_count != 0:
        return _fail(case, "model_called")
    committed = isinstance(state.escalation, Committed)
    sms = kinds.count("sms_logged")
    if committed and sms != 1:
        return _fail(case, "sms_count")
    if not committed and sms != 0:
        return _fail(case, "sms_count")
    if _clinical_leak(case, events, state):
        return _fail(case, "clinical_leak")
    if _bad_transfer(state, spoken, tenant, case.clock):
        return _fail(case, "transfer_target")
    if await _booking_mismatch(events, adapter):
        return _fail(case, "booking_confirmed")
    if any(_TEXT_PROMISE.search(text) for text in texts):
        return _fail(case, "text_promise")
    return None


def _spoken_subsequence(texts: Sequence[str], expect: Sequence[SpokenExpect]) -> bool:
    index = 0
    for rule in expect:
        while index < len(texts):
            chunk = texts[index]
            index += 1
            if rule.kind == "exact" and chunk == rule.text:
                break
            if rule.kind == "contains" and rule.text in chunk:
                break
        else:
            return False
    return True


def _clinical_leak(case: EvalCase, events: Sequence[Event], state: CallState) -> bool:
    if not isinstance(state.escalation, Committed):
        return False
    if state.escalation.trip not in _clinical_trip_names():
        return False
    rendered = [str(value) for event in events for value in event.payload.values()]
    blob = "\n".join(rendered)
    for utterance in case.utterances:
        if utterance in blob:
            return True
        for line in state.transcript:
            if isinstance(line, CallerLine) and utterance in line.text:
                return True
    return False


def _bad_transfer(
    state: CallState, spoken: Sequence[OutputChunk], tenant: Any, now: datetime
) -> bool:
    numbers = [chunk.transfer_number for chunk in spoken if chunk.transfer_number]
    if not numbers:
        return False
    if not isinstance(state.escalation, Committed) or not isinstance(
        state.escalation.path, Transfer
    ):
        return True
    target = state.pending_transfer_number
    if target is None or target == tenant.main_number:
        return True
    if any(number != target for number in numbers):
        return True
    tier = next(
        (row for row in tenant.transfer_tiers if row.tier == state.escalation.path.tier),
        None,
    )
    if tier is None or tier.live_hours is None:
        return True
    return not tier.live_hours.is_open(now, tenant.tz)


async def _booking_mismatch(events: Sequence[Event], adapter: MockBookingAdapter) -> bool:
    confirmed = [event for event in events if event.kind == "booking_confirmed"]
    if not confirmed:
        return False
    booking_id = confirmed[0].payload.get("booking_id")
    if not isinstance(booking_id, str) or not booking_id:
        return True
    try:
        await adapter.get_booking(booking_id)
    except Exception:
        return True
    return False


def _path_label(path: object) -> str:
    from core.state import Callback, EmergencyEnd

    if isinstance(path, EmergencyEnd):
        return "emergency_end"
    if isinstance(path, Transfer):
        return "transfer"
    if isinstance(path, Callback):
        return "callback"
    return "unknown"


def _fail(case: EvalCase, check: str) -> CaseFailure:
    return CaseFailure(category=case.category, id=case.id, check=check)


class _OfferRewrite:
    def __init__(self, inner: ScriptedLLM, state: CallState, tz: ZoneInfo) -> None:
        self.inner = inner
        self.state = state
        self.tz = tz

    @property
    def call_count(self) -> int:
        return self.inner.call_count

    def stream(self, messages: Sequence[Any], tools: Sequence[Any], **kwargs: Any) -> Any:
        return _RewriteStream(self.inner.stream(messages, tools, **kwargs), self.state, self.tz)


class _RewriteStream:
    def __init__(self, inner: Any, state: CallState, tz: ZoneInfo) -> None:
        self.inner = inner
        self.state = state
        self.tz = tz

    def __aiter__(self) -> _RewriteStream:
        return self

    async def __anext__(self) -> Any:
        from core.llm import ToolCall

        event = await self.inner.__anext__()
        if not isinstance(event, ToolCall):
            return event
        slot = event.args.get("slot_start_iso")
        if slot != "$first_offered" or not self.state.offered_slots:
            return event
        start = self.state.offered_slots[0].start
        rewritten = dict(event.args)
        local = start.astimezone(self.tz).replace(microsecond=0)
        rewritten["slot_start_iso"] = local.strftime("%Y-%m-%dT%H:%M:%S")
        return ToolCall(event.id, event.name, rewritten)

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()


def format_report(report: Report) -> str:
    categories: dict[str, list[CaseFailure | None]] = {}
    for case, failure in report.results:
        categories.setdefault(case.category, []).append(failure)
    failed = [failure for _case, failure in report.results if failure is not None]
    passed = len(report.results) - len(failed)
    lines = [f"eval: {passed} passed, {len(failed)} failed"]
    width = max((len(name) for name in categories), default=8)
    for name in sorted(categories):
        rows = categories[name]
        ok = sum(1 for item in rows if item is None)
        lines.append(f"  {name.ljust(width)}  {ok}/{len(rows)}")
    for failure in failed:
        lines.append(f"FAIL {failure.category}/{failure.id}  check={failure.check}")
    if failed:
        lines.append(f"eval: {passed} passed, {len(failed)} failed")
    return "\n".join(lines) + "\n"


async def run_all(cases: Sequence[EvalCase]) -> Report:
    report = Report()
    for case in cases:
        report.add(case, await run_case(case))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    _ = argv
    cases = load_cases()
    report = asyncio.run(run_all(cases))
    sys.stdout.write(format_report(report))
    return 1 if any(failure is not None for _case, failure in report.results) else 0
