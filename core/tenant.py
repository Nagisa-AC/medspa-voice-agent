"""Tenant config: models, validation rules, YAML loading and the registry."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from zoneinfo import ZoneInfo, available_timezones

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainValidator,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- errors


@dataclass(frozen=True)
class Issue:
    path: str
    message: str


@dataclass(frozen=True)
class FileIssues:
    source: str
    issues: tuple[Issue, ...]


class TenantRulesError(Exception):
    """Raised by the Tenant model validator.

    Deliberately not a ValueError, so Pydantic lets it propagate unwrapped and every
    issue keeps its own field path.
    """

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(f"{i.path}: {i.message}" for i in self.issues))


class TenantConfigError(Exception):
    def __init__(self, files: Iterable[FileIssues]) -> None:
        self.files = tuple(files)
        super().__init__(self._render())

    def _render(self) -> str:
        noun = "file is" if len(self.files) == 1 else "files are"
        lines = [f"{len(self.files)} tenant {noun} invalid"]
        for f in self.files:
            lines.append("")
            lines.append(f.source)
            lines.extend(f"  {i.path or '<file>'}: {i.message}" for i in f.issues)
        return "\n".join(lines)


# --------------------------------------------------------------------------- field types

_E164 = re.compile(r"\+[1-9]\d{1,14}")
_SLUG = re.compile(r"[a-z][a-z0-9_]*")


def _e164(value: str) -> str:
    if not _E164.fullmatch(value):
        raise ValueError("must be an E.164 phone number like +12035550100")
    return value


def _slug(value: str) -> str:
    if not _SLUG.fullmatch(value):
        raise ValueError(
            "must be lowercase letters, digits and underscores, starting with a letter"
        )
    return value


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
E164 = Annotated[str, AfterValidator(_e164)]
Slug = Annotated[str, AfterValidator(_slug)]
Price = Annotated[Decimal, Field(gt=0, allow_inf_nan=False)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
StrictBool = Annotated[bool, Field(strict=True)]

# --------------------------------------------------------------------------- live hours

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_CLAUSE = re.compile(
    r"(?P<d1>[A-Za-z]+)(?:\s*-\s*(?P<d2>[A-Za-z]+))?\s+(?P<start>\S+?)\s*-\s*(?P<end>\S+)"
)
_TIME = re.compile(r"(\d{2}):(\d{2})")
_FORMAT_HINT = 'expected e.g. "Mon-Fri 09:00-18:00, Sat 10:00-16:00"'


@dataclass(frozen=True)
class Window:
    weekday: int  # Monday == 0, as datetime.weekday()
    start_min: int
    end_min: int  # exclusive; 1440 means 24:00


@dataclass(frozen=True)
class LiveHours:
    text: str
    windows: tuple[Window, ...]

    @classmethod
    def parse(cls, text: str) -> LiveHours:
        clauses = [c.strip() for c in text.split(",")]
        if not text.strip():
            raise ValueError(f"is empty; {_FORMAT_HINT}")
        windows: list[tuple[Window, str]] = []
        for clause in clauses:
            m = _CLAUSE.fullmatch(clause)
            if m is None:
                raise ValueError(f"cannot parse {clause!r}; {_FORMAT_HINT}")
            first = _day(m["d1"], clause)
            last = _day(m["d2"], clause) if m["d2"] else first
            start = _minutes(m["start"], clause, allow_24=False)
            end = _minutes(m["end"], clause, allow_24=True)
            if end == start:
                raise ValueError(f"{clause!r} is an empty window")
            if end < start:
                raise ValueError(f"{clause!r} crosses midnight; split it at 24:00 into two clauses")
            day = first
            while True:
                windows.append((Window(day, start, end), clause))
                if day == last:
                    break
                day = (day + 1) % 7
        ordered = sorted(windows, key=lambda w: (w[0].weekday, w[0].start_min))
        for (a, a_src), (b, b_src) in pairwise(ordered):
            if a.weekday == b.weekday and b.start_min < a.end_min:
                raise ValueError(f"{a_src!r} overlaps {b_src!r} on {_DAYS[a.weekday].title()}")
        return cls(text=text, windows=tuple(w for w, _ in ordered))

    def is_open(self, now: datetime, tz: ZoneInfo) -> bool:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")
        local = now.astimezone(tz)
        minute = local.hour * 60 + local.minute
        return any(
            w.weekday == local.weekday() and w.start_min <= minute < w.end_min for w in self.windows
        )


def _day(name: str, clause: str) -> int:
    key = name.lower()
    if key not in _DAYS:
        raise ValueError(f"unknown day {name!r} in {clause!r}; use Mon Tue Wed Thu Fri Sat Sun")
    return _DAYS.index(key)


def _minutes(text: str, clause: str, *, allow_24: bool) -> int:
    m = _TIME.fullmatch(text)
    if m is None:
        raise ValueError(f"time {text!r} in {clause!r} must be 24-hour HH:MM")
    hours, minutes = int(m[1]), int(m[2])
    if allow_24 and hours == 24 and minutes == 0:
        return 1440
    if hours > 23 or minutes > 59:
        raise ValueError(f"time {text!r} in {clause!r} is out of range")
    return hours * 60 + minutes


def _live_hours(value: object) -> LiveHours:
    if not isinstance(value, str):
        raise ValueError("must be a string")
    return LiveHours.parse(value)


# --------------------------------------------------------------------------- models


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Service(_Model):
    id: Slug
    name: Text
    aliases: tuple[Text, ...] = ()
    quotable: StrictBool = True
    price_min: Price | None = None
    price_max: Price | None = None
    price_unit: Text | None = None
    duration_min: PositiveInt | None = None
    room_type: Text | None = None
    provider_types: tuple[Text, ...] = ()
    notes: Text | None = None
    deposit_required: StrictBool = False
    deposit_amount: Price | None = None
    consult_required: StrictBool = False


class TransferTier(_Model):
    tier: PositiveInt
    kind: Literal["transfer", "sms_summary"]
    target: E164
    label: Slug
    answers_live: StrictBool = False
    live_hours: Annotated[LiveHours, PlainValidator(_live_hours)] | None = None


class FaqEntry(_Model):
    id: Slug
    topics: Annotated[tuple[Text, ...], Field(min_length=1)]
    say: Text
    approved_by: Text
    approved_on: Annotated[date, Field(strict=True)]


class Integration(_Model):
    provider: Text
    location_id: Text


class Tenant(_Model):
    id: Slug
    name: Text
    vertical: Text
    timezone: str
    main_number: E164
    greeting: Text
    record_disclosure: Text
    ai_identity_reply: Text
    reminder_line: Text
    max_turns: PositiveInt
    max_seconds: PositiveInt
    retention_days: PositiveInt
    services: Annotated[tuple[Service, ...], Field(min_length=1)]
    facts: dict[Slug, Text] = Field(default_factory=dict)
    policies: dict[Slug, Text] = Field(default_factory=dict)
    faq: tuple[FaqEntry, ...] = ()
    transfer_tiers: tuple[TransferTier, ...]
    callback_promise: Text
    integration: Integration

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        if "/" not in value or value.startswith("Etc/") or value not in _zones():
            raise ValueError("must be a region-style IANA timezone like America/New_York")
        return value

    @model_validator(mode="after")
    def _check_rules(self, info: ValidationInfo) -> Self:
        context = info.context
        today = context.get("today") if isinstance(context, dict) else None
        if not isinstance(today, date) or isinstance(today, datetime):
            raise TypeError("validate tenants with context={'today': date}; use parse_tenant()")
        issues = check_rules(self, today)
        if issues:
            raise TenantRulesError(issues)
        return self


@cache
def _zones() -> frozenset[str]:
    return frozenset(available_timezones())


def normalize_phrase(text: str) -> str:
    """Casefold, turn punctuation into spaces, collapse whitespace."""
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold()).split())


# --------------------------------------------------------------------------- rules


def check_rules(tenant: Tenant, today: date) -> list[Issue]:
    issues: list[Issue] = []
    issues += _service_rules(tenant.services)
    issues += _tier_rules(tenant.transfer_tiers, tenant.main_number)
    issues += _faq_rules(tenant.faq, today)
    return issues


def _service_rules(services: tuple[Service, ...]) -> list[Issue]:
    issues: list[Issue] = []
    seen_ids: dict[str, int] = {}
    owners: dict[str, int] = {}
    for i, s in enumerate(services):
        if s.id in seen_ids:
            issues.append(Issue(f"services[{i}].id", f"duplicates services[{seen_ids[s.id]}].id"))
        seen_ids.setdefault(s.id, i)

    # A service's name and its aliases all resolve to it; a phrase may resolve to one service.
    for i, s in enumerate(services):
        own_aliases: set[str] = set()
        phrases = [(f"services[{i}].name", s.name)]
        phrases += [(f"services[{i}].aliases[{j}]", a) for j, a in enumerate(s.aliases)]
        for path, phrase in phrases:
            key = normalize_phrase(phrase)
            if not key:
                issues.append(Issue(path, "has no letters or digits"))
                continue
            if path.endswith("]"):
                if key in own_aliases:
                    issues.append(Issue(path, f"duplicate alias {phrase!r} in this service"))
                own_aliases.add(key)
            owner = owners.setdefault(key, i)
            if owner != i:
                issues.append(
                    Issue(path, f"{phrase!r} also belongs to service {services[owner].id!r}")
                )

    for i, s in enumerate(services):
        p = f"services[{i}]"
        prices = {"price_min": s.price_min, "price_max": s.price_max, "price_unit": s.price_unit}
        if s.quotable:
            for field, value in prices.items():
                if value is None:
                    issues.append(Issue(f"{p}.{field}", "required when quotable"))
            if s.price_min is not None and s.price_max is not None and s.price_min > s.price_max:
                issues.append(
                    Issue(f"{p}.price_max", f"must be >= price_min ({s.price_min} > {s.price_max})")
                )
            if not s.consult_required:
                if s.duration_min is None:
                    issues.append(Issue(f"{p}.duration_min", "required for a bookable service"))
                if s.room_type is None:
                    issues.append(Issue(f"{p}.room_type", "required for a bookable service"))
                if not s.provider_types:
                    issues.append(Issue(f"{p}.provider_types", "required for a bookable service"))
        else:
            for field, value in {**prices, "deposit_amount": s.deposit_amount}.items():
                if value is not None:
                    issues.append(Issue(f"{p}.{field}", "not allowed when quotable is false"))
        if s.deposit_required and s.deposit_amount is None:
            issues.append(Issue(f"{p}.deposit_amount", "required when deposit_required is true"))
    return issues


def _tier_rules(tiers: tuple[TransferTier, ...], main_number: str) -> list[Issue]:
    issues: list[Issue] = []
    seen: dict[int, int] = {}
    sms: list[int] = []
    for i, t in enumerate(tiers):
        p = f"transfer_tiers[{i}]"
        if t.tier in seen:
            issues.append(Issue(f"{p}.tier", f"duplicates transfer_tiers[{seen[t.tier]}].tier"))
        seen.setdefault(t.tier, i)
        if t.target == main_number:
            issues.append(Issue(f"{p}.target", "equals main_number; it is never a tier target"))
        if t.kind == "sms_summary":
            sms.append(i)
            if t.answers_live:
                issues.append(Issue(f"{p}.answers_live", "not allowed on an sms_summary tier"))
        if t.answers_live and t.live_hours is None:
            issues.append(Issue(f"{p}.live_hours", "required when answers_live is true"))
        if not t.answers_live and t.live_hours is not None:
            issues.append(Issue(f"{p}.live_hours", "only allowed when answers_live is true"))
    if not sms:
        issues.append(Issue("transfer_tiers", "needs exactly one sms_summary tier; found none"))
    for i in sms[1:]:
        issues.append(Issue(f"transfer_tiers[{i}].kind", "only one sms_summary tier is allowed"))
    return issues


def _faq_rules(faq: tuple[FaqEntry, ...], today: date) -> list[Issue]:
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    for i, entry in enumerate(faq):
        if entry.id in seen:
            issues.append(Issue(f"faq[{i}].id", f"duplicates faq[{seen[entry.id]}].id"))
        seen.setdefault(entry.id, i)
        if entry.approved_on > today:
            issues.append(Issue(f"faq[{i}].approved_on", f"is in the future (today is {today})"))
    return issues


# --------------------------------------------------------------------------- YAML


class _Loader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys and reads floats as exact Decimals."""


def _construct_mapping(loader: _Loader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


def _construct_decimal(loader: _Loader, node: yaml.ScalarNode) -> Decimal:
    text = str(loader.construct_scalar(node)).replace("_", "")
    try:
        value = Decimal(text)
    except InvalidOperation:
        value = Decimal("NaN")
    if not value.is_finite():
        raise yaml.constructor.ConstructorError(
            None, None, f"number {text!r} is not a finite decimal", node.start_mark
        )
    return value


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
_Loader.add_constructor("tag:yaml.org,2002:float", _construct_decimal)


def _yaml_hint(err: Mapping[str, Any]) -> str:
    value = err.get("input")
    if err["type"] == "string_type" and isinstance(value, bool | int | Decimal):
        return f" (got {type(value).__name__} {value!r}; quote it in YAML)"
    return ""


def _path(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _pydantic_issues(exc: ValidationError) -> list[Issue]:
    issues = []
    for err in exc.errors(include_url=False):
        msg = str(err["msg"]).removeprefix("Value error, ")
        issues.append(Issue(_path(tuple(err["loc"])), msg + _yaml_hint(err)))
    return issues


def parse_tenant(data: object, *, source: str, today: date) -> Tenant:
    """Validate one tenant's parsed YAML. Raises TenantConfigError naming source and fields."""
    try:
        return Tenant.model_validate(data, context={"today": today})
    except ValidationError as exc:
        raise TenantConfigError([FileIssues(source, tuple(_pydantic_issues(exc)))]) from None
    except TenantRulesError as exc:
        raise TenantConfigError([FileIssues(source, exc.issues)]) from None


def read_yaml(path: Path) -> object:
    """Parse a config file with the strict loader. Raises TenantConfigError on YAML errors."""
    try:
        return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}, column {mark.column + 1}: " if mark else ""
        problem = getattr(exc, "problem", None) or str(exc)
        raise TenantConfigError([FileIssues(str(path), (Issue("", where + problem),))]) from None


def load_tenant_file(path: Path, *, today: date) -> Tenant:
    tenant = parse_tenant(read_yaml(path), source=str(path), today=today)
    if tenant.id != path.stem:
        issue = Issue("id", f"must equal the file name {path.stem!r} (got {tenant.id!r})")
        raise TenantConfigError([FileIssues(str(path), (issue,))])
    return tenant


# --------------------------------------------------------------------------- registry


def _is_tenant_file(path: Path) -> bool:
    return path.suffix == ".yaml" and not path.name.endswith(".mock.yaml")


class TenantRegistry:
    def __init__(self, tenants: Iterable[Tenant]) -> None:
        self._tenants = {t.id: t for t in tenants}

    @classmethod
    def load(cls, directory: Path, *, today: date) -> TenantRegistry:
        """Load every tenant file in a directory; report every broken file at once."""
        tenants: list[Tenant] = []
        broken: list[FileIssues] = []
        for path in sorted(p for p in directory.iterdir() if p.is_file() and _is_tenant_file(p)):
            try:
                tenants.append(load_tenant_file(path, today=today))
            except TenantConfigError as exc:
                broken.extend(exc.files)
        if broken:
            raise TenantConfigError(broken)
        return cls(tenants)

    def get(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tenants))
