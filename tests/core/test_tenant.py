import copy
import shutil
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from core.tenant import (
    Issue,
    Tenant,
    TenantConfigError,
    TenantRegistry,
    TenantRulesError,
    load_tenant_file,
    normalize_phrase,
    parse_tenant,
    read_yaml,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent / "fixtures" / "tenants" / "test_spa.yaml"
TODAY = date(2026, 10, 29)

Data = dict[str, Any]


def base() -> Data:
    data = read_yaml(BASE)
    assert isinstance(data, dict)
    return copy.deepcopy(data)


def issues_for(data: Data, source: str = "test_spa.yaml") -> list[Issue]:
    with pytest.raises(TenantConfigError) as info:
        parse_tenant(data, source=source, today=TODAY)
    [file] = info.value.files
    assert file.source == source
    return list(file.issues)


def svc(i: int) -> Callable[[Data], Data]:
    return lambda d: d["services"][i]


def tier(i: int) -> Callable[[Data], Data]:
    return lambda d: d["transfer_tiers"][i]


def faq(i: int) -> Callable[[Data], Data]:
    return lambda d: d["faq"][i]


def top(d: Data) -> Data:
    return d


def set_(where: Callable[[Data], Data], key: str, value: object) -> Callable[[Data], None]:
    return lambda d: where(d).__setitem__(key, value)


def drop(where: Callable[[Data], Data], key: str) -> Callable[[Data], None]:
    return lambda d: where(d).pop(key)


def add_faq_copy(d: Data) -> None:
    d["faq"].append(dict(d["faq"][0]))


def add_service(d: Data) -> None:
    extra = copy.deepcopy(d["services"][0])
    extra["name"] = "Delta"
    extra["aliases"] = ["delta"]
    d["services"].append(extra)


def two(*patches: Callable[[Data], None]) -> Callable[[Data], None]:
    def apply(d: Data) -> None:
        for p in patches:
            p(d)

    return apply


# (case id, patch applied to the valid base, field path that must be reported)
BROKEN: list[tuple[str, Callable[[Data], None], str]] = [
    # Service ids unique
    ("service_id_duplicate", add_service, "services[3].id"),
    # An alias belongs to one service
    ("alias_in_two_services", set_(svc(1), "aliases", ["alpha"]), "services[1].aliases[0]"),
    ("alias_differs_only_in_case", set_(svc(1), "aliases", ["ALPHA!"]), "services[1].aliases[0]"),
    ("alias_is_other_name", set_(svc(1), "aliases", ["alpha treatment"]), "services[1].aliases[0]"),
    ("alias_duplicate_in_service", set_(svc(0), "aliases", ["x", "X"]), "services[0].aliases[1]"),
    ("alias_blank", set_(svc(0), "aliases", ["  "]), "services[0].aliases[0]"),
    ("alias_punctuation_only", set_(svc(0), "aliases", ["--"]), "services[0].aliases[0]"),
    # Quotable services need price_min, price_max (Decimal), price_unit, min <= max
    ("quotable_no_price_min", drop(svc(0), "price_min"), "services[0].price_min"),
    ("quotable_no_price_max", drop(svc(0), "price_max"), "services[0].price_max"),
    ("quotable_no_price_unit", drop(svc(0), "price_unit"), "services[0].price_unit"),
    ("price_min_above_max", set_(svc(0), "price_min", 30), "services[0].price_max"),
    ("price_not_decimal", set_(svc(0), "price_min", "ten"), "services[0].price_min"),
    ("price_zero", set_(svc(0), "price_min", 0), "services[0].price_min"),
    ("price_bool", set_(svc(0), "price_min", True), "services[0].price_min"),
    # quotable: false carries no prices
    ("unquotable_price_min", set_(svc(2), "price_min", 5), "services[2].price_min"),
    ("unquotable_price_max", set_(svc(2), "price_max", 5), "services[2].price_max"),
    ("unquotable_price_unit", set_(svc(2), "price_unit", "per x"), "services[2].price_unit"),
    ("unquotable_deposit", set_(svc(2), "deposit_amount", 5), "services[2].deposit_amount"),
    # Bookable quotable services carry booking constraints; deposits need an amount
    ("bookable_no_duration", drop(svc(0), "duration_min"), "services[0].duration_min"),
    ("bookable_no_room", drop(svc(0), "room_type"), "services[0].room_type"),
    ("bookable_no_providers", set_(svc(0), "provider_types", []), "services[0].provider_types"),
    ("deposit_no_amount", drop(svc(1), "deposit_amount"), "services[1].deposit_amount"),
    ("deposit_not_required", drop(svc(1), "deposit_required"), "services[1].deposit_amount"),
    (
        "deposit_required_false",
        set_(svc(1), "deposit_required", False),
        "services[1].deposit_amount",
    ),
    # Valid IANA timezone (region-style)
    ("timezone_unknown", set_(top, "timezone", "Mars/Base"), "timezone"),
    ("timezone_abbreviation", set_(top, "timezone", "EST"), "timezone"),
    ("timezone_utc", set_(top, "timezone", "UTC"), "timezone"),
    ("timezone_etc", set_(top, "timezone", "Etc/GMT+5"), "timezone"),
    # main_number E.164
    ("main_number_not_e164", set_(top, "main_number", "203-555-0100"), "main_number"),
    ("main_number_no_plus", set_(top, "main_number", "15555550100"), "main_number"),
    ("tier_target_not_e164", set_(tier(0), "target", "+0123"), "transfer_tiers[0].target"),
    # No tier targets main_number
    ("tier_targets_main", set_(tier(1), "target", "+15555550100"), "transfer_tiers[1].target"),
    ("sms_targets_main", set_(tier(2), "target", "+15555550100"), "transfer_tiers[2].target"),
    # answers_live: true requires parseable live_hours
    ("live_without_hours", set_(tier(1), "answers_live", True), "transfer_tiers[1].live_hours"),
    ("hours_unparseable", set_(tier(0), "live_hours", "9-5"), "transfer_tiers[0].live_hours"),
    (
        "hours_overnight",
        set_(tier(0), "live_hours", "Fri 22:00-02:00"),
        "transfer_tiers[0].live_hours",
    ),
    ("hours_not_string", set_(tier(0), "live_hours", 9), "transfer_tiers[0].live_hours"),
    (
        "hours_without_live",
        set_(tier(1), "live_hours", "Mon 09:00-10:00"),
        "transfer_tiers[1].live_hours",
    ),
    ("sms_answers_live", set_(tier(2), "answers_live", True), "transfer_tiers[2].answers_live"),
    ("answers_live_string", set_(tier(0), "answers_live", "yes"), "transfer_tiers[0].answers_live"),
    # Exactly one sms_summary tier; unique tier numbers
    ("no_sms_tier", lambda d: d["transfer_tiers"].pop(), "transfer_tiers"),
    ("two_sms_tiers", set_(tier(1), "kind", "sms_summary"), "transfer_tiers[2].kind"),
    ("tier_number_duplicate", set_(tier(1), "tier", 1), "transfer_tiers[1].tier"),
    ("tier_kind_unknown", set_(tier(1), "kind", "voicemail"), "transfer_tiers[1].kind"),
    # Every faq entry has say, non-empty topics, approved_by, approved_on
    ("faq_no_say", drop(faq(0), "say"), "faq[0].say"),
    ("faq_blank_say", set_(faq(0), "say", " "), "faq[0].say"),
    ("faq_no_topics", drop(faq(0), "topics"), "faq[0].topics"),
    ("faq_empty_topics", set_(faq(0), "topics", []), "faq[0].topics"),
    ("faq_no_approved_by", drop(faq(0), "approved_by"), "faq[0].approved_by"),
    ("faq_no_approved_on", drop(faq(0), "approved_on"), "faq[0].approved_on"),
    ("faq_approved_on_text", set_(faq(0), "approved_on", "Jan 15"), "faq[0].approved_on"),
    (
        "faq_approved_in_future",
        set_(faq(0), "approved_on", date(2026, 10, 30)),
        "faq[0].approved_on",
    ),
    ("faq_id_duplicate", add_faq_copy, "faq[1].id"),
    # callback_promise and ai_identity_reply present
    ("no_callback_promise", drop(top, "callback_promise"), "callback_promise"),
    ("blank_callback_promise", set_(top, "callback_promise", "   "), "callback_promise"),
    ("no_ai_identity_reply", drop(top, "ai_identity_reply"), "ai_identity_reply"),
    ("blank_ai_identity_reply", set_(top, "ai_identity_reply", ""), "ai_identity_reply"),
    # Shape: unknown keys, slugs, positive ints
    ("unknown_key", set_(tier(0), "answer_live", True), "transfer_tiers[0].answer_live"),
    ("tenant_id_not_slug", set_(top, "id", "Test Spa"), "id"),
    ("max_turns_zero", set_(top, "max_turns", 0), "max_turns"),
    ("max_turns_text", set_(top, "max_turns", "10"), "max_turns"),
    ("no_services", set_(top, "services", []), "services"),
]


@pytest.mark.parametrize(("case", "patch", "path"), BROKEN, ids=[c[0] for c in BROKEN])
def test_broken_config_names_file_and_field(
    case: str, patch: Callable[[Data], None], path: str
) -> None:
    data = base()
    patch(data)
    paths = [i.path for i in issues_for(data)]
    assert path in paths, paths


def test_base_fixture_is_valid() -> None:
    tenant = load_tenant_file(BASE, today=TODAY)
    assert tenant.id == "test_spa"


def test_all_rule_violations_reported_together() -> None:
    data = base()
    data["services"][1]["aliases"] = ["alpha"]
    data["transfer_tiers"][1]["target"] = data["main_number"]
    data["faq"][0]["approved_on"] = date(2027, 1, 1)
    paths = {i.path for i in issues_for(data)}
    assert paths == {"services[1].aliases[0]", "transfer_tiers[1].target", "faq[0].approved_on"}


def test_error_message_lists_file_and_paths() -> None:
    data = base()
    data["services"][0]["price_min"] = 30
    with pytest.raises(TenantConfigError) as info:
        parse_tenant(data, source="tenants/x.yaml", today=TODAY)
    text = str(info.value)
    assert "tenants/x.yaml" in text
    assert "services[0].price_max: must be >= price_min (30 > 20.50)" in text


def test_approved_on_today_is_allowed() -> None:
    data = base()
    data["faq"][0]["approved_on"] = TODAY
    parse_tenant(data, source="x", today=TODAY)


def test_alias_equal_to_own_name_is_allowed() -> None:
    data = base()
    data["services"][0]["aliases"] = ["Alpha Treatment"]
    parse_tenant(data, source="x", today=TODAY)


def test_rules_error_passes_through_pydantic_unwrapped() -> None:
    # The per-field paths depend on Pydantic letting a non-ValueError escape the model
    # validator. If an upgrade starts wrapping it, this test fails.
    data = base()
    data["transfer_tiers"][1]["target"] = data["main_number"]
    with pytest.raises(TenantRulesError) as info:
        Tenant.model_validate(data, context={"today": TODAY})
    assert not isinstance(info.value, ValidationError)
    assert info.value.issues == (
        Issue("transfer_tiers[1].target", "equals main_number; it is never a tier target"),
    )


def test_validating_without_today_is_refused() -> None:
    with pytest.raises(TypeError, match="today"):
        Tenant.model_validate(base())


def test_models_are_frozen() -> None:
    tenant = parse_tenant(base(), source="x", today=TODAY)
    with pytest.raises(ValidationError):
        tenant.max_turns = 5  # type: ignore[misc]


@pytest.mark.parametrize("field", ["facts", "policies"])
def test_text_maps_are_read_only(field: str) -> None:
    tenant = parse_tenant(base(), source="x", today=TODAY)
    mapping: Any = getattr(tenant, field)
    with pytest.raises(TypeError):
        mapping["hours"] = "changed"
    with pytest.raises(TypeError):
        del mapping[next(iter(mapping))]


def test_missing_text_maps_default_to_read_only() -> None:
    data = base()
    del data["facts"]
    tenant = parse_tenant(data, source="x", today=TODAY)
    mapping: Any = tenant.facts
    assert dict(mapping) == {}
    with pytest.raises(TypeError):
        mapping["hours"] = "changed"


def test_normalize_phrase() -> None:
    assert normalize_phrase("  Lip-Filler!! ") == "lip filler"
    assert normalize_phrase("Juvéderm") == "juvéderm"


# --------------------------------------------------------------------------- YAML traps


def write_variant(tmp_path: Path, old: str, new: str, name: str = "test_spa.yaml") -> Path:
    text = BASE.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / name
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def file_issues(path: Path) -> list[Issue]:
    with pytest.raises(TenantConfigError) as info:
        load_tenant_file(path, today=TODAY)
    [file] = info.value.files
    assert file.source == str(path)
    return list(file.issues)


def test_unquoted_phone_number_is_rejected_with_hint(tmp_path: Path) -> None:
    path = write_variant(tmp_path, 'main_number: "+15555550100"', "main_number: +15555550100")
    [issue] = file_issues(path)
    assert issue.path == "main_number"
    assert "quote it in YAML" in issue.message


def test_yaml_boolean_alias_is_rejected_with_hint(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "aliases: [beta]", "aliases: [no]")
    [issue] = file_issues(path)
    assert issue.path == "services[1].aliases[0]"
    assert "quote it in YAML" in issue.message


def test_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "max_turns: 10", "max_turns: 10\nmax_turns: 20")
    [issue] = file_issues(path)
    assert "duplicate key 'max_turns'" in issue.message
    assert "line 12" in issue.message


def test_nested_duplicate_key_is_rejected(tmp_path: Path) -> None:
    path = write_variant(
        tmp_path, "answers_live: false}", "answers_live: false, answers_live: true}"
    )
    [issue] = file_issues(path)
    assert "duplicate key 'answers_live'" in issue.message


def test_unhashable_key_names_file_and_line(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "max_turns: 10", "max_turns: 10\n? [a, b]\n: x")
    [issue] = file_issues(path)
    assert "unhashable key" in issue.message
    assert "line 12" in issue.message


def test_yaml_syntax_error_names_line(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "max_seconds: 60", "max_seconds: [60")
    [issue] = file_issues(path)
    assert issue.path == ""
    assert "line" in issue.message


def test_nan_price_is_rejected(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "price_max: 20.50", "price_max: .nan")
    [issue] = file_issues(path)
    assert "not a finite decimal" in issue.message


def test_float_prices_load_as_exact_decimals() -> None:
    tenant = load_tenant_file(BASE, today=TODAY)
    price_max = tenant.services[0].price_max
    assert isinstance(price_max, Decimal)
    assert str(price_max) == "20.50"


def test_id_must_match_file_name(tmp_path: Path) -> None:
    path = write_variant(tmp_path, "id: test_spa", "id: test_spa", name="other_spa.yaml")
    [issue] = file_issues(path)
    assert issue.path == "id"


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "test_spa.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    assert file_issues(path)


# --------------------------------------------------------------------------- registry


def test_registry_loads_and_skips_mock_files(tmp_path: Path) -> None:
    shutil.copy(BASE, tmp_path / "test_spa.yaml")
    (tmp_path / "test_spa.mock.yaml").write_text("not: [a tenant", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    registry = TenantRegistry.load(tmp_path, today=TODAY)
    assert registry.ids == ("test_spa",)
    assert registry.get("test_spa") is not None
    assert registry.get("unknown") is None


def test_registry_reports_every_broken_file(tmp_path: Path) -> None:
    write_variant(tmp_path, "max_turns: 10", "max_turns: 0", name="a_spa.yaml")
    write_variant(tmp_path, "id: test_spa", "id: test_spa", name="b_spa.yaml")
    shutil.copy(BASE, tmp_path / "test_spa.yaml")
    with pytest.raises(TenantConfigError) as info:
        TenantRegistry.load(tmp_path, today=TODAY)
    assert [Path(f.source).name for f in info.value.files] == ["a_spa.yaml", "b_spa.yaml"]
    assert str(info.value).startswith("2 tenant files are invalid")


# --------------------------------------------------------------------------- Glow


def test_glow_loads() -> None:
    registry = TenantRegistry.load(ROOT / "tenants", today=TODAY)
    glow = registry.get("glow_aesthetics")
    assert glow is not None
    assert glow.tz.key == "America/New_York"
    tox = glow.services[0]
    assert (tox.price_min, tox.price_max) == (Decimal(13), Decimal(15))
    assert glow.services[2].quotable is False
    assert glow.transfer_tiers[0].live_hours is not None
