import ast
import copy
import json
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

import tools.faq
from core.guardrail import Blocked, Pass, check_clinical, check_numbers, load_output_pack
from core.numbers import Found, find_numbers
from core.state import CallState, new_call_state
from core.tenant import Tenant, extra_price_words, load_tenant_file, parse_tenant, read_yaml
from tools.faq import LookupFaqResult, lookup_faq

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NOW = datetime(2026, 10, 29, 14, 0, tzinfo=UTC)
BASE = Path(__file__).resolve().parents[1] / "core" / "fixtures" / "tenants" / "test_spa.yaml"
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
SPA = load_tenant_file(BASE, today=TODAY)
PACK = load_output_pack(ROOT / "packs" / "medspa.output.yaml")
TOX = next(entry for entry in GLOW.faq if entry.id == "tox_duration")
CLINICAL_SAY = "Botox usually lasts three months"
_GLOW_WORDS = extra_price_words(GLOW)
glow_extract = partial(
    find_numbers,
    extra_price_words=_GLOW_WORDS.extra_price_words,
    service_words=_GLOW_WORDS.service_words,
)


def glow_state() -> CallState:
    return new_call_state(
        tenant=GLOW, call_id="call_1", data_key_id="key_1", now=NOW, extract=glow_extract
    )


def spa_state(tenant: Tenant = SPA) -> CallState:
    words = extra_price_words(tenant)
    extract = partial(
        find_numbers,
        extra_price_words=words.extra_price_words,
        service_words=words.service_words,
    )
    return new_call_state(
        tenant=tenant, call_id="call_1", data_key_id="key_1", now=NOW, extract=extract
    )


def spa_extract_for(tenant: Tenant) -> Callable[[str], Iterable[Found]]:
    words = extra_price_words(tenant)
    return partial(
        find_numbers,
        extra_price_words=words.extra_price_words,
        service_words=words.service_words,
    )


def spa_with_faq(entries: list[dict[str, Any]]) -> Tenant:
    data: Any = copy.deepcopy(read_yaml(BASE))
    data["faq"] = entries
    return parse_tenant(data, source="test_spa.yaml", today=TODAY)


def faq_entry(
    faq_id: str, topics: list[str], say: str, *, approved_on: date = date(2026, 1, 15)
) -> dict[str, Any]:
    return {
        "id": faq_id,
        "topics": topics,
        "say": say,
        "approved_by": "Test Approver",
        "approved_on": approved_on,
    }


def ask(
    tenant: Tenant,
    state: CallState,
    question_text: str,
    extract: Callable[[str], Iterable[Found]],
) -> LookupFaqResult:
    return lookup_faq(tenant, state, question_text, deflect=PACK.deflect, extract=extract)


def assert_say_passes(
    state: CallState, say: str, extract: Callable[[str], Iterable[Found]]
) -> None:
    result = check_numbers(
        say,
        allowed_amounts=state.allowed_amounts,
        allowed_numbers=state.allowed_numbers,
        extract=extract,
    )
    assert isinstance(result, Pass)


def test_botox_duration_serves_tox_duration() -> None:
    state = glow_state()
    before = set(state.allowed_numbers)
    assert D(3) not in before
    result = ask(GLOW, state, "how long does botox last", glow_extract)
    assert result.say == TOX.say
    assert result.model["found"] is True
    assert result.model["faq_id"] == "tox_duration"
    assert result.model["already_spoken"] is True
    assert D(3) in state.allowed_numbers
    assert D(4) in state.allowed_numbers
    assert_say_passes(state, result.say, glow_extract)


def test_two_entries_deflect() -> None:
    tenant = spa_with_faq(
        [
            faq_entry("alpha_duration", ["alpha last"], "A test answer."),
            faq_entry("how_long", ["how long"], "Another answer."),
        ]
    )
    extract = spa_extract_for(tenant)
    state = spa_state(tenant)
    result = ask(tenant, state, "how long does alpha last", extract)
    assert result.say == PACK.deflect
    assert result.model["found"] is False
    assert "faq_id" not in result.model
    assert_say_passes(state, result.say, extract)


def test_nested_topics_across_entries_deflect() -> None:
    tenant = spa_with_faq(
        [
            faq_entry("botox_last", ["botox last"], "Duration from the longer topic."),
            faq_entry("bare_last", ["last"], "Duration from the nested topic."),
        ]
    )
    extract = spa_extract_for(tenant)
    state = spa_state(tenant)
    result = ask(tenant, state, "how long does botox last", extract)
    assert result.say == PACK.deflect
    assert result.model["found"] is False
    assert_say_passes(state, result.say, extract)


def test_no_match_deflect() -> None:
    state = glow_state()
    result = ask(GLOW, state, "something private", glow_extract)
    assert result.say == PACK.deflect
    assert result.model["found"] is False
    assert "faq_id" not in result.model
    assert_say_passes(state, result.say, glow_extract)


def test_two_topics_on_one_entry_still_serves() -> None:
    state = glow_state()
    result = ask(GLOW, state, "how long does botox last before it wears off", glow_extract)
    assert result.say == TOX.say
    assert result.model["faq_id"] == "tox_duration"
    assert_say_passes(state, result.say, glow_extract)


@pytest.mark.parametrize(
    "question_text",
    [
        "how long does botox last",
        "how long does botox last before it wears off",
        "something private",
    ],
)
def test_every_glow_say_passes_number_check(question_text: str) -> None:
    state = glow_state()
    result = ask(GLOW, state, question_text, glow_extract)
    assert_say_passes(state, result.say, glow_extract)
    assert result.model["already_spoken"] is True
    assert "say" not in result.model


def test_fixture_faq_say_is_blocked_by_clinical_check() -> None:
    tenant = spa_with_faq([faq_entry("clinical_duration", ["botox last"], CLINICAL_SAY)])
    extract = spa_extract_for(tenant)
    state = spa_state(tenant)
    result = ask(tenant, state, "how long does botox last", extract)
    assert result.say == CLINICAL_SAY
    clinical = check_clinical(result.say, PACK)
    assert isinstance(clinical, Blocked)
    assert clinical.check == "clinical"
    assert "lasts" in clinical.pattern_ids
    assert_say_passes(state, result.say, extract)


def test_model_payload_has_no_answer_text() -> None:
    state = glow_state()
    result = ask(GLOW, state, "how long does botox last", glow_extract)
    dumped = json.dumps(dict(result.model))
    for fragment in ("three", "four", "months", "clients", "injector"):
        assert fragment not in dumped
    assert TOX.say not in dumped
    assert "say" not in result.model
    assert isinstance(result.model, MappingProxyType)


def test_result_is_frozen() -> None:
    state = glow_state()
    result = ask(GLOW, state, "how long does botox last", glow_extract)
    assert isinstance(result, LookupFaqResult)
    with pytest.raises(TypeError):
        result.model["found"] = False  # type: ignore[index]


def test_blank_deflect_raises() -> None:
    state = glow_state()
    with pytest.raises(ValueError, match="non-empty pack line"):
        lookup_faq(GLOW, state, "how long does botox last", deflect="  ", extract=glow_extract)


def test_faq_import_layers() -> None:
    tree = ast.parse(Path(tools.faq.__file__).read_text(encoding="utf-8"))
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
        "types",
    }
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "integrations.booking" not in imported
    assert "retell" not in imported
