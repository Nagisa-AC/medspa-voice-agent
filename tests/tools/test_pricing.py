import ast
import json
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType

import pytest

import tools.pricing
from core.guardrail import Pass, check_numbers
from core.numbers import Found, find_numbers
from core.state import CallState, new_call_state
from core.tenant import extra_price_words, load_tenant_file
from tools.pricing import QuotePriceResult, format_money, quote_price

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NOW = datetime(2026, 10, 29, 14, 0, tzinfo=UTC)
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
SPA = load_tenant_file(
    Path(__file__).resolve().parents[1] / "core" / "fixtures" / "tenants" / "test_spa.yaml",
    today=TODAY,
)
_GLOW_WORDS = extra_price_words(GLOW)
_SPA_WORDS = extra_price_words(SPA)
glow_extract = partial(
    find_numbers,
    extra_price_words=_GLOW_WORDS.extra_price_words,
    service_words=_GLOW_WORDS.service_words,
)
spa_extract = partial(
    find_numbers,
    extra_price_words=_SPA_WORDS.extra_price_words,
    service_words=_SPA_WORDS.service_words,
)


def glow_state() -> CallState:
    return new_call_state(
        tenant=GLOW, call_id="call_1", data_key_id="key_1", now=NOW, extract=glow_extract
    )


def spa_state() -> CallState:
    return new_call_state(
        tenant=SPA, call_id="call_1", data_key_id="key_1", now=NOW, extract=spa_extract
    )


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


@pytest.mark.parametrize(
    ("value", "spoken"),
    [
        (D(13), "$13"),
        (D("13.00"), "$13"),
        (D("13.5"), "$13.50"),
        (D("13.50"), "$13.50"),
        (D(1200), "$1,200"),
        (D("1200.5"), "$1,200.50"),
    ],
)
def test_format_money(value: Decimal, spoken: str) -> None:
    assert format_money(value) == spoken


def test_botox_range_never_a_total() -> None:
    state = glow_state()
    result = quote_price(GLOW, state, "how much is botox", extract=glow_extract)
    assert result.say == "Botox typically runs $13 to $15 per unit."
    assert "$13 to $15 per unit" in result.say
    assert "total" not in result.say.lower()
    assert result.model["service_id"] == "tox"
    assert result.model["found"] is True
    assert result.model["consult_required"] is False
    assert result.model["deposit_required"] is False
    assert D(13) in state.allowed_amounts
    assert D(15) in state.allowed_amounts
    assert_say_passes(state, result.say, glow_extract)


def test_weight_loss_has_no_number() -> None:
    state = glow_state()
    before = set(state.allowed_amounts)
    result = quote_price(GLOW, state, "ozempic", extract=glow_extract)
    assert result.say == (
        "I don't have a number for Weight Loss Program over the phone. I can book you a consult."
    )
    assert "$" not in result.say
    assert not any(char.isdigit() for char in result.say)
    assert result.model["service_id"] == "weightloss"
    assert result.model["consult_required"] is True
    assert result.model["deposit_required"] is False
    assert state.allowed_amounts == before
    assert_say_passes(state, result.say, glow_extract)


def test_filler_range_admits_both_figures() -> None:
    state = glow_state()
    result = quote_price(GLOW, state, "lip filler", extract=glow_extract)
    assert result.say == "Dermal Filler typically runs $700 to $950 per syringe."
    assert "deposit" not in result.say.lower()
    assert "100" not in result.say
    assert result.model["deposit_required"] is True
    assert D(700) in state.allowed_amounts
    assert D(950) in state.allowed_amounts
    assert_say_passes(state, result.say, glow_extract)


def test_flat_price_when_min_equals_max() -> None:
    state = spa_state()
    result = quote_price(SPA, state, "beta", extract=spa_extract)
    assert result.say == "Beta Treatment is $100 per session."
    assert D(100) in state.allowed_amounts
    assert_say_passes(state, result.say, spa_extract)


def test_cents_in_a_range() -> None:
    state = spa_state()
    result = quote_price(SPA, state, "alpha", extract=spa_extract)
    assert result.say == "Alpha Treatment typically runs $10 to $20.50 per item."
    assert D(10) in state.allowed_amounts
    assert D("20.50") in state.allowed_amounts
    assert_say_passes(state, result.say, spa_extract)


def test_no_match_hedge() -> None:
    state = glow_state()
    result = quote_price(GLOW, state, "something private", extract=glow_extract)
    assert result.say == "Let me have someone follow up with an exact number."
    assert result.model["found"] is False
    assert "service_id" not in result.model
    assert_say_passes(state, result.say, glow_extract)


def test_ambiguous_names_in_mention_order() -> None:
    state = glow_state()
    result = quote_price(GLOW, state, "botox and filler", extract=glow_extract)
    assert result.say == "Which were you asking about, Botox or Dermal Filler?"
    assert result.model["found"] is False
    assert result.model["matches"] == ("tox", "filler")
    assert "$" not in result.say
    assert_say_passes(state, result.say, glow_extract)


def test_ambiguous_three_names() -> None:
    state = spa_state()
    result = quote_price(SPA, state, "alpha and beta and gamma", extract=spa_extract)
    assert result.say == (
        "Which were you asking about, Alpha Treatment, Beta Treatment, or Gamma Program?"
    )
    assert result.model["matches"] == ("alpha", "beta", "gamma")
    assert_say_passes(state, result.say, spa_extract)


@pytest.mark.parametrize(
    "service_text",
    ["how much is botox", "lip filler", "ozempic", "something private", "botox and filler"],
)
def test_every_glow_say_passes_number_check(service_text: str) -> None:
    state = glow_state()
    result = quote_price(GLOW, state, service_text, extract=glow_extract)
    assert_say_passes(state, result.say, glow_extract)
    assert result.model["already_spoken"] is True
    assert "say" not in result.model


def test_model_payload_has_no_price_digits() -> None:
    state = glow_state()
    for text in ("botox", "lip filler"):
        result = quote_price(GLOW, state, text, extract=glow_extract)
        dumped = json.dumps(dict(result.model))
        assert "$" not in dumped
        for fragment in ("13", "15", "700", "950"):
            assert fragment not in dumped
        assert isinstance(result.model, MappingProxyType)


def test_result_is_frozen() -> None:
    state = glow_state()
    result = quote_price(GLOW, state, "botox", extract=glow_extract)
    assert isinstance(result, QuotePriceResult)
    with pytest.raises(TypeError):
        result.model["found"] = False  # type: ignore[index]


def test_pricing_import_layers() -> None:
    tree = ast.parse(Path(tools.pricing.__file__).read_text(encoding="utf-8"))
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
        "decimal",
        "types",
    }
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.guardrail" not in imported
    assert "integrations.booking" not in imported
    assert "retell" not in imported
