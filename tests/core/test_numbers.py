import ast
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import core.numbers
from core.numbers import Found, find_numbers, integer_to_words
from core.state import NumberExtractor, new_call_state
from core.tenant import load_tenant_file

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=date(2026, 10, 29))
NOW = datetime(2026, 10, 29, 14, 0, tzinfo=UTC)


def pairs(text: str, extra: tuple[str, ...] = ()) -> list[tuple[Decimal, str]]:
    return [(found.value, found.kind) for found in find_numbers(text, extra_price_words=extra)]


# --------------------------------------------------------------------------- table


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("700", [(D(700), "plain")]),
        ("$700", [(D(700), "amount")]),
        ("$ 700", [(D(700), "amount")]),
        ("$1,200", [(D(1200), "amount")]),
        ("$13.50", [(D("13.50"), "amount")]),
        ("1,200.50", [(D("1200.50"), "plain")]),
        ("13.5", [(D("13.5"), "plain")]),
        ("$700.", [(D(700), "amount")]),
        ("700 dollars", [(D(700), "amount")]),
        ("seven hundred dollars", [(D(700), "amount")]),
        ("fifty bucks", [(D(50), "amount")]),
        ("50 cents", [(D("0.5"), "amount")]),
        ("13 dollars and 50 cents", [(D("13.5"), "amount")]),
        ("a dollar fifty", [(D("1.50"), "amount")]),
        ("buck fifty", [(D("1.50"), "amount")]),
        ("a buck fifty", [(D("1.50"), "amount")]),
        ("a dollar", [(D(1), "amount")]),
        ("one fifty", [(D("1.50"), "plain"), (D(150), "plain")]),
        ("nine fifty", [(D("9.50"), "plain"), (D(950), "plain")]),
        ("nine-fifty", [(D("9.50"), "plain"), (D(950), "plain")]),
        ("thirteen fifty", [(D("13.50"), "plain"), (D(1350), "plain")]),
        ("eleven fifty", [(D("11.50"), "plain"), (D(1150), "plain")]),
        ("twenty-one fifty", [(D("21.50"), "plain"), (D(2150), "plain")]),
        ("two fifty", [(D("2.50"), "plain"), (D(250), "plain")]),
        ("two thirty", [(D(2), "plain"), (D(30), "plain")]),
        ("2:00 PM", [(D(2), "plain"), (D(0), "plain")]),
        ("2:00", [(D(2), "plain"), (D(0), "plain")]),
        ("2 PM", [(D(2), "plain")]),
        ("2PM", [(D(2), "plain")]),
        ("2pm", [(D(2), "plain")]),
        ("2 p.m.", [(D(2), "plain")]),
        ("2 am", [(D(2), "plain")]),
        ("two o'clock", [(D(2), "plain"), (D(0), "plain")]),
        ("two thirty pm", [(D(2), "plain"), (D(30), "plain")]),
        ("seven hundred", [(D(700), "plain")]),
        ("twelve hundred", [(D(1200), "plain")]),
        ("a hundred", [(D(100), "plain")]),
        ("a thousand", [(D(1000), "plain")]),
        ("seven hundred and fifty", [(D(750), "plain")]),
        ("twenty-four", [(D(24), "plain")]),
        ("zero", [(D(0), "plain")]),
        ("three thousand", [(D(3000), "plain")]),
        ("two thousand four hundred", [(D(2400), "plain")]),
        ("ninety-nine", [(D(99), "plain")]),
        ("one hundred thousand", [(D(100000), "plain")]),
        ("4th", [(D(4), "plain")]),
        ("3rd", [(D(3), "plain")]),
        ("1st", [(D(1), "plain")]),
        ("twenty-fourth", [(D(24), "plain")]),
        ("first", [(D(1), "plain")]),
        ("$13-15", [(D(13), "amount"), (D(15), "amount")]),
        ("$13 to 15", [(D(13), "amount"), (D(15), "amount")]),
        ("$700 to $950", [(D(700), "amount"), (D(950), "amount")]),
        ("13-15", [(D(13), "plain"), (D(15), "plain")]),
        ("13 to 15", [(D(13), "plain"), (D(15), "plain")]),
        ("thirteen to fifteen dollars", [(D(13), "amount"), (D(15), "amount")]),
        ("20 to 60 units", [(D(20), "plain"), (D(60), "plain")]),
        ("Botox is 12 a unit", [(D(12), "price_context")]),
        ("two thirty a unit", [(D(2), "price_context"), (D(30), "price_context")]),
        ("I'll need about 2 seconds", [(D(2), "plain")]),
        ("it runs 800", [(D(800), "price_context")]),
        ("filler's about 900 a syringe", [(D(900), "price_context")]),
        ("costs 50", [(D(50), "price_context")]),
        ("starts at 700", [(D(700), "price_context")]),
        ("starting at 13", [(D(13), "price_context")]),
        ("goes for 40", [(D(40), "price_context")]),
        ("is about 900", [(D(900), "price_context")]),
        ("is around 10", [(D(10), "price_context")]),
        ("pay 50", [(D(50), "price_context")]),
        ("price 13", [(D(13), "price_context")]),
        ("12 per unit", [(D(12), "price_context")]),
        ("12 each syringe", [(D(12), "price_context")]),
        ("12 every treatment", [(D(12), "price_context")]),
        ("12 a vial", [(D(12), "price_context")]),
        ("12 a session", [(D(12), "price_context")]),
        ("12 a area", [(D(12), "price_context")]),
        ("$13.50 per unit", [(D("13.50"), "amount")]),
        ("4th a unit", [(D(4), "price_context")]),
        ("Open 9 to 6", [(D(9), "plain"), (D(6), "plain")]),
        ("Suite 200", [(D(200), "plain")]),
        ("24 hours", [(D(24), "plain")]),
        ("$50 fee", [(D(50), "amount")]),
        ("under 18", [(D(18), "plain")]),
        ("1420 Chapel", [(D(1420), "plain")]),
        (
            "Monday through Friday 9 to 6, Saturday 10 to 4",
            [(D(9), "plain"), (D(6), "plain"), (D(10), "plain"), (D(4), "plain")],
        ),
        (
            "24 hours notice, otherwise there's a $50 fee.",
            [(D(24), "plain"), (D(50), "amount")],
        ),
    ],
)
def test_table(text: str, expected: list[tuple[Decimal, str]]) -> None:
    assert pairs(text) == expected


def test_phone_groups() -> None:
    found = set(pairs("203-555-0100"))
    assert found == {
        (D("2035550100"), "plain"),
        (D(203), "plain"),
        (D(555), "plain"),
        (D(100), "plain"),
    }


def test_phone_plus_one() -> None:
    found = set(pairs("+1 203 555 0100"))
    assert found == {
        (D("12035550100"), "plain"),
        (D(203), "plain"),
        (D(555), "plain"),
        (D(100), "plain"),
    }


def test_compact_phone_is_one_number() -> None:
    assert pairs("+12035550100") == [(D("12035550100"), "plain")]


def test_tenant_price_unit_phrase() -> None:
    extra = ("per 15 minutes",)
    found = pairs("10 per 15 minutes", extra)
    assert (D(10), "price_context") in found
    assert D(15) in {value for value, _ in found}


def test_glow_price_unit_phrase_promotes() -> None:
    extra = ("per unit", "per syringe")
    assert pairs("12 per unit", extra) == [(D(12), "price_context")]
    assert pairs("900 per syringe", extra) == [(D(900), "price_context")]


def test_span_matches_source() -> None:
    text = "It costs $13.50 today."
    found = find_numbers(text)
    assert found == [Found(D("13.50"), "amount", (text.index("$13.50"), text.index("$13.50") + 6))]
    assert text[found[0].span[0] : found[0].span[1]] == "$13.50"


def test_bind_extra_price_words_at_call_site() -> None:
    extract: NumberExtractor = partial(find_numbers, extra_price_words=("per unit",))
    found = list(extract("Botox is 12 a unit"))
    assert [(item.value, item.kind) for item in found] == [(D(12), "price_context")]


def test_values_are_decimal_never_float() -> None:
    for found in find_numbers("$13.50 and one fifty and 1,200"):
        assert type(found.value) is Decimal


# --------------------------------------------------------------------------- hypothesis


def _money_text(amount: Decimal) -> str:
    quantized = amount.quantize(D("0.01"))
    if quantized == quantized.to_integral_value():
        n = int(quantized)
        body = f"{n:,}" if n >= 1000 else str(n)
        return f"${body}"
    return f"${quantized:.2f}"


@given(st.integers(min_value=0, max_value=9999))
def test_round_trip_integer_words(n: int) -> None:
    text = integer_to_words(n)
    values = {found.value for found in find_numbers(text)}
    assert D(n) in values, text


@given(st.integers(min_value=0, max_value=9999))
def test_round_trip_integer_digits_and_dollars(n: int) -> None:
    forms = [str(n), f"${n}", f"{n} dollars"]
    if n >= 1000:
        forms.append(f"${n:,}")
    for text in forms:
        values = {found.value for found in find_numbers(text)}
        assert D(n) in values, text


@given(
    st.decimals(
        min_value=0,
        max_value=9999,
        places=2,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_round_trip_money(amount: Decimal) -> None:
    text = _money_text(amount)
    found = find_numbers(text)
    assert any(item.value == amount and item.kind == "amount" for item in found), (text, found)


# --------------------------------------------------------------------------- glow seed


def test_glow_seed_includes_cancellation_fee() -> None:
    units = tuple(service.price_unit for service in GLOW.services if service.price_unit)

    def extract(text: str) -> list[Found]:
        return find_numbers(text, extra_price_words=units)

    state = new_call_state(
        tenant=GLOW,
        call_id="call_1",
        data_key_id="key_1",
        now=NOW,
        extract=extract,
    )
    assert state.allowed_amounts == {D(50)}
    assert state.allowed_numbers == {D(n) for n in (1420, 200, 9, 6, 10, 4, 24, 18, 50)}


# --------------------------------------------------------------------------- layers


def test_numbers_import_layers() -> None:
    tree = ast.parse(Path(core.numbers.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "core.state" not in imported
    assert "core.tenant" not in imported
    assert "core.policy" not in imported
    assert "core.turn" not in imported
    assert "retell" not in imported
    assert "core.crypto" not in imported
    assert "core.events" not in imported
