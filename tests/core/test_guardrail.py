import ast
from datetime import UTC, date, datetime
from decimal import Decimal
from functools import partial
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import core.guardrail
from core.guardrail import (
    ABBREVIATIONS,
    Blocked,
    OutputPackError,
    Pass,
    SentenceBuffer,
    check_clinical,
    check_numbers,
    check_sentence,
    load_output_pack,
)
from core.numbers import find_numbers
from core.state import CallState, admit_numbers, new_call_state
from core.tenant import extra_price_words, load_tenant_file

D = Decimal
ROOT = Path(__file__).resolve().parents[2]
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=date(2026, 10, 29))
NOW = datetime(2026, 10, 29, 14, 0, tzinfo=UTC)
_GLOW_WORDS = extra_price_words(GLOW)
glow_extract = partial(
    find_numbers,
    extra_price_words=_GLOW_WORDS.extra_price_words,
    service_words=_GLOW_WORDS.service_words,
)


def split_text(text: str) -> list[str]:
    buffer = SentenceBuffer()
    sentences = buffer.feed(text)
    sentences.extend(buffer.flush())
    return sentences


def split_chunks(chunks: list[str]) -> list[str]:
    buffer = SentenceBuffer()
    sentences: list[str] = []
    for chunk in chunks:
        sentences.extend(buffer.feed(chunk))
    sentences.extend(buffer.flush())
    return sentences


# --------------------------------------------------------------------------- spec and table


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$13.50 per unit.", ["$13.50 per unit."]),
        ("1420 Chapel St. Suite 200.", ["1420 Chapel St. Suite 200."]),
        ("nine fifty.", ["nine fifty."]),
        ("See you at 2 p.m. tomorrow.", ["See you at 2 p.m. tomorrow."]),
        ("See e.g. the flyer.", ["See e.g. the flyer."]),
        ("That is i.e. fine.", ["That is i.e. fine."]),
        ("Compare a vs. b today.", ["Compare a vs. b today."]),
        ("Dr. Jane is in.", ["Dr. Jane is in."]),
        ("Meet Mr. Smith today.", ["Meet Mr. Smith today."]),
        ("Meet Mrs. Smith today.", ["Meet Mrs. Smith today."]),
        ("Meet Ms. Smith today.", ["Meet Ms. Smith today."]),
        ("Apt. 4 is ready.", ["Apt. 4 is ready."]),
        ("Ste. 200 is open.", ["Ste. 200 is open."]),
        ("Room No. 12 is free.", ["Room No. 12 is free."]),
        ("John Jr. is here.", ["John Jr. is here."]),
        ("Closes at 9 a.m. tomorrow.", ["Closes at 9 a.m. tomorrow."]),
        ("No. We don't.", ["No. We don't."]),
        ("Hello? Yes.", ["Hello? ", "Yes."]),
        ("Stop! Go.", ["Stop! ", "Go."]),
        ('"Hello." Next.', ['"Hello." ', "Next."]),
        ("Wait… okay.", ["Wait… ", "okay."]),
        ("Wait... okay.", ["Wait... ", "okay."]),
        ("Hello.\nWorld.", ["Hello.\n", "World."]),
        ("Hello.\n\nWorld.", ["Hello.\n", "\nWorld."]),
        ("Hello.  World.", ["Hello. ", " World."]),
        ("nine\nfifty.", ["nine\nfifty."]),
        ("The 1st. Next.", ["The 1st. ", "Next."]),
        ("Hello.World", ["Hello.World"]),
        ("", []),
    ],
)
def test_table(text: str, expected: list[str]) -> None:
    assert split_text(text) == expected


def test_required_abbreviations_are_listed() -> None:
    required = {
        "st.",
        "ste.",
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "jr.",
        "apt.",
        "no.",
        "vs.",
        "e.g.",
        "i.e.",
        "a.m.",
        "p.m.",
    }
    assert required <= {item.casefold() for item in ABBREVIATIONS}


def test_decimal_split_across_chunks_stays_one_sentence() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("$13.") == []
    assert buffer.feed("50 per unit.") == []
    assert buffer.flush() == ["$13.50 per unit."]


def test_number_words_split_across_chunks_stay_one_sentence() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("nine ") == []
    assert buffer.feed("fifty.") == []
    assert buffer.flush() == ["nine fifty."]


def test_extra_space_chunk_split_matches_whole() -> None:
    text = "Hello.  World."
    chunks = ["Hello. ", " World."]
    assert "".join(chunks) == text
    expected = ["Hello. ", " World."]
    assert split_text(text) == expected
    assert split_chunks(chunks) == expected


def test_blank_line_split_between_newlines_matches_whole() -> None:
    text = "Hello.\n\nWorld."
    chunks = ["Hello.\n", "\nWorld."]
    assert "".join(chunks) == text
    expected = ["Hello.\n", "\nWorld."]
    assert split_text(text) == expected
    assert split_chunks(chunks) == expected


def test_period_at_chunk_end_waits() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("The fee is $50.") == []
    assert buffer.feed(" Next.") == ["The fee is $50. "]
    assert buffer.flush() == ["Next."]


def test_character_at_a_time_matches_whole() -> None:
    text = "$13.50 per unit. 1420 Chapel St. Suite 200."
    assert split_chunks(list(text)) == split_text(text)


def test_empty_feed_and_flush() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("") == []
    assert buffer.flush() == []


def test_flush_remainder_without_terminator() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("No terminator") == []
    assert buffer.flush() == ["No terminator"]


def test_run_on_without_terminator_flushes_whole() -> None:
    text = "nine fifty " * 200
    assert len(text) >= 2000
    buffer = SentenceBuffer()
    assert buffer.feed(text) == []
    assert buffer.flush() == [text]


def test_concatenated_output_equals_input_for_spec_cases() -> None:
    for text in (
        "$13.50 per unit.",
        "1420 Chapel St. Suite 200.",
        "Hello? Yes.",
        "nine\nfifty.",
    ):
        assert "".join(split_text(text)) == text


# --------------------------------------------------------------------------- hypothesis


_HYPOTHESIS_PIECES = (". ", " ", "\n", "$13", "50", "Dr", "nine", "fifty", "?")


@given(
    st.lists(st.sampled_from(_HYPOTHESIS_PIECES), max_size=40).map("".join),
    st.lists(st.integers(min_value=0), max_size=25),
)
def test_random_chunks_match_whole_and_concat(text: str, cuts: list[int]) -> None:
    whole = split_text(text)
    assert "".join(whole) == text
    n = len(text)
    points = sorted({0, n, *(cut % (n + 1) for cut in cuts)}) if n else [0]
    if n:
        chunks = [text[a:b] for a, b in pairwise(points)]
    else:
        chunks = [""]
    assert split_chunks(chunks) == whole


# --------------------------------------------------------------------------- layers


def test_guardrail_import_layers() -> None:
    tree = ast.parse(Path(core.guardrail.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections.abc",
        "core.numbers",
        "dataclasses",
        "decimal",
        "pathlib",
        "re",
        "typing",
        "yaml",
    }
    assert "core.state" not in imported
    assert "core.tenant" not in imported
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "retell" not in imported


# --------------------------------------------------------------------------- number check


def _glow_state() -> CallState:
    return new_call_state(
        tenant=GLOW,
        call_id="call_1",
        data_key_id="key_1",
        now=NOW,
        extract=glow_extract,
    )


def _check(sentence: str, state: CallState | None = None) -> Pass | Blocked:
    call = state if state is not None else _glow_state()
    return check_numbers(
        sentence,
        allowed_amounts=call.allowed_amounts,
        allowed_numbers=call.allowed_numbers,
        extract=glow_extract,
    )


def test_glow_extra_price_words_include_units_and_names() -> None:
    folded = {item.casefold() for item in _GLOW_WORDS.extra_price_words}
    names = {item.casefold() for item in _GLOW_WORDS.service_words}
    assert "per unit" in folded
    assert "per syringe" in folded
    assert "botox" in names
    assert "filler" in names
    assert "botox" not in folded


def test_unquoted_total_blocks() -> None:
    result = _check("$450 total")
    assert isinstance(result, Blocked)
    assert result.reason == "unquoted_amount"


def test_filler_syringe_price_blocks() -> None:
    result = _check("filler's about 900 a syringe")
    assert isinstance(result, Blocked)
    assert result.reason == "unquoted_price"


def test_seeded_cancellation_fee_passes() -> None:
    assert isinstance(_check("there's a $50 fee"), Pass)


def test_quote_range_passes_only_after_admit() -> None:
    sentence = "$700 to $950 per syringe"
    state = _glow_state()
    assert isinstance(_check(sentence, state), Blocked)
    admit_numbers(state, sentence, glow_extract)
    assert isinstance(_check(sentence, state), Pass)


def test_suite_number_passes() -> None:
    assert isinstance(_check("Suite 200"), Pass)


def test_about_two_seconds_passes() -> None:
    assert isinstance(_check("I'll need about 2 seconds"), Pass)
    assert isinstance(_check("about 2 seconds"), Pass)


def test_botox_is_twelve_blocks() -> None:
    result = _check("Botox is 12")
    assert isinstance(result, Blocked)
    assert result.reason == "unquoted_price"
    assert result.blocked_price_context >= 1


def test_two_thirty_a_unit_stays_price_context() -> None:
    found = list(glow_extract("two thirty a unit"))
    assert {(item.value, item.kind) for item in found} == {
        (D(2), "price_context"),
        (D(30), "price_context"),
    }
    assert isinstance(_check("two thirty a unit"), Blocked)


def test_booking_readback_with_service_name_passes() -> None:
    assert isinstance(_check("So that's Botox Thursday at 2 PM?"), Pass)


def test_ordinal_slot_readback_passes_when_plains_admitted() -> None:
    state = _glow_state()
    state.allowed_numbers.update({D(12), D(2), D(30)})
    assert isinstance(_check("Botox on the 12th at 2:30", state), Pass)


def test_nine_fifty_needs_both_readings() -> None:
    state = _glow_state()
    assert isinstance(_check("nine fifty", state), Blocked)
    state.allowed_numbers.add(D("9.50"))
    assert isinstance(_check("nine fifty", state), Blocked)
    state.allowed_numbers.add(D(950))
    assert isinstance(_check("nine fifty", state), Pass)


def test_plain_two_point_five_blocks() -> None:
    result = _check("2.5")
    assert isinstance(result, Blocked)
    assert result.reason == "unknown_number"


def test_blocked_repr_has_no_digits_from_the_sentence() -> None:
    sentence = "Call 203-555-0100 for $450."
    result = _check(sentence)
    assert isinstance(result, Blocked)
    shown = repr(result) + str(result)
    for fragment in ("203", "555", "0100", "450", sentence):
        assert fragment not in shown


# --------------------------------------------------------------------------- clinical output check


PACK = load_output_pack(ROOT / "packs" / "medspa.output.yaml")
_DEFLECT = (
    "That's a great question for our injector. I can book you a consult, "
    "or have them call you back."
)
_PATTERN_CASES: list[tuple[str, str, str]] = [
    ("side_effects", "There can be side effects.", "I'll set that aside."),
    ("risks", "There are some risks.", "The parking is around the back."),
    ("safe", "it's totally safe", "safe to say"),
    ("pain", "it can be painful", "pain in the neck"),
    ("hurt", "it might hurt a little", "See you Thursday."),
    ("numb", "you may feel numbing", "I'll take the number"),
    ("downtime", "there's no downtime", "I'll write it down"),
    ("recovery", "recovery takes a week", "I'll recover the slot."),
    ("heal", "it heals in a few days", "health insurance is on file"),
    ("swelling", "you'll have some swelling", "The building has parking."),
    ("bruise", "some bruising is common", "I'll browse the calendar"),
    ("lasts", "Botox can last up to four months", "last name"),
    ("wears_off", "it wears off in three months", "the wear and tear"),
    ("results", "you'll see results in two weeks", "search results"),
    ("dose", "the dose is decided at the visit", "See you soon."),
    ("mg", "that's 2 mg", "I'm good"),
    ("units_needed", "units you'll need", "20 to 60 units"),
    ("contraindication", "that's a contraindication", "I have no concerns"),
    ("pregnancy", "if you're pregnant", "I'll take the premium slot"),
    ("breastfeeding", "while breastfeeding", "I'll book the morning"),
    ("allergy", "any allergies", "I'll see you later."),
    ("candidate", "are you a candidate", "I'll take Tuesday"),
    ("aftercare", "aftercare is rest", "after the appointment"),
    ("avoid_aftercare", "avoid exercise for a day", "avoid the highway"),
    (
        "reassurance",
        "A headache after Botox is totally normal, it'll go away.",
        "I can book Thursday morning.",
    ),
    ("symptoms", "there's some redness", "The parking lot is behind the building."),
    ("advice", "take ibuprofen if needed", "That's a nice idea."),
]


def _write_pack(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pack.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _clinical(sentence: str) -> Pass | Blocked:
    return check_clinical(sentence, PACK)


def test_production_pack_loads() -> None:
    assert PACK.deflect == _DEFLECT
    assert {pattern.id for pattern in PACK.patterns} == {row[0] for row in _PATTERN_CASES}


def test_every_pack_id_has_a_case() -> None:
    assert [pattern.id for pattern in PACK.patterns] == [row[0] for row in _PATTERN_CASES]


@pytest.mark.parametrize(("pattern_id", "positive", "negative"), _PATTERN_CASES)
def test_each_pattern_has_a_hit_and_a_miss(pattern_id: str, positive: str, negative: str) -> None:
    hit = _clinical(positive)
    assert isinstance(hit, Blocked)
    assert hit.check == "clinical"
    assert pattern_id in hit.pattern_ids
    miss = _clinical(negative)
    assert isinstance(miss, Pass)


@pytest.mark.parametrize(
    "sentence",
    [
        "Botox can last up to four months",
        "it'll last about three months",
        "lasts three months",
        "Botox usually lasts three months",
        "Filler lasts about a year.",
    ],
)
def test_lasts_true_hits(sentence: str) -> None:
    result = _clinical(sentence)
    assert isinstance(result, Blocked)
    assert "lasts" in result.pattern_ids


@pytest.mark.parametrize(
    "sentence",
    [
        "last name",
        "last week",
        "at last",
        "your last visit",
        "last weekend",
        "last time",
        "last appointment",
        "last night",
        "last month",
        "last year",
        "last call",
    ],
)
def test_lasts_false_hits(sentence: str) -> None:
    assert isinstance(_clinical(sentence), Pass)


@pytest.mark.parametrize(
    "sentence",
    [
        "you'll see results in two weeks",
        "results vary",
        "great results",
    ],
)
def test_results_true_hits(sentence: str) -> None:
    result = _clinical(sentence)
    assert isinstance(result, Blocked)
    assert "results" in result.pattern_ids


def test_search_results_do_not_block() -> None:
    assert isinstance(_clinical("search results"), Pass)
    assert isinstance(_clinical("search result"), Pass)


def test_key_reassurance_sentence_also_hits_symptoms() -> None:
    sentence = "A headache after Botox is totally normal, it'll go away."
    result = _clinical(sentence)
    assert isinstance(result, Blocked)
    assert "reassurance" in result.pattern_ids
    assert "symptoms" in result.pattern_ids


def test_curly_apostrophe_units_needed() -> None:
    sentence = "units you\u2019ll need"
    result = _clinical(sentence)
    assert isinstance(result, Blocked)
    assert "units_needed" in result.pattern_ids
    shown = repr(result) + str(result)
    assert sentence not in shown
    assert "you\u2019ll" not in shown


def test_clinical_blocked_repr_has_no_sentence() -> None:
    sentence = "Botox can last up to four months"
    result = _clinical(sentence)
    assert isinstance(result, Blocked)
    shown = repr(result) + str(result)
    assert sentence not in shown
    assert "Botox" not in shown
    assert "four" not in shown
    assert result.reason is None
    assert result.blocked_amount == 0


def test_clinical_wins_when_both_checks_fail() -> None:
    sentence = "Botox is 12 and it's totally safe"
    result = check_sentence(
        sentence,
        PACK,
        allowed_amounts=_glow_state().allowed_amounts,
        allowed_numbers=_glow_state().allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(result, Blocked)
    assert result.check == "clinical"
    assert "safe" in result.pattern_ids
    numbers = _check(sentence)
    assert isinstance(numbers, Blocked)
    assert numbers.check == "numbers"


def test_check_sentence_number_only() -> None:
    sentence = "So that'll be about $450 total."
    result = check_sentence(
        sentence,
        PACK,
        allowed_amounts=_glow_state().allowed_amounts,
        allowed_numbers=_glow_state().allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(result, Blocked)
    assert result.check == "numbers"
    assert result.pattern_ids == ()


def test_check_sentence_passes_booking_readback() -> None:
    result = check_sentence(
        "So that's Botox Thursday at 2 PM?",
        PACK,
        allowed_amounts=_glow_state().allowed_amounts,
        allowed_numbers=_glow_state().allowed_numbers,
        extract=glow_extract,
    )
    assert isinstance(result, Pass)


def test_load_missing_deflect(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        "patterns:\n  - id: side_effects\n    regex: 'x'\n",
    )
    with pytest.raises(OutputPackError, match="missing deflect"):
        load_output_pack(path)


def test_load_blank_deflect(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        "deflect: '   '\npatterns:\n  - id: side_effects\n    regex: 'x'\n",
    )
    with pytest.raises(OutputPackError, match="missing deflect"):
        load_output_pack(path)


def test_load_duplicate_id(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        "deflect: keep\npatterns:\n"
        "  - id: side_effects\n    regex: 'a'\n"
        "  - id: side_effects\n    regex: 'b'\n",
    )
    with pytest.raises(OutputPackError, match="duplicate id 'side_effects'"):
        load_output_pack(path)


def test_load_invalid_regex(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        "deflect: keep\npatterns:\n  - id: side_effects\n    regex: '('\n",
    )
    with pytest.raises(OutputPackError, match=r"patterns\[0\]\.regex invalid"):
        load_output_pack(path)


def test_load_missing_id(tmp_path: Path) -> None:
    path = _write_pack(
        tmp_path,
        "deflect: keep\npatterns:\n  - regex: 'x'\n",
    )
    with pytest.raises(OutputPackError, match=r"patterns\[0\]\.id must be a slug"):
        load_output_pack(path)


def test_load_empty_patterns(tmp_path: Path) -> None:
    path = _write_pack(tmp_path, "deflect: keep\npatterns: []\n")
    with pytest.raises(OutputPackError, match="patterns must be a non-empty list"):
        load_output_pack(path)


def test_load_invalid_yaml(tmp_path: Path) -> None:
    path = _write_pack(tmp_path, "deflect: [\n")
    with pytest.raises(OutputPackError, match="invalid YAML"):
        load_output_pack(path)
