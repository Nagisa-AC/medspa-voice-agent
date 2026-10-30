import ast
from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import core.guardrail
from core.guardrail import ABBREVIATIONS, SentenceBuffer


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


@given(
    st.text(max_size=200),
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
    assert imported == {"__future__", "typing"}
    assert "core.numbers" not in imported
    assert "core.state" not in imported
    assert "core.tenant" not in imported
    assert "core.turn" not in imported
    assert "retell" not in imported
