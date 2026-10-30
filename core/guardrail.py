"""Sentence buffer and number check on model sentences. Clinical check lands later."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Set
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from core.numbers import Found

_ELLIPSIS = "\u2026"
_ASCII_ELLIPSIS = "..."
_CLOSERS = frozenset({'"', "'", "\u201d", "\u2019", ")", "]"})
_SIMPLE_ABBREVS = frozenset({"st", "ste", "dr", "mr", "mrs", "ms", "jr", "apt", "no", "vs"})
_DOTTED_ABBREVS = frozenset({"e.g.", "i.e.", "a.m.", "p.m."})

ABBREVIATIONS = frozenset({f"{word}." for word in _SIMPLE_ABBREVS}) | _DOTTED_ABBREVS

_Boundary = int | Literal["wait"] | None


class SentenceBuffer:
    """Hold streamed model text until a sentence boundary, so numbers stay whole."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, token: str) -> list[str]:
        """Append a model chunk. Return every sentence this chunk completed."""
        self._buf += token
        return self._extract(end_of_stream=False)

    def flush(self) -> list[str]:
        """End of stream: emit remaining complete sentences, then the remainder."""
        completed = self._extract(end_of_stream=True)
        if self._buf:
            completed.append(self._buf)
            self._buf = ""
        return completed

    def _extract(self, *, end_of_stream: bool) -> list[str]:
        completed: list[str] = []
        while self._buf:
            end = _find_sentence_end(self._buf, end_of_stream)
            if end is None:
                break
            completed.append(self._buf[:end])
            self._buf = self._buf[end:]
        return completed


def _find_sentence_end(buf: str, end_of_stream: bool) -> int | None:
    i = 0
    n = len(buf)
    while i < n:
        if buf.startswith(_ASCII_ELLIPSIS, i):
            result = _close_boundary(buf, i + 3, end_of_stream)
            if result == "wait":
                return None
            if isinstance(result, int):
                return result
            i += 1
            continue
        char = buf[i]
        if char == _ELLIPSIS or char in "?!":
            result = _close_boundary(buf, i + 1, end_of_stream)
            if result == "wait":
                return None
            if isinstance(result, int):
                return result
            i += 1
            continue
        if char == ".":
            if i + 1 < n and buf[i + 1].isdigit():
                i += 1
                continue
            if i + 1 >= n and not end_of_stream:
                return None
            result = _close_boundary(buf, i + 1, end_of_stream)
            if result == "wait":
                return None
            if isinstance(result, int):
                if _is_abbreviation(buf, i):
                    i += 1
                    continue
                return result
        i += 1
    return None


def _close_boundary(buf: str, j: int, end_of_stream: bool) -> _Boundary:
    n = len(buf)
    while j < n and buf[j] in _CLOSERS:
        j += 1
    if j < n:
        if buf[j].isspace():
            return j + 1
        return None
    if end_of_stream:
        return n
    return "wait"


def _is_abbreviation(buf: str, period: int) -> bool:
    head = buf[: period + 1].casefold()
    for dotted in _DOTTED_ABBREVS:
        if not head.endswith(dotted):
            continue
        before = period + 1 - len(dotted)
        if before <= 0 or not buf[before - 1].isalnum():
            return True
    i = period - 1
    while i >= 0 and buf[i].isalpha():
        i -= 1
    word = buf[i + 1 : period].casefold()
    if word not in _SIMPLE_ABBREVS:
        return False
    return i < 0 or not buf[i].isalnum()


# --------------------------------------------------------------------------- number check


@dataclass(frozen=True)
class Pass:
    pass


NumberBlockReason = Literal["unquoted_amount", "unquoted_price", "unknown_number"]
NumberExtractor = Callable[[str], Iterable[Found]]


@dataclass(frozen=True)
class Blocked:
    reason: NumberBlockReason
    blocked_amount: int
    blocked_price_context: int
    blocked_plain: int
    check: Literal["numbers"] = "numbers"

    def __repr__(self) -> str:
        return (
            f"Blocked(check={self.check!r}, reason={self.reason!r}, "
            f"blocked_amount={self.blocked_amount}, "
            f"blocked_price_context={self.blocked_price_context}, "
            f"blocked_plain={self.blocked_plain})"
        )

    def __str__(self) -> str:
        return repr(self)


def check_numbers(
    sentence: str,
    *,
    allowed_amounts: Set[Decimal],
    allowed_numbers: Set[Decimal],
    extract: NumberExtractor,
) -> Pass | Blocked:
    """Block unquoted amounts, prices, and unknown plains. Result holds counts only."""
    n_amount = 0
    n_price = 0
    n_plain = 0
    for found in extract(sentence):
        if found.kind in ("amount", "price_context"):
            if found.value not in allowed_amounts:
                if found.kind == "amount":
                    n_amount += 1
                else:
                    n_price += 1
        elif found.value not in allowed_numbers and not _small_plain(found.value):
            n_plain += 1
    if n_amount == 0 and n_price == 0 and n_plain == 0:
        return Pass()
    if n_amount:
        reason: NumberBlockReason = "unquoted_amount"
    elif n_price:
        reason = "unquoted_price"
    else:
        reason = "unknown_number"
    return Blocked(
        reason=reason,
        blocked_amount=n_amount,
        blocked_price_context=n_price,
        blocked_plain=n_plain,
    )


def _small_plain(value: Decimal) -> bool:
    if value != value.to_integral_value():
        return False
    return Decimal(0) <= value <= Decimal(12)
