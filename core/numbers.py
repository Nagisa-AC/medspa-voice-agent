"""Number extraction for the turn-loop check. Tokenizer plus ordered rules; no float."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

NumberKind = Literal["amount", "price_context", "plain"]
TokenKind = Literal["word", "digits", "money", "punct", "ordinal"]
_TOKEN_KINDS: dict[str, TokenKind] = {
    "word": "word",
    "digits": "digits",
    "money": "money",
    "punct": "punct",
    "ordinal": "ordinal",
}

_EN_DASH = "\u2013"
_EM_DASH = "\u2014"
_TOKEN_RE = re.compile(
    r"""
    (?P<money>\$)
    |(?P<ordinal>\d+(?:st|nd|rd|th))
    |(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)
    |(?P<word>[A-Za-z]+(?:'[A-Za-z]+)?)
    |(?P<punct>[:.,+\-()"""
    + _EN_DASH
    + _EM_DASH
    + r"""])
    """,
    re.VERBOSE,
)

_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_TEENS: dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_HOUR_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_ORDINAL_ONES: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
}
_ORDINAL_WORDS: dict[str, int] = {
    **_ORDINAL_ONES,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_DOLLAR_WORDS = frozenset({"dollar", "dollars", "buck", "bucks"})
_CENT_WORDS = frozenset({"cent", "cents"})
_CURRENCY_WORDS = _DOLLAR_WORDS | _CENT_WORDS
_UNIT_WORDS = frozenset(
    {"unit", "syringe", "vial", "session", "area", "treatment", "shot", "injection"}
)
_UNIT_PREFIX = frozenset({"a", "an", "per", "each", "every"})
_EACH_AFTER = frozenset({"each", "apiece"})
_CUE_WORDS = frozenset(
    {"cost", "costs", "run", "runs", "charge", "charges", "pay", "price", "prices"}
)
_CUE_PHRASES: tuple[tuple[str, ...], ...] = (
    ("starts", "at"),
    ("starting", "at"),
    ("goes", "for"),
    ("is", "about"),
    ("is", "around"),
)
_PHONE_PUNCT = frozenset({"+", "-", _EN_DASH, _EM_DASH, ".", "(", ")"})
_HYPHENS = frozenset({"-", _EN_DASH, _EM_DASH})
_PRICE_WINDOW = 3
_CURRENCY_AFTER = 2


@dataclass(frozen=True)
class Found:
    value: Decimal
    kind: NumberKind
    span: tuple[int, int]


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    start: int
    end: int
    norm: str
    number: Decimal | None


@dataclass(frozen=True)
class _Hit:
    value: Decimal
    start: int
    end: int
    tok_lo: int
    tok_hi: int
    marked_amount: bool


def find_numbers(text: str, *, extra_price_words: Iterable[str] = ()) -> list[Found]:
    """Return every number in `text`, left to right. Bind extra_price_words at the call site."""
    tokens = tokenize(text)
    extras = tuple(p for p in extra_price_words if p.strip())
    hits: list[_Hit] = []
    i = 0
    while i < len(tokens):
        matched = _first_rule(tokens, i)
        if matched is None:
            i += 1
            continue
        batch, nxt = matched
        hits.extend(batch)
        i = nxt
    return _classify(tokens, hits, extras)


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        kind = _TOKEN_KINDS.get(match.lastgroup or "")
        if kind is None:
            continue
        raw = match.group()
        start, end = match.span()
        number: Decimal | None = None
        if kind == "digits":
            number = _decimal_from_digits(raw)
        elif kind == "ordinal":
            number = Decimal(raw[:-2])
        tokens.append(
            Token(
                kind=kind,
                text=raw,
                start=start,
                end=end,
                norm=raw.casefold(),
                number=number,
            )
        )
    return tokens


def integer_to_words(n: int) -> str:
    """English words for 0..999_999, using the same tables the parser reads."""
    if n < 0 or n > 999_999:
        raise ValueError(f"integer_to_words out of range: {n}")
    if n < 20:
        return _small_word(n)
    if n < 100:
        tens, ones = divmod(n, 10)
        head = next(word for word, value in _TENS.items() if value == tens * 10)
        return head if ones == 0 else f"{head}-{_small_word(ones)}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = f"{_small_word(hundreds)} hundred"
        return head if rest == 0 else f"{head} {integer_to_words(rest)}"
    thousands, rest = divmod(n, 1000)
    head = f"{integer_to_words(thousands)} thousand"
    return head if rest == 0 else f"{head} {integer_to_words(rest)}"


def _small_word(n: int) -> str:
    if n < 10:
        return next(word for word, value in _ONES.items() if value == n)
    return next(word for word, value in _TEENS.items() if value == n)


def _decimal_from_digits(text: str) -> Decimal:
    return Decimal(text.replace(",", ""))


def _as_int(value: Decimal | None) -> int | None:
    if value is None or value != value.to_integral_value():
        return None
    return int(value)


def _is_hyphen(token: Token) -> bool:
    return token.kind == "punct" and token.norm in _HYPHENS


def _skip_hyphen(tokens: list[Token], i: int) -> int:
    if i < len(tokens) and _is_hyphen(tokens[i]):
        return i + 1
    return i


def _is_range_conn(tokens: list[Token], i: int) -> bool:
    if i >= len(tokens):
        return False
    token = tokens[i]
    if token.kind == "word" and token.norm == "to":
        return True
    return _is_hyphen(token)


def _first_rule(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    for rule in (
        _try_time,
        _try_dollar_prefix,
        _try_currency_words,
        _try_spoken_fifty,
        _try_ordinal,
        _try_word_number,
        _try_phone,
        _try_digit_range,
        _try_bare_digits,
    ):
        hit = rule(tokens, i)
        if hit is not None:
            return hit
    return None


def _hit(
    value: Decimal,
    tokens: list[Token],
    lo: int,
    hi: int,
    *,
    marked_amount: bool,
    start: int | None = None,
    end: int | None = None,
) -> _Hit:
    return _Hit(
        value=value,
        start=tokens[lo].start if start is None else start,
        end=tokens[hi - 1].end if end is None else end,
        tok_lo=lo,
        tok_hi=hi,
        marked_amount=marked_amount,
    )


# --------------------------------------------------------------------------- word integers


def _parse_0_99(tokens: list[Token], i: int) -> tuple[int, int] | None:
    if i >= len(tokens) or tokens[i].kind != "word":
        return None
    word = tokens[i].norm
    if word in _TEENS:
        return _TEENS[word], i + 1
    if word in _ONES:
        return _ONES[word], i + 1
    if word in _TENS:
        j = _skip_hyphen(tokens, i + 1)
        if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm in _ONES:
            return _TENS[word] + _ONES[tokens[j].norm], j + 1
        return _TENS[word], i + 1
    return None


def _optional_and(tokens: list[Token], i: int) -> int:
    if i < len(tokens) and tokens[i].kind == "word" and tokens[i].norm == "and":
        return i + 1
    return i


def _parse_below_thousand(tokens: list[Token], i: int) -> tuple[int, int] | None:
    small = _parse_0_99(tokens, i)
    if small is None:
        return None
    n, j = small
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm == "hundred":
        k = _optional_and(tokens, j + 1)
        rest = _parse_0_99(tokens, k)
        if rest is not None:
            return n * 100 + rest[0], rest[1]
        return n * 100, j + 1
    return n, j


def _rest_after_thousand(tokens: list[Token], i: int) -> tuple[int, int]:
    k = _optional_and(tokens, i)
    rest = _parse_below_thousand(tokens, k)
    if rest is None:
        return 0, i
    return rest[0], rest[1]


def _parse_word_int(tokens: list[Token], i: int) -> tuple[int, int] | None:
    if i >= len(tokens) or tokens[i].kind != "word":
        return None
    if tokens[i].norm in {"a", "an"} and i + 1 < len(tokens) and tokens[i + 1].kind == "word":
        scale = tokens[i + 1].norm
        if scale == "thousand":
            rest, k = _rest_after_thousand(tokens, i + 2)
            return 1000 + rest, k
        if scale == "hundred":
            k = _optional_and(tokens, i + 2)
            ones = _parse_0_99(tokens, k)
            if ones is not None:
                return 100 + ones[0], ones[1]
            return 100, i + 2
        return None
    below = _parse_below_thousand(tokens, i)
    if below is None:
        return None
    n, j = below
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm == "thousand":
        rest, k = _rest_after_thousand(tokens, j + 1)
        return n * 1000 + rest, k
    return n, j


def _parse_numeric(tokens: list[Token], i: int) -> tuple[Decimal, int] | None:
    if i >= len(tokens):
        return None
    number = tokens[i].number
    if tokens[i].kind == "digits" and number is not None:
        return number, i + 1
    parsed = _parse_word_int(tokens, i)
    if parsed is None:
        return None
    return Decimal(parsed[0]), parsed[1]


# --------------------------------------------------------------------------- times


def _am_pm(tokens: list[Token], i: int) -> int | None:
    if i >= len(tokens) or tokens[i].kind != "word":
        return None
    if tokens[i].norm in {"am", "pm"}:
        return i + 1
    if tokens[i].norm not in {"a", "p"}:
        return None
    j = i + 1
    if j < len(tokens) and tokens[j].kind == "punct" and tokens[j].norm == ".":
        j += 1
    if j >= len(tokens) or tokens[j].kind != "word" or tokens[j].norm != "m":
        return None
    j += 1
    if j < len(tokens) and tokens[j].kind == "punct" and tokens[j].norm == ".":
        j += 1
    return j


def _parse_hour(tokens: list[Token], i: int) -> tuple[int, int] | None:
    token = tokens[i]
    if token.kind == "digits":
        hour = _as_int(token.number)
        if hour is not None and 0 <= hour <= 23:
            return hour, i + 1
        return None
    if token.kind == "word" and token.norm in _HOUR_WORDS:
        return _HOUR_WORDS[token.norm], i + 1
    return None


def _try_word_minutes(tokens: list[Token], i: int) -> tuple[int, int] | None:
    parsed = _parse_0_99(tokens, i)
    if parsed is None:
        return None
    n, j = parsed
    if n == 50 and j == i + 1:
        return None
    if 0 <= n <= 59:
        return n, j
    return None


def _extend_am_pm(tokens: list[Token], k: int) -> int:
    later = _am_pm(tokens, k)
    return k if later is None else later


def _try_time(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    hour = _parse_hour(tokens, i)
    if hour is None:
        return None
    h, j = hour
    if j < len(tokens) and tokens[j].kind == "punct" and tokens[j].norm == ":":
        if j + 1 < len(tokens) and tokens[j + 1].kind == "digits":
            minute = _as_int(tokens[j + 1].number)
            if minute is not None and 0 <= minute <= 59:
                k = _extend_am_pm(tokens, j + 2)
                return _time_hits(h, minute, tokens, i, j + 1, k), k
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm in {"o'clock", "oclock"}:
        k = _extend_am_pm(tokens, j + 1)
        return _time_hits(h, 0, tokens, i, j, k), k
    minutes = _try_word_minutes(tokens, j)
    if minutes is not None:
        m, k = minutes
        k = _extend_am_pm(tokens, k)
        return _time_hits(h, m, tokens, i, j, k), k
    ampm = _am_pm(tokens, j)
    if ampm is not None:
        return [_hit(Decimal(h), tokens, i, ampm, marked_amount=False)], ampm
    return None


def _time_hits(
    hour: int,
    minute: int,
    tokens: list[Token],
    hour_i: int,
    minute_i: int,
    end_i: int,
) -> list[_Hit]:
    hour_hit = _hit(Decimal(hour), tokens, hour_i, hour_i + 1, marked_amount=False)
    minute_hit = _hit(Decimal(minute), tokens, minute_i, minute_i + 1, marked_amount=False)
    if end_i > minute_i + 1:
        minute_hit = _hit(
            Decimal(minute),
            tokens,
            minute_i,
            end_i,
            marked_amount=False,
            start=tokens[minute_i].start,
            end=tokens[end_i - 1].end,
        )
    return [hour_hit, minute_hit]


# --------------------------------------------------------------------------- currency


def _try_a_dollar_forms(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    j = i
    had_article = False
    if tokens[i].kind == "word" and tokens[i].norm in {"a", "an"}:
        had_article = True
        j = i + 1
    if j >= len(tokens) or tokens[j].kind != "word" or tokens[j].norm not in _DOLLAR_WORDS:
        return None
    k = _skip_hyphen(tokens, j + 1)
    if k < len(tokens) and tokens[k].kind == "word" and tokens[k].norm == "fifty":
        return [_hit(Decimal("1.50"), tokens, i, k + 1, marked_amount=True)], k + 1
    if had_article:
        return [_hit(Decimal(1), tokens, i, j + 1, marked_amount=True)], j + 1
    return None


def _cents_after_dollars(tokens: list[Token], j: int, val: Decimal) -> tuple[Decimal, int]:
    k = j
    if k < len(tokens) and tokens[k].kind == "word" and tokens[k].norm == "and":
        k += 1
    cents = _parse_numeric(tokens, k)
    if cents is None:
        return val, j
    amount, m = cents
    if m < len(tokens) and tokens[m].kind == "word" and tokens[m].norm in _CENT_WORDS:
        return val + amount / Decimal(100), m + 1
    return val, j


def _try_dollar_prefix(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    if tokens[i].kind != "money":
        return None
    left = _parse_numeric(tokens, i + 1)
    if left is None:
        return None
    val1, j = left
    if _is_range_conn(tokens, j):
        k = j + 1
        if k < len(tokens) and tokens[k].kind == "money":
            k += 1
        right = _parse_numeric(tokens, k)
        if right is not None:
            val2, m = right
            return [
                _hit(val1, tokens, i, m, marked_amount=True, end=tokens[j - 1].end),
                _hit(val2, tokens, k, m, marked_amount=True),
            ], m
    return [_hit(val1, tokens, i, j, marked_amount=True)], j


def _try_currency_words(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    special = _try_a_dollar_forms(tokens, i)
    if special is not None:
        return special
    left = _parse_numeric(tokens, i)
    if left is None:
        return None
    val1, j = left
    if _is_range_conn(tokens, j):
        k = j + 1
        if k < len(tokens) and tokens[k].kind == "money":
            k += 1
        right = _parse_numeric(tokens, k)
        if right is not None:
            val2, m = right
            if m < len(tokens) and tokens[m].kind == "word" and tokens[m].norm in _DOLLAR_WORDS:
                return [
                    _hit(val1, tokens, i, j, marked_amount=True),
                    _hit(val2, tokens, k, m + 1, marked_amount=True),
                ], m + 1
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm in _DOLLAR_WORDS:
        val, m = _cents_after_dollars(tokens, j + 1, val1)
        return [_hit(val, tokens, i, m, marked_amount=True)], m
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm in _CENT_WORDS:
        return [_hit(val1 / Decimal(100), tokens, i, j + 1, marked_amount=True)], j + 1
    return None


# --------------------------------------------------------------------------- spoken prices


def _try_spoken_fifty(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    n: int | None = None
    j = i
    small = _parse_0_99(tokens, i)
    if small is not None:
        n, j = small
    elif tokens[i].kind == "digits":
        parsed = _as_int(tokens[i].number)
        if parsed is not None and 0 <= parsed <= 99:
            n = parsed
            j = i + 1
    if n is None:
        return None
    j = _skip_hyphen(tokens, j)
    if j >= len(tokens) or tokens[j].kind != "word" or tokens[j].norm != "fifty":
        return None
    k = j + 1
    cents = Decimal(n) + Decimal("0.50")
    hundreds = Decimal(n * 100 + 50)
    return [
        _hit(cents, tokens, i, k, marked_amount=False),
        _hit(hundreds, tokens, i, k, marked_amount=False),
    ], k


def _try_word_number(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    parsed = _parse_word_int(tokens, i)
    if parsed is None:
        return None
    n, j = parsed
    return [_hit(Decimal(n), tokens, i, j, marked_amount=False)], j


def _try_ordinal(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    token = tokens[i]
    if token.kind == "ordinal" and token.number is not None:
        return [_hit(token.number, tokens, i, i + 1, marked_amount=False)], i + 1
    if token.kind != "word":
        return None
    if token.norm in _ORDINAL_WORDS:
        return [_hit(Decimal(_ORDINAL_WORDS[token.norm]), tokens, i, i + 1, marked_amount=False)], (
            i + 1
        )
    tens = _TENS.get(token.norm)
    if tens is None:
        return None
    j = _skip_hyphen(tokens, i + 1)
    if j < len(tokens) and tokens[j].kind == "word" and tokens[j].norm in _ORDINAL_ONES:
        value = Decimal(tens + _ORDINAL_ONES[tokens[j].norm])
        return [_hit(value, tokens, i, j + 1, marked_amount=False)], j + 1
    return None


# --------------------------------------------------------------------------- phones, ranges, digits


def _try_phone(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    j = i
    if tokens[j].kind == "punct" and tokens[j].norm == "+":
        j += 1
    if j >= len(tokens) or tokens[j].kind != "digits":
        return None
    groups: list[tuple[Decimal, int, int]] = []
    digit_count = 0
    k = j
    last_digit_hi = j
    while k < len(tokens):
        token = tokens[k]
        if token.kind == "digits":
            if "," in token.text or "." in token.text:
                break
            n_digits = sum(ch.isdigit() for ch in token.text)
            digit_count += n_digits
            if token.number is not None:
                groups.append((token.number, k, k + 1))
            last_digit_hi = k + 1
            k += 1
            continue
        if token.kind == "punct" and token.norm in _PHONE_PUNCT:
            k += 1
            continue
        break
    if not 7 <= digit_count <= 15 or not groups:
        return None
    raw_digits = "".join(
        tokens[g_lo].text for _, g_lo, _ in groups if tokens[g_lo].kind == "digits"
    )
    full = Decimal(raw_digits)
    hits = [_hit(full, tokens, i, last_digit_hi, marked_amount=False)]
    if len(groups) > 1:
        for value, g_lo, g_hi in groups:
            if _as_int(value) is not None and abs(int(value)) >= 10:
                hits.append(_hit(value, tokens, g_lo, g_hi, marked_amount=False))
    return hits, last_digit_hi


def _try_digit_range(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    if tokens[i].kind != "digits" or tokens[i].number is None:
        return None
    if not _is_range_conn(tokens, i + 1):
        return None
    k = i + 2
    if k >= len(tokens) or tokens[k].kind != "digits" or tokens[k].number is None:
        return None
    left = tokens[i].number
    right = tokens[k].number
    if left is None or right is None:
        return None
    return [
        _hit(left, tokens, i, i + 1, marked_amount=False),
        _hit(right, tokens, k, k + 1, marked_amount=False),
    ], k + 1


def _try_bare_digits(tokens: list[Token], i: int) -> tuple[list[_Hit], int] | None:
    number = tokens[i].number
    if tokens[i].kind != "digits" or number is None:
        return None
    return [_hit(number, tokens, i, i + 1, marked_amount=False)], i + 1


# --------------------------------------------------------------------------- kind


def _token_distance(idx: int, lo: int, hi: int) -> int:
    if idx < lo:
        return lo - idx
    if idx >= hi:
        return idx - (hi - 1)
    return 0


def _has_unit_price(tokens: list[Token], lo: int, hi: int) -> bool:
    left = max(0, lo - _PRICE_WINDOW)
    right = min(len(tokens), hi + _PRICE_WINDOW)
    for idx in range(left, right):
        if tokens[idx].kind != "word" or tokens[idx].norm not in _UNIT_WORDS:
            continue
        if idx == 0:
            continue
        prev = tokens[idx - 1]
        if prev.kind != "word" or prev.norm not in _UNIT_PREFIX:
            continue
        if _token_distance(idx, lo, hi) <= _PRICE_WINDOW:
            return True
    return False


def _has_left_cue(tokens: list[Token], lo: int) -> bool:
    left = tokens[max(0, lo - _PRICE_WINDOW) : lo]
    norms = [token.norm for token in left]
    for i, norm in enumerate(norms):
        if norm in _CUE_WORDS:
            return True
        for phrase in _CUE_PHRASES:
            if tuple(norms[i : i + len(phrase)]) == phrase:
                return True
    return False


def _window_has_phrase(tokens: list[Token], lo: int, hi: int, phrase: str) -> bool:
    parts = [token.norm for token in tokenize(phrase)]
    if not parts:
        return False
    left = max(0, lo - _PRICE_WINDOW)
    right = min(len(tokens), hi + _PRICE_WINDOW)
    norms = [token.norm for token in tokens[left:right]]
    length = len(parts)
    return any(norms[i : i + length] == parts for i in range(len(norms) - length + 1))


def _has_each_or_apiece(tokens: list[Token], hi: int) -> bool:
    if hi >= len(tokens):
        return False
    token = tokens[hi]
    return token.kind == "word" and token.norm in _EACH_AFTER


def _has_per_word(tokens: list[Token], lo: int, hi: int) -> bool:
    left = max(0, lo - _PRICE_WINDOW)
    right = min(len(tokens), hi + _PRICE_WINDOW)
    for idx in range(left, right):
        if tokens[idx].kind != "word" or tokens[idx].norm != "per":
            continue
        nxt = idx + 1
        if nxt >= len(tokens) or tokens[nxt].kind != "word":
            continue
        if _token_distance(idx, lo, hi) <= _PRICE_WINDOW:
            return True
    return False


def _has_currency_after(tokens: list[Token], hit: _Hit) -> bool:
    stop = min(len(tokens), hit.tok_hi + _CURRENCY_AFTER)
    for idx in range(hit.tok_hi, stop):
        token = tokens[idx]
        if token.kind == "word" and token.norm in _CURRENCY_WORDS:
            return True
    return False


def _in_price_window(tokens: list[Token], hit: _Hit, extras: tuple[str, ...]) -> bool:
    if _has_unit_price(tokens, hit.tok_lo, hit.tok_hi):
        return True
    if _has_each_or_apiece(tokens, hit.tok_hi):
        return True
    if _has_per_word(tokens, hit.tok_lo, hit.tok_hi):
        return True
    if _has_left_cue(tokens, hit.tok_lo):
        return True
    return any(_window_has_phrase(tokens, hit.tok_lo, hit.tok_hi, phrase) for phrase in extras)


def _classify(tokens: list[Token], hits: list[_Hit], extras: tuple[str, ...]) -> list[Found]:
    found: list[Found] = []
    for hit in hits:
        if hit.marked_amount or _has_currency_after(tokens, hit):
            kind: NumberKind = "amount"
        elif _in_price_window(tokens, hit, extras):
            kind = "price_context"
        else:
            kind = "plain"
        found.append(Found(value=hit.value, kind=kind, span=(hit.start, hit.end)))
    found.sort(key=lambda item: (item.span[0], item.span[1], item.value))
    return found
