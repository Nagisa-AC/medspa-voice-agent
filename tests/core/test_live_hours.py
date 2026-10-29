from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from core.tenant import LiveHours, Window

NY = ZoneInfo("America/New_York")
GLOW_HOURS = "Mon-Fri 09:00-18:00, Sat 10:00-16:00"


def utc(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def local(
    tz: ZoneInfo, y: int, mo: int, d: int, h: int, mi: int, s: int = 0, *, fold: int = 0
) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=tz, fold=fold)


@pytest.mark.parametrize(
    ("text", "windows"),
    [
        ("Mon 09:00-17:00", [(0, 540, 1020)]),
        ("mon-tue 09:00 - 10:00", [(0, 540, 600), (1, 540, 600)]),
        ("Sat-Mon 10:00-11:00", [(0, 600, 660), (5, 600, 660), (6, 600, 660)]),
        ("Mon 09:00-12:00, Mon 13:00-17:00", [(0, 540, 720), (0, 780, 1020)]),
        ("Sun 00:00-24:00", [(6, 0, 1440)]),
        ("Mon 09:00-10:00,Mon 10:00-11:00", [(0, 540, 600), (0, 600, 660)]),
    ],
)
def test_parse_valid(text: str, windows: list[tuple[int, int, int]]) -> None:
    parsed = LiveHours.parse(text)
    assert parsed.windows == tuple(Window(*w) for w in windows)
    assert parsed.text == text


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "empty"),
        ("Mon-Fri", "cannot parse"),
        ("Thur 09:00-10:00", "unknown day"),
        ("Mon 9:00-17:00", "HH:MM"),
        ("Mon 9am-5pm", "HH:MM"),
        ("Mon 09:00-25:00", "out of range"),
        ("Mon 24:00-24:00", "out of range"),
        ("Mon 09:60-10:00", "out of range"),
        ("Mon 09:00-09:00", "empty window"),
        ("Fri 22:00-02:00", "crosses midnight"),
        ("Mon 09:00-12:00, Mon 11:00-13:00", "overlaps"),
        ("Mon-Fri 09:00-17:00, Wed 16:00-18:00", "overlaps"),
        ("Mon 09:00-10:00,", "cannot parse"),
    ],
)
def test_parse_invalid(text: str, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        LiveHours.parse(text)


def test_24_00_is_end_only() -> None:
    with pytest.raises(ValueError):
        LiveHours.parse("Mon 24:00-23:00")


def test_escalation_times_in_tenant_zone() -> None:
    hours = LiveHours.parse(GLOW_HOURS)
    assert hours.is_open(local(NY, 2026, 11, 3, 11, 0), NY)  # Tuesday 11:00
    assert not hours.is_open(local(NY, 2026, 11, 8, 14, 0), NY)  # Sunday 14:00


def test_end_is_exclusive() -> None:
    hours = LiveHours.parse(GLOW_HOURS)
    assert hours.is_open(local(NY, 2026, 11, 3, 17, 59, 59), NY)
    assert not hours.is_open(local(NY, 2026, 11, 3, 18, 0), NY)
    assert hours.is_open(local(NY, 2026, 11, 3, 9, 0), NY)
    assert not hours.is_open(local(NY, 2026, 11, 3, 8, 59), NY)


def test_same_utc_time_flips_across_spring_dst_change() -> None:
    # US DST starts Sunday 2026-03-08. 13:30 UTC is 08:30 EST before, 09:30 EDT after.
    hours = LiveHours.parse(GLOW_HOURS)
    assert not hours.is_open(utc(2026, 3, 6, 13, 30), NY)  # Friday
    assert hours.is_open(utc(2026, 3, 9, 13, 30), NY)  # Monday


def test_same_utc_time_flips_across_autumn_dst_change() -> None:
    # US DST ends Sunday 2026-11-01. 13:30 UTC is 09:30 EDT before, 08:30 EST after.
    hours = LiveHours.parse(GLOW_HOURS)
    assert hours.is_open(utc(2026, 10, 30, 13, 30), NY)  # Friday
    assert not hours.is_open(utc(2026, 11, 2, 13, 30), NY)  # Monday


def test_repeated_hour_counts_both_times() -> None:
    hours = LiveHours.parse("Sun 01:00-02:00")
    first = local(NY, 2026, 11, 1, 1, 30, fold=0)
    second = local(NY, 2026, 11, 1, 1, 30, fold=1)
    assert first.utcoffset() != second.utcoffset()
    assert hours.is_open(first, NY)
    assert hours.is_open(second, NY)


def test_skipped_hour_is_never_open() -> None:
    hours = LiveHours.parse("Sun 02:00-03:00")
    assert not hours.is_open(utc(2026, 3, 8, 6, 59), NY)  # 01:59 EST
    assert not hours.is_open(utc(2026, 3, 8, 7, 0), NY)  # 03:00 EDT


def test_uses_tenant_zone_not_server_zone() -> None:
    hours = LiveHours.parse("Mon 09:00-17:00")
    now = utc(2026, 11, 2, 1, 0)  # Mon 10:00 in Tokyo, Sun 20:00 in New York
    assert hours.is_open(now, ZoneInfo("Asia/Tokyo"))
    assert not hours.is_open(now, NY)


def test_naive_time_is_refused() -> None:
    hours = LiveHours.parse(GLOW_HOURS)
    with pytest.raises(ValueError, match="timezone-aware"):
        hours.is_open(datetime(2026, 11, 3, 11, 0), NY)  # noqa: DTZ001
