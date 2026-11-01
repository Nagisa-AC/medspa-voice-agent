from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from demo import SYNTHETIC_CALLER, make_clock, parse_args


def test_flags_and_synthetic_caller() -> None:
    args = parse_args(["--sunday", "--no-caller-id"])
    assert args.sunday is True
    assert args.no_caller_id is True
    assert SYNTHETIC_CALLER.startswith("+1")
    assert "555" in SYNTHETIC_CALLER


def test_sunday_clock_advances_from_fixed_instant() -> None:
    now = 10.0

    def mono() -> float:
        return now

    clock = make_clock(sunday=True, tz=ZoneInfo("America/New_York"), mono=mono)
    first = clock()
    now = 12.5
    second = clock()
    assert second - first == timedelta(seconds=2.5)
    assert first.weekday() == 6


def test_demo_does_not_import_retell() -> None:
    tree = ast.parse(Path("demo.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "retell" not in imported
    assert "tests" not in imported and not any(name.startswith("tests.") for name in imported)
