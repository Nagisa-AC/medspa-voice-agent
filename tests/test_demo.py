from __future__ import annotations

import ast
import os
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from demo import SYNTHETIC_CALLER, load_dotenv, make_clock, parse_args


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


def test_dotenv_loads_known_keys_without_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_MODEL", "already-set")
    path = tmp_path / ".env"
    path.write_text(
        "ANTHROPIC_API_KEY=sk-test-not-a-real-key\nANTHROPIC_MODEL=should-not-win\nUNKNOWN=nope\n",
        encoding="utf-8",
    )
    load_dotenv(path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test-not-a-real-key"
    assert os.environ["ANTHROPIC_MODEL"] == "already-set"
    assert "UNKNOWN" not in os.environ
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_env_example_has_no_secret() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=" in text
    assert "sk-" not in text


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
