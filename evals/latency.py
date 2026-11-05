"""Print p50/p95/max from the latency sidecar. `python -m evals.latency`."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

HEADER = (
    "Server-side timings (TurnLoop entry → first chunk). "
    "Excludes Retell endpointing, network, and TTS."
)
_TYPES = ("backend", "model", "tools")
_METRICS = ("t_first_chunk_ms", "t_model_first_token_ms", "t_tool_ms", "t_total_ms")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "data" / "latency.jsonl"


def percentile(values: Sequence[int], p: float) -> int:
    if not values:
        raise ValueError("percentile of empty sample")
    ordered = sorted(values)
    index = math.ceil(p / 100 * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def load_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        loaded = json.loads(raw)
        if not isinstance(loaded, dict) or loaded.get("kind") != "turn_latency":
            continue
        payload = loaded.get("payload")
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def _ints(rows: Sequence[dict[str, object]], field: str) -> list[int]:
    found: list[int] = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        found.append(value)
    return found


def _cell(values: Sequence[int]) -> str:
    if not values:
        return "—"
    p50 = percentile(values, 50)
    p95 = percentile(values, 95)
    return f"{p50}/{p95}/{max(values)}"


def render(rows: Sequence[dict[str, object]]) -> str:
    lines = [HEADER, ""]
    header = (
        f"{'turn_type':<10} {'n':>4}  "
        f"{'first_chunk p50/p95/max':<24} "
        f"{'first_token p50/p95/max':<24} "
        f"{'tool p50/p95/max':<20} "
        f"{'total p50/p95/max'}"
    )
    lines.append(header)
    groups: list[tuple[str, Sequence[dict[str, object]]]] = [("all", rows)]
    for name in _TYPES:
        groups.append((name, [row for row in rows if row.get("turn_type") == name]))
    for name, group in groups:
        cells = [_cell(_ints(group, field)) for field in _METRICS]
        lines.append(
            f"{name:<10} {len(group):>4}  {cells[0]:<24} {cells[1]:<24} {cells[2]:<20} {cells[3]}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=HEADER)
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_LOG),
        help="latency jsonl (default: data/latency.jsonl)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    stream = out if out is not None else sys.stdout
    stream.write(render(load_rows(Path(args.path))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
