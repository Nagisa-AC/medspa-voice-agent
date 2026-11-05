from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from app import LatencyTap
from core.crypto import InMemoryKeyStore
from core.events import EventDraft, InMemoryEventLog
from evals.latency import load_rows, main, percentile, render

NOW = datetime(2026, 11, 5, 16, 0, tzinfo=UTC)


def test_percentile_nearest_rank() -> None:
    values = [10, 20, 30, 40, 100]
    assert percentile(values, 50) == 30
    assert percentile(values, 95) == 100
    assert percentile([7], 50) == 7
    assert percentile([7], 95) == 7


def test_render_header_is_server_side() -> None:
    text = render([])
    assert "Server-side timings" in text
    assert "TurnLoop entry" in text
    assert "endpointing" in text
    assert "TTS" in text
    assert "all" in text
    assert "backend" in text
    assert "n=0" in text or "   0" in text


def test_cli_reads_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "latency.jsonl"
    rows = [
        {
            "call_id": "c1",
            "tenant_id": "glow",
            "at": NOW.isoformat(),
            "kind": "turn_latency",
            "payload": {
                "response_id": "1",
                "turn_type": "backend",
                "t_first_chunk_ms": 10,
                "t_total_ms": 12,
            },
        },
        {
            "call_id": "c1",
            "tenant_id": "glow",
            "at": NOW.isoformat(),
            "kind": "turn_latency",
            "payload": {
                "response_id": "2",
                "turn_type": "model",
                "t_first_chunk_ms": 100,
                "t_total_ms": 120,
                "t_model_first_token_ms": 80,
            },
        },
        {
            "call_id": "c2",
            "tenant_id": "glow",
            "at": NOW.isoformat(),
            "kind": "caller_utterance",
            "payload": {"text": "secret words", "turn": 1},
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_rows(path)
    assert len(loaded) == 2
    out = StringIO()
    assert main([str(path)], out=out) == 0
    printed = out.getvalue()
    assert "secret words" not in printed
    assert "backend" in printed
    assert "model" in printed
    assert "10/10/10" in printed


async def test_latency_tap_writes_ids_only(tmp_path: Path) -> None:
    keys = InMemoryKeyStore()
    await keys.create("call_1")
    path = tmp_path / "latency.jsonl"
    log = LatencyTap(InMemoryEventLog(keys), path)
    await log.record_caller_utterance(
        call_id="call_1",
        tenant_id="glow",
        at=NOW,
        turn=1,
        text="I am pregnant",
        clinical_class=True,
        trip="clinical_question",
    )
    await log.append(
        EventDraft(
            call_id="call_1",
            tenant_id="glow",
            at=NOW,
            kind="turn_latency",
            payload={
                "response_id": "1",
                "turn_type": "backend",
                "t_first_chunk_ms": 5,
                "t_total_ms": 8,
            },
        )
    )
    raw = path.read_text(encoding="utf-8")
    assert "pregnant" not in raw
    assert "I am" not in raw
    assert '"kind":"turn_latency"' in raw
    assert '"call_id":"call_1"' in raw
    assert raw.count("\n") == 1


def test_cli_missing_file_is_empty(tmp_path: Path) -> None:
    out = StringIO()
    assert main([str(tmp_path / "missing.jsonl")], out=out) == 0
    assert "Server-side timings" in out.getvalue()
