from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator
from functools import partial
from pathlib import Path

import pytest
import yaml

import evals.harness
from core.crypto import InMemoryKeyStore
from core.events import EventDraft, InMemoryEventLog
from core.guardrail import Blocked, check_clinical, load_output_pack
from core.llm import ScriptedLLM
from core.numbers import find_numbers
from core.policy import load_input_pack
from core.state import CallerLine, CallState, Committed, Transfer
from core.tenant import extra_price_words, load_tenant_file
from core.turn import OutputChunk, TurnLoop
from evals.harness import (
    CaseFailure,
    CaseLoadError,
    Report,
    _clinical_leak,
    check_model_order,
    format_report,
    load_case_path,
    load_cases,
    run_case,
)
from tests.core.test_policy import PATTERN_CASES

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures"
TODAY = __import__("datetime").date(2026, 10, 29)
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
PACK = load_output_pack(ROOT / "packs" / "medspa.output.yaml")
_WORDS = extra_price_words(GLOW)
EXTRACT = partial(
    find_numbers,
    extra_price_words=_WORDS.extra_price_words,
    service_words=_WORDS.service_words,
)


def test_loader_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "tenant: glow\nclock: '2026-11-03T11:00:00-05:00'\n"
        "utterances: ['hi']\nscript_path: scripts/unused.yaml\nnope: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError, match="unknown fields") as info:
        load_case_path(path)
    assert "hi" not in str(info.value)


def test_loader_rejects_naive_clock(tmp_path: Path) -> None:
    path = tmp_path / "naive.yaml"
    path.write_text(
        "tenant: glow\nclock: '2026-11-03T11:00:00'\n"
        "utterances: ['hi']\nscript_path: scripts/unused.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError, match="timezone-aware"):
        load_case_path(path)


def test_loader_rejects_naive_clock_at(tmp_path: Path) -> None:
    path = tmp_path / "naive_at.yaml"
    path.write_text(
        "tenant: glow\nclock: '2026-11-03T11:00:00-05:00'\n"
        "clock_at: {1: '2026-11-03T11:07:00'}\n"
        "utterances: ['hi']\nscript_path: scripts/unused.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError, match="timezone-aware"):
        load_case_path(path)


def test_loader_rejects_unknown_tenant(tmp_path: Path) -> None:
    path = tmp_path / "tenant.yaml"
    path.write_text(
        "tenant: missing\nclock: '2026-11-03T11:00:00-05:00'\n"
        "utterances: ['hi']\nscript_path: scripts/unused.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError, match="tenant"):
        load_case_path(path)


def test_loader_rejects_empty_utterances(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text(
        "tenant: glow\nclock: '2026-11-03T11:00:00-05:00'\n"
        "utterances: []\nscript_path: scripts/unused.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(CaseLoadError, match="utterances"):
        load_case_path(path)


def test_every_pack_pattern_has_a_case() -> None:
    expected: set[tuple[str, str]] = set()
    for name in ("universal.yaml", "medspa.yaml"):
        pack = load_input_pack(ROOT / "packs" / name)
        for trip in pack.tripwires:
            for pattern in trip.patterns:
                expected.add((trip.name, pattern.id))
    got = {(trip, pid) for trip, pid, _pos, _neg in PATTERN_CASES}
    assert got == expected
    cases = load_cases()
    ids = {case.id for case in cases if case.category in {"escalation", "identity"}}
    missing = {f"{trip}__{pid}" for trip, pid in expected} - ids
    assert missing == set()


def test_clinical_rows_cover_every_output_pattern() -> None:
    raw = yaml.safe_load(
        (ROOT / "evals" / "cases" / "clinical_output" / "sentences.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = raw["rows"]
    assert len(rows) >= 30
    covered = {row["pattern"] for row in rows}
    pack_ids = {pattern.id for pattern in PACK.patterns}
    assert pack_ids <= covered


def test_clinical_row_labels_match_check_clinical() -> None:
    raw = yaml.safe_load(
        (ROOT / "evals" / "cases" / "clinical_output" / "sentences.yaml").read_text(
            encoding="utf-8"
        )
    )
    for index, row in enumerate(raw["rows"]):
        result = check_clinical(row["text"], PACK)
        assert isinstance(result, Blocked)
        assert row["pattern"] in result.pattern_ids, index


def test_sentence_tables_expand_ids() -> None:
    cases = load_cases()
    price = [case for case in cases if case.category == "price_invention"]
    clinical = [case for case in cases if case.category == "clinical_output"]
    assert [case.id for case in price] == [f"sentences_{i}" for i in range(len(price))]
    assert len(price) >= 50
    assert price[0].never_spoken
    assert [case.id for case in clinical] == [f"sentences_{i}" for i in range(len(clinical))]
    assert len(clinical) >= 30


def test_e2e_category_has_nine_cases() -> None:
    cases = [case for case in load_cases() if case.category == "e2e"]
    assert {case.id for case in cases} == {
        "full_booking",
        "slot_taken_then_other",
        "clinical_mid_booking",
        "identity_mid_call",
        "unquotable_consult",
        "invent_after_quote",
        "callback_sunday_new_number",
        "two_soft_with_gap",
        "duration_mid_booking",
    }
    duration = next(case for case in cases if case.id == "duration_mid_booking")
    assert duration.clock_at
    taken = next(case for case in cases if case.id == "slot_taken_then_other")
    assert taken.hooks.occupy_first
    consult = next(case for case in cases if case.id == "unquotable_consult")
    assert consult.history_absent


def test_price_invention_meets_floor() -> None:
    raw = yaml.safe_load(
        (ROOT / "evals" / "cases" / "price_invention" / "sentences.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert len(raw["rows"]) >= 50


async def test_planted_identity_passes() -> None:
    case = load_case_path(FIX / "identity" / "ok.yaml")
    assert await run_case(case) is None


async def test_planted_emergency_model_call_omits_utterance() -> None:
    case = load_case_path(FIX / "escalation" / "emergency_model.yaml")
    failure = await run_case(case)
    assert failure is not None
    assert failure.check == "spoken_missing"
    from evals.harness import Report

    report = Report()
    report.add(case, failure)
    text = format_report(report)
    assert "emergency_model" in text
    assert "spoken_missing" in text
    assert "breathe" not in text
    assert case.utterances[0] not in text


async def test_text_promise_omits_sentence() -> None:
    case = load_case_path(FIX / "continue" / "text_promise.yaml")
    failure = await run_case(case)
    assert failure is not None
    assert failure.check == "text_promise"
    from evals.harness import Report

    report = Report()
    report.add(case, failure)
    text = format_report(report)
    assert "text_promise" in text
    assert "I will text you" not in text


def test_model_chunk_before_quote_is_caught() -> None:
    chunks = (
        OutputChunk(text="So that's $700.", origin="model"),
        OutputChunk(
            text="Dermal Filler typically runs $700 to $950 per syringe.",
            origin="backend",
        ),
    )
    assert (
        check_model_order(
            chunks,
            allowed_amounts=set(),
            allowed_numbers=set(),
            pack=PACK,
            extract=EXTRACT,
        )
        == "model_sentence"
    )
    reversed_chunks = (chunks[1], chunks[0])
    assert (
        check_model_order(
            reversed_chunks,
            allowed_amounts=set(),
            allowed_numbers=set(),
            pack=PACK,
            extract=EXTRACT,
        )
        is None
    )


def test_harness_does_not_import_retell() -> None:
    tree = ast.parse(Path(evals.harness.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "retell" not in imported


async def test_planted_clinical_leak_omits_phrase() -> None:
    case = load_case_path(
        ROOT / "evals" / "cases" / "escalation" / "clinical_question__keloid.yaml"
    )
    phrase = "I keloid easily"
    keys = InMemoryKeyStore()
    await keys.create(f"eval_{case.id}")
    log = InMemoryEventLog(keys)
    await log.append(
        EventDraft(
            call_id=f"eval_{case.id}",
            tenant_id="glow_aesthetics",
            at=case.clock,
            kind="tool_invoked",
            payload={"tool": "lookup_faq", "args": json.dumps({"question_text": phrase})},
        )
    )
    events = await log.read(f"eval_{case.id}")
    state = CallState(
        call_id=f"eval_{case.id}",
        tenant_id="glow_aesthetics",
        started_at=case.clock,
        data_key_id="k",
    )
    state.record_commit("clinical_question", Transfer(tier=1, target="+12025550100"))
    state.transcript.append(CallerLine("[redacted:clinical_question]", case.clock))
    assert _clinical_leak(case, events, state)
    failure = CaseFailure(category=case.category, id=case.id, check="clinical_leak")
    report = Report()
    report.add(case, failure)
    text = format_report(report)
    assert failure.check == "clinical_leak"
    assert "clinical_leak" in text
    assert phrase not in text


class _StreamOnCommit(TurnLoop):
    async def handle_caller_turn(
        self, state: CallState, utterance: str
    ) -> AsyncIterator[OutputChunk]:
        chunks = [chunk async for chunk in super().handle_caller_turn(state, utterance)]
        if isinstance(state.escalation, Committed):
            stream = self.model.stream(
                (),
                (),
                max_output_tokens=self.max_output_tokens,
                first_token_timeout=self.first_token_timeout,
                idle_timeout=self.idle_timeout,
            )
            try:
                async for _event in stream:
                    pass
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        for chunk in chunks:
            yield chunk


async def test_planted_model_after_commit() -> None:
    case = load_case_path(
        ROOT / "evals" / "cases" / "escalation" / "medical_emergency__cant_breathe.yaml"
    )
    failure = await run_case(case, loop_cls=_StreamOnCommit)
    assert failure is not None
    assert failure.check == "model_after_commit"


def test_scripted_from_mapping_round_trips() -> None:
    llm = ScriptedLLM.from_mapping(
        {
            "turns": [
                {
                    "id": "one",
                    "events": [{"type": "text", "text": "hi"}, {"type": "done"}],
                }
            ]
        }
    )
    assert llm.call_count == 0
