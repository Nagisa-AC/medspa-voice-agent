"""Interactive Glow text demo. Synthetic data only. Does not talk to Retell."""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

from core.crypto import InMemoryKeyStore
from core.events import EventDraft, InMemoryEventLog
from core.guardrail import load_output_pack
from core.llm import Model
from core.numbers import find_numbers
from core.policy import load_policy_engine
from core.state import CallState, InMemorySessionStore, new_call_state
from core.tenant import extra_price_words, load_tenant_file
from core.turn import TurnLoop
from integrations.booking import MockBookingAdapter
from integrations.llm_anthropic import AnthropicLLM

ROOT = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")
SUNDAY = datetime(2026, 11, 8, 14, 0, tzinfo=NY)
SYNTHETIC_CALLER = "+12035551234"
BANNER = "Synthetic data only. Do not type real patient information. Type quit or exit to leave."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glow text demo (synthetic data only)")
    parser.add_argument(
        "--sunday",
        action="store_true",
        help="shift the clock to Sunday 2026-11-08 14:00 America/New_York plus elapsed",
    )
    parser.add_argument(
        "--no-caller-id",
        action="store_true",
        help="leave caller_number unset",
    )
    return parser.parse_args(argv)


def make_clock(
    *, sunday: bool, tz: ZoneInfo, mono: Callable[[], float] = time.monotonic
) -> Callable[[], datetime]:
    if not sunday:
        return lambda: datetime.now(tz)
    origin = mono()

    def clock() -> datetime:
        return SUNDAY + timedelta(seconds=mono() - origin)

    return clock


def _print(line: str, out: TextIO) -> None:
    print(line, file=out, flush=True)


async def _new_event_kinds(log: InMemoryEventLog, call_id: str, seen: int, out: TextIO) -> int:
    events = await log.read(call_id)
    kinds = [event.kind for event in events[seen:]]
    if kinds:
        _print("events: " + ",".join(kinds), out)
    return len(events)


async def drive(
    loop: TurnLoop,
    state: CallState,
    log: InMemoryEventLog,
    model: Model,
    *,
    read_line: Callable[[], str | None],
    out: TextIO = sys.stdout,
) -> None:
    seen = 0
    async for chunk in loop.connect(state):
        _print(f"[{chunk.origin}] {chunk.text}", out)
    seen = await _new_event_kinds(log, state.call_id, seen, out)
    while True:
        raw = read_line()
        if raw is None:
            break
        line = raw.strip()
        if not line:
            continue
        if line.lower() in {"quit", "exit"}:
            break
        take = getattr(model, "take_first_token_times", None)
        if callable(take):
            take()
        async for chunk in loop.handle_caller_turn(state, line):
            _print(f"[{chunk.origin}] {chunk.text}", out)
        if callable(take):
            for seconds in take():
                _print(f"[latency] first_token={seconds * 1000:.0f}ms", out)
        seen = await _new_event_kinds(log, state.call_id, seen, out)


async def run(args: argparse.Namespace, *, model: Model | None = None) -> int:
    today = SUNDAY.date() if args.sunday else datetime.now(NY).date()
    tenant = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=today)
    clock = make_clock(sunday=args.sunday, tz=tenant.tz)
    now = clock()
    words = extra_price_words(tenant)
    extract = partial(
        find_numbers,
        extra_price_words=words.extra_price_words,
        service_words=words.service_words,
    )
    keys = InMemoryKeyStore()
    data_key_id = await keys.create("demo_1")
    log = InMemoryEventLog(keys)
    store = InMemorySessionStore()
    state = new_call_state(
        tenant=tenant, call_id="demo_1", data_key_id=data_key_id, now=now, extract=extract
    )
    if not args.no_caller_id:
        state.caller_number = SYNTHETIC_CALLER
    await store.create(state)
    payload: dict[str, str | int | bool | list[str]] = {}
    if state.caller_number:
        payload["caller_number"] = state.caller_number
    await log.append(
        EventDraft(
            call_id=state.call_id,
            tenant_id=state.tenant_id,
            at=now,
            kind="call_started",
            payload=payload,
        )
    )
    llm = model if model is not None else AnthropicLLM()
    loop = TurnLoop(
        tenant,
        store,
        log,
        llm,
        MockBookingAdapter.from_yaml(
            ROOT / "tenants" / "glow_aesthetics.mock.yaml", tz=tenant.tz, clock=clock
        ),
        secrets.token_bytes(32),
        clock=clock,
        engine=load_policy_engine(tenant),
        pack=load_output_pack(ROOT / "packs" / "medspa.output.yaml"),
        extract=extract,
    )
    _print(BANNER, sys.stdout)
    await drive(loop, state, log, llm, read_line=_stdin_line, out=sys.stdout)
    return 0


def _stdin_line() -> str | None:
    try:
        return input("> ")
    except EOFError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
