from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Sequence
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import Runtime, Settings, _build_loop, create_app
from core.crypto import InMemoryKeyStore
from core.escalation import EMERGENCY_LINE, TRANSFER_LINE
from core.events import InMemoryEventLog
from core.llm import (
    CannedDone,
    CannedText,
    CannedTurn,
    Done,
    Message,
    ScriptedLLM,
    TextDelta,
    ToolSchema,
)
from core.state import CallerLine, InMemorySessionStore
from core.tenant import TenantRegistry
from core.turn import GOODBYE, HEDGE
from integrations.retell import (
    ALLOWED_STATUS,
    CallInfo,
    CallLookupError,
    ConfigResponse,
    PingPongResponse,
    ResponseResponse,
    RetellApiLookup,
    check_call,
    last_user_text,
    parse_inbound,
    require_token,
    token_matches,
    valid_call_id,
)

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 11, 5)
NOW = datetime(2026, 11, 3, 11, 0, tzinfo=ZoneInfo("America/New_York"))
TOKEN = "w" * 32
AGENT = "agent_glow"
TENANT = "glow_aesthetics"
CALL = "call_ws_1"
EMERGENCY = "I can't breathe"
FILLER = "hello there friend"
SAMPLE_PING = {"interaction_type": "ping_pong", "timestamp": 123456}
SAMPLE_DETAILS = {
    "interaction_type": "call_details",
    "call": {"call_id": CALL, "agent_id": AGENT, "call_status": "registered"},
}
SAMPLE_UPDATE = {
    "interaction_type": "update_only",
    "transcript": [
        {"role": "agent", "content": "Hi"},
        {"role": "user", "content": "Hello"},
    ],
}
SAMPLE_RESPONSE = {
    "interaction_type": "response_required",
    "response_id": 1,
    "transcript": [{"role": "user", "content": "Hello"}],
}
SAMPLE_REMINDER = {
    "interaction_type": "reminder_required",
    "response_id": 2,
    "transcript": [],
}


class FakeLookup:
    def __init__(
        self,
        *,
        status: str = "ongoing",
        agent_id: str = AGENT,
        from_number: str | None = "+12035550148",
        error: CallLookupError | None = None,
    ) -> None:
        self.status = status
        self.agent_id = agent_id
        self.from_number = from_number
        self.error = error
        self.calls: list[str] = []

    async def get_call(self, call_id: str) -> CallInfo:
        self.calls.append(call_id)
        if self.error is not None:
            raise self.error
        return CallInfo(
            call_id=call_id,
            agent_id=self.agent_id,
            call_status=self.status,
            from_number=self.from_number,
        )


class _PlayStream:
    def __init__(self, play: AsyncIterator[TextDelta | Done]) -> None:
        self._play = play

    def __aiter__(self) -> _PlayStream:
        return self

    async def __anext__(self) -> TextDelta | Done:
        return await self._play.__anext__()

    async def aclose(self) -> None:
        close = getattr(self._play, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except (GeneratorExit, asyncio.CancelledError, StopAsyncIteration):
            return


class GateModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        max_output_tokens: int,
        first_token_timeout: float,
        idle_timeout: float,
    ) -> _PlayStream:
        del messages, tools, max_output_tokens, first_token_timeout, idle_timeout
        self.calls += 1
        return _PlayStream(self._play())

    async def _play(self) -> AsyncIterator[TextDelta | Done]:
        self.entered.set()
        while not self.release.is_set():
            await asyncio.sleep(0.01)
        yield TextDelta("Happy to help.")
        yield Done()


def _scripted() -> ScriptedLLM:
    turns = tuple(
        CannedTurn(id=f"t{i}", events=(CannedText("Happy to help."), CannedDone()))
        for i in range(6)
    )
    return ScriptedLLM(turns)


def make_runtime(
    *,
    model: Any | None = None,
    lookup: FakeLookup | None = None,
    token: str = TOKEN,
    agent_id: str = AGENT,
) -> tuple[Runtime, FakeLookup]:
    settings = Settings(
        ws_token=token,
        agent_id=agent_id,
        retell_api_key="k" * 32,
        booking_secret=b"n" * 32,
    )
    tenants = TenantRegistry.load(ROOT / "tenants", today=TODAY)
    keys = InMemoryKeyStore()
    log = InMemoryEventLog(keys)
    store = InMemorySessionStore()
    fake = lookup if lookup is not None else FakeLookup()
    chosen = model if model is not None else _scripted()
    loops = {}
    for tid in tenants.ids:
        tenant = tenants.get(tid)
        assert tenant is not None
        loops[tid] = _build_loop(
            tenant, store, log, chosen, settings.booking_secret, clock=lambda: NOW
        )
    runtime = Runtime(settings, fake, tenants, store, keys, log, loops)
    return runtime, fake


def ws_url(token: str = TOKEN, tenant: str = TENANT, call_id: str = CALL) -> str:
    return f"/llm-websocket/{token}/{tenant}/{call_id}"


def receive_until(ws: Any, pred: Any, limit: int = 30) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for _ in range(limit):
        frame = ws.receive_json()
        frames.append(frame)
        if pred(frame):
            return frames
    raise AssertionError(frames)


def drain_begin(ws: Any) -> list[dict[str, Any]]:
    first = ws.receive_json()
    assert first["response_type"] == "config"
    assert first["config"]["auto_reconnect"] is True
    assert first["config"]["call_details"] is True
    rest = receive_until(
        ws,
        lambda frame: (
            frame.get("response_type") == "response"
            and frame.get("response_id") == 0
            and frame.get("content_complete") is True
        ),
    )
    return [first, *rest]


def response_required(response_id: int, text: str) -> dict[str, object]:
    return {
        "interaction_type": "response_required",
        "response_id": response_id,
        "transcript": [{"role": "user", "content": text}],
    }


def update_only(text: str) -> dict[str, object]:
    return {
        "interaction_type": "update_only",
        "transcript": [{"role": "user", "content": text}],
    }


def reminder_required(response_id: int) -> dict[str, object]:
    return {"interaction_type": "reminder_required", "response_id": response_id, "transcript": []}


def load_state(runtime: Runtime, call_id: str = CALL) -> Any:
    return asyncio.run(runtime.store.get(call_id))


def load_kinds(runtime: Runtime, call_id: str = CALL) -> list[str]:
    return [event.kind for event in asyncio.run(runtime.log.read(call_id))]


def caller_lines(state: Any) -> list[CallerLine]:
    return [line for line in state.transcript if isinstance(line, CallerLine)]


# --------------------------------------------------------------------------- parse / auth helpers


def test_sample_inbound_events_parse() -> None:
    assert parse_inbound(SAMPLE_PING).interaction_type == "ping_pong"  # type: ignore[union-attr]
    assert parse_inbound(SAMPLE_DETAILS).interaction_type == "call_details"  # type: ignore[union-attr]
    assert parse_inbound(SAMPLE_UPDATE).interaction_type == "update_only"  # type: ignore[union-attr]
    assert parse_inbound(SAMPLE_RESPONSE).response_id == 1  # type: ignore[union-attr]
    assert parse_inbound(SAMPLE_REMINDER).interaction_type == "reminder_required"  # type: ignore[union-attr]
    assert parse_inbound({"interaction_type": "nope"}) is None
    assert parse_inbound({"interaction_type": "response_required"}) is None


def test_sample_outbound_required_fields() -> None:
    config = ConfigResponse(config={"auto_reconnect": True, "call_details": True}).model_dump()
    assert config["response_type"] == "config"
    ping = PingPongResponse(timestamp=9).model_dump()
    assert ping == {"response_type": "ping_pong", "timestamp": 9}
    response = ResponseResponse(response_id=0, content="Hi", content_complete=True).model_dump()
    assert response["response_type"] == "response"
    assert response["content_complete"] is True


def test_last_user_text_skips_agent() -> None:
    assert (
        last_user_text([{"role": "agent", "content": "Hi"}, {"role": "user", "content": "Hello"}])
        == "Hello"
    )
    assert last_user_text([]) is None


def test_token_matches_rejects_length_mismatch() -> None:
    assert token_matches(TOKEN, TOKEN) is True
    assert token_matches("x" * 32, TOKEN) is False
    assert token_matches("short", TOKEN) is False


def test_require_token_min_length() -> None:
    with pytest.raises(ValueError, match="32"):
        require_token("too-short")
    assert require_token(TOKEN) == TOKEN


def test_valid_call_id_shape() -> None:
    assert valid_call_id("call_ws-1") is True
    assert valid_call_id("A" * 128) is True
    assert valid_call_id("A" * 129) is False
    assert valid_call_id("bad.call") is False
    assert valid_call_id("has space") is False
    assert valid_call_id("") is False


@pytest.mark.parametrize("status", sorted(ALLOWED_STATUS))
def test_check_call_accepts_registered_and_ongoing(status: str) -> None:
    check_call(CallInfo(CALL, AGENT, status), AGENT)


@pytest.mark.parametrize("status", ["not_connected", "ended", "error", "unknown", ""])
def test_check_call_rejects_other_statuses(status: str) -> None:
    with pytest.raises(CallLookupError) as exc:
        check_call(CallInfo(CALL, AGENT, status), AGENT)
    assert exc.value.reason == "status"


def test_check_call_rejects_wrong_agent() -> None:
    with pytest.raises(CallLookupError) as exc:
        check_call(CallInfo(CALL, "other", "ongoing"), AGENT)
    assert exc.value.reason == "agent"


@pytest.mark.asyncio
async def test_retell_api_lookup_timeout_and_errors() -> None:
    class Slow:
        call = SimpleNamespace(retrieve=lambda _id: asyncio.sleep(1))

    with pytest.raises(CallLookupError) as timed:
        await RetellApiLookup(Slow(), timeout_s=0.01).get_call(CALL)
    assert timed.value.reason == "timeout"

    class Boom:
        async def retrieve(self, _id: str) -> None:
            raise RuntimeError("nope")

        call = None

        def __init__(self) -> None:
            self.call = self

    with pytest.raises(CallLookupError) as lookup:
        await RetellApiLookup(Boom()).get_call(CALL)
    assert lookup.value.reason == "lookup"

    class Ok:
        def __init__(self) -> None:
            self.call = self

        async def retrieve(self, call_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                agent_id=AGENT, call_status="ongoing", from_number="+12035550148"
            )

    info = await RetellApiLookup(Ok()).get_call(CALL)
    assert info.agent_id == AGENT
    assert info.call_status == "ongoing"
    assert "12035550148" not in repr(info)


def test_call_info_repr_omits_from_number() -> None:
    info = CallInfo(CALL, AGENT, "ongoing", from_number="+12035550148")
    assert "+12035550148" not in repr(info)
    assert "+12035550148" not in str(info)


# --------------------------------------------------------------------------- websocket auth


def test_invalid_call_id_closes_without_logging_it(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    runtime, lookup = make_runtime()
    client = TestClient(create_app(runtime))
    bad = "bad.call"
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url(call_id=bad)):
            pass
    assert exc.value.code == 1008
    assert lookup.calls == []
    assert bad not in caplog.text
    assert "invalid" in caplog.text


def test_bad_token_closes_without_lookup() -> None:
    runtime, lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url(token="x" * 32)):
            pass
    assert exc.value.code == 1008
    assert lookup.calls == []


def test_unknown_tenant_closes_without_lookup() -> None:
    runtime, lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url(tenant="nope")):
            pass
    assert exc.value.code == 1008
    assert lookup.calls == []


@pytest.mark.parametrize("reason,error", [("lookup", "lookup"), ("timeout", "timeout")])
def test_lookup_failure_closes(reason: str, error: str) -> None:
    runtime, lookup = make_runtime(lookup=FakeLookup(error=CallLookupError(error)))
    client = TestClient(create_app(runtime))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url()):
            pass
    assert exc.value.code == 1008
    assert lookup.calls == [CALL]
    del reason


def test_wrong_agent_closes() -> None:
    runtime, lookup = make_runtime(lookup=FakeLookup(agent_id="other"))
    client = TestClient(create_app(runtime))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url()):
            pass
    assert exc.value.code == 1008
    assert lookup.calls == [CALL]


@pytest.mark.parametrize("status", ["not_connected", "ended", "error", "mystery"])
def test_disallowed_status_closes(status: str) -> None:
    runtime, _lookup = make_runtime(lookup=FakeLookup(status=status))
    client = TestClient(create_app(runtime))
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(ws_url()):
            pass
    assert exc.value.code == 1008


@pytest.mark.parametrize("status", ["registered", "ongoing"])
def test_allowed_status_connects(status: str) -> None:
    runtime, lookup = make_runtime(lookup=FakeLookup(status=status))
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        frames = drain_begin(ws)
    assert lookup.calls == [CALL]
    spoken = [frame["content"] for frame in frames if frame.get("response_type") == "response"]
    assert len(spoken) == 2
    assert frames[-1]["content_complete"] is True
    assert frames[-1]["response_id"] == 0


def test_first_connect_greets_then_disclosure() -> None:
    runtime, _lookup = make_runtime()
    tenant = runtime.tenants.get(TENANT)
    assert tenant is not None
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        frames = drain_begin(ws)
    spoken = [frame["content"] for frame in frames if frame.get("response_type") == "response"]
    assert spoken == [tenant.greeting, tenant.record_disclosure]
    state = load_state(runtime)
    assert state is not None
    assert state.caller_number == "+12035550148"
    assert state.turn_count == 0


def test_second_socket_supersedes_first() -> None:
    gate = GateModel()
    runtime, _lookup = make_runtime(model=gate)
    with TestClient(create_app(runtime)) as client:
        with client.websocket_connect(ws_url()) as first:
            drain_begin(first)
            first.send_json(response_required(1, FILLER))
            assert gate.entered.wait(timeout=2)
            with client.websocket_connect(ws_url()) as second:
                with pytest.raises(WebSocketDisconnect) as exc:
                    first.receive_json()
                assert exc.value.code == 1001
                frames = drain_begin(second)
                spoken = [f["content"] for f in frames if f.get("response_type") == "response"]
                assert spoken == [""]
                second.send_json(response_required(1, FILLER))
                gate.release.set()
                receive_until(
                    second,
                    lambda frame: (
                        frame.get("response_type") == "response"
                        and frame.get("response_id") == 1
                        and frame.get("content_complete") is True
                    ),
                )
    state = load_state(runtime)
    assert state is not None
    assert [line.text for line in caller_lines(state)] == [FILLER]
    assert state.turn_count == 1
    assert "error" not in load_kinds(runtime)
    assert runtime.live == {}


def test_reconnect_empty_begin_does_not_greet_again() -> None:
    runtime, _lookup = make_runtime()
    tenant = runtime.tenants.get(TENANT)
    assert tenant is not None
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        first = drain_begin(ws)
    with client.websocket_connect(ws_url()) as ws:
        second = drain_begin(ws)
    spoken_first = [f["content"] for f in first if f.get("response_type") == "response"]
    spoken_second = [f["content"] for f in second if f.get("response_type") == "response"]
    assert spoken_first == [tenant.greeting, tenant.record_disclosure]
    assert spoken_second == [""]
    assert second[-1]["content_complete"] is True


def test_binary_frame_closes_1007() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_bytes(b"nope")
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
    assert exc.value.code == 1007


# --------------------------------------------------------------------------- protocol mapping


def test_response_required_echoes_id_and_completes() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, FILLER))
        frames = receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 1
                and frame.get("content_complete") is True
            ),
        )
    spoken = [frame for frame in frames if frame.get("response_type") == "response"]
    assert all(frame.get("response_id") == 1 for frame in spoken)
    state = load_state(runtime)
    assert state is not None
    assert state.turn_count == 1
    assert [line.text for line in caller_lines(state)] == [FILLER]


def test_newer_response_id_silences_old() -> None:
    gate = GateModel()
    runtime, _lookup = make_runtime(model=gate)
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, "hi"))
        assert gate.entered.wait(timeout=2)
        ws.send_json(response_required(2, FILLER))
        gate.release.set()
        frames = receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 2
                and frame.get("content_complete") is True
            ),
        )
    old = [f for f in frames if f.get("response_id") == 1]
    assert old == []
    assert any(f.get("content") == "Happy to help." and f.get("response_id") == 2 for f in frames)


def test_continue_mid_turn_then_response_required_records_once() -> None:
    gate = GateModel()
    runtime, _lookup = make_runtime(model=gate)
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, "hi"))
        assert gate.entered.wait(timeout=2)
        ws.send_json(update_only(FILLER))
        ws.send_json(response_required(2, FILLER))
        gate.release.set()
        receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 2
                and frame.get("content_complete") is True
            ),
        )
    state = load_state(runtime)
    assert state is not None
    assert state.turn_count == 1
    assert [line.text for line in caller_lines(state)] == [FILLER]


def test_commit_mid_turn_cancels_old_and_commits_once() -> None:
    gate = GateModel()
    runtime, _lookup = make_runtime(model=gate)
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, FILLER))
        assert gate.entered.wait(timeout=2)
        ws.send_json(update_only(EMERGENCY))
        frames = receive_until(ws, lambda frame: frame.get("response_type") == "agent_interrupt")
        interrupt = frames[-1]
        assert interrupt["content"] == EMERGENCY_LINE
        assert interrupt["no_interruption_allowed"] is True
        assert interrupt["end_call"] is True
        assert interrupt["content_complete"] is True
        ws.send_json(response_required(2, EMERGENCY))
        follow = receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 2
                and frame.get("content_complete") is True
            ),
        )
    old = [f for f in frames + follow if f.get("response_id") == 1 and f.get("content")]
    assert old == []
    state = load_state(runtime)
    assert state is not None
    assert state.turn_count == 1
    assert len(caller_lines(state)) == 1
    kinds = load_kinds(runtime)
    assert kinds.count("tripwire_fired") == 1
    assert kinds.count("escalation_committed") == 1
    assert kinds.count("escalation_path_chosen") == 1
    assert "error" not in kinds
    assert "Happy to help." not in [f.get("content") for f in frames + follow]


def test_update_only_skips_when_committed() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, EMERGENCY))
        receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response" and frame.get("content_complete") is True
            ),
        )
        ws.send_json(update_only("get me a real person"))
        ws.send_json(SAMPLE_PING)
        assert ws.receive_json()["response_type"] == "ping_pong"
    kinds = load_kinds(runtime)
    assert kinds.count("tripwire_fired") == 1
    assert kinds.count("escalation_committed") == 1
    assert "error" not in kinds


def test_update_only_with_no_inflight_does_not_record() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(update_only(FILLER))
        ws.send_json(SAMPLE_PING)
        pong = ws.receive_json()
    assert pong == {"response_type": "ping_pong", "timestamp": 123456}
    state = load_state(runtime)
    assert state is not None
    assert caller_lines(state) == []
    assert state.turn_count == 0


def test_update_only_logs_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(update_only(FILLER))
        ws.send_json(SAMPLE_PING)
        assert ws.receive_json()["response_type"] == "ping_pong"
    events = [record for record in caplog.records if record.name == "core.events"]
    assert any(
        record.levelno == logging.DEBUG and "update_only" in record.getMessage()
        for record in events
    )
    assert not any(
        record.levelno >= logging.INFO and "update_only" in record.getMessage() for record in events
    )


def test_reminder_then_goodbye() -> None:
    runtime, _lookup = make_runtime()
    tenant = runtime.tenants.get(TENANT)
    assert tenant is not None
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        spoken: list[str] = []
        ends: list[bool] = []
        for index in range(3):
            ws.send_json(reminder_required(index + 1))
            frames = receive_until(
                ws,
                lambda frame: (
                    frame.get("response_type") == "response"
                    and frame.get("content_complete") is True
                ),
            )
            spoken.extend(
                frame["content"] for frame in frames if frame.get("response_type") == "response"
            )
            ends.append(bool(frames[-1].get("end_call")))
    assert spoken[:2] == [tenant.reminder_line, tenant.reminder_line]
    assert GOODBYE in spoken[-1]
    assert ends == [False, False, True]


def test_ping_during_slow_turn() -> None:
    gate = GateModel()
    runtime, _lookup = make_runtime(model=gate)
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, FILLER))
        assert gate.entered.wait(timeout=2)
        ws.send_json(SAMPLE_PING)
        pong = ws.receive_json()
        assert pong == {"response_type": "ping_pong", "timestamp": 123456}
        gate.release.set()
        receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 1
                and frame.get("content_complete") is True
            ),
        )


def test_transfer_number_on_first_chunk() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, "get me a real person"))
        frames = receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response"
                and frame.get("response_id") == 1
                and frame.get("content_complete") is True
            ),
        )
    spoken = [f for f in frames if f.get("response_type") == "response"]
    assert spoken[0]["content"] == TRANSFER_LINE
    assert spoken[0]["transfer_number"] == "+12035550101"
    assert spoken[-1]["content_complete"] is True


def test_unknown_event_ignored_socket_stays_up() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json({"interaction_type": "tool_call_invocation", "tool": "nope"})
        ws.send_json(SAMPLE_PING)
        assert ws.receive_json()["response_type"] == "ping_pong"


def test_privacy_logs_ids_not_utterance_or_token(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, EMERGENCY))
        receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response" and frame.get("content_complete") is True
            ),
        )
    text = caplog.text
    assert "ws_open" in text
    assert CALL in text
    assert TOKEN not in text
    assert EMERGENCY not in text
    assert "breathe" not in text
    assert "/llm-websocket/" not in text


def test_adapter_exception_sends_hedge() -> None:
    class Boom(GateModel):
        async def _play(self) -> AsyncIterator[TextDelta | Done]:
            raise RuntimeError("boom")
            yield TextDelta("no")  # pragma: no cover

    runtime, _lookup = make_runtime(model=Boom())
    client = TestClient(create_app(runtime))
    with client.websocket_connect(ws_url()) as ws:
        drain_begin(ws)
        ws.send_json(response_required(1, FILLER))
        frames = receive_until(
            ws,
            lambda frame: (
                frame.get("response_type") == "response" and frame.get("content_complete") is True
            ),
        )
    assert any(frame.get("content") == HEDGE for frame in frames)
