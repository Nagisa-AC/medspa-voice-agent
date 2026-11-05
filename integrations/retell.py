"""Retell Custom LLM websocket adapter. core/ never imports this module."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, ValidationError

from core.events import log_ids
from core.policy import Arm, Commit, Continue, Reply
from core.state import CallState, Committed
from core.turn import HEDGE, OutputChunk, TurnLoop

_TOKEN_MIN = 32
_LOOKUP_TIMEOUT_S = 2.0
ALLOWED_STATUS = frozenset({"registered", "ongoing"})
_CLOSE_POLICY = 1008
_CLOSE_BAD_DATA = 1007
_CLOSE_GOING_AWAY = 1001
_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class CallLookupError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)

    def __repr__(self) -> str:
        return f"CallLookupError({self.reason!r})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True)
class CallInfo:
    call_id: str
    agent_id: str
    call_status: str
    from_number: str | None = None

    def __repr__(self) -> str:
        return (
            f"CallInfo(call_id={self.call_id!r}, agent_id={self.agent_id!r}, "
            f"call_status={self.call_status!r})"
        )

    def __str__(self) -> str:
        return repr(self)


class CallLookup(Protocol):
    async def get_call(self, call_id: str) -> CallInfo: ...


class RetellApiLookup:
    def __init__(self, client: Any, *, timeout_s: float = _LOOKUP_TIMEOUT_S) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def __repr__(self) -> str:
        return "RetellApiLookup()"

    def __str__(self) -> str:
        return repr(self)

    async def get_call(self, call_id: str) -> CallInfo:
        try:
            raw = await asyncio.wait_for(self._client.call.retrieve(call_id), self._timeout_s)
        except TimeoutError:
            raise CallLookupError("timeout") from None
        except Exception:
            raise CallLookupError("lookup") from None
        agent = str(getattr(raw, "agent_id", "") or "")
        status = str(getattr(raw, "call_status", "") or "")
        from_raw = getattr(raw, "from_number", None)
        from_number = from_raw if isinstance(from_raw, str) and from_raw else None
        if not agent:
            raise CallLookupError("lookup")
        return CallInfo(
            call_id=call_id, agent_id=agent, call_status=status, from_number=from_number
        )


def check_call(info: CallInfo, agent_id: str) -> None:
    if info.agent_id != agent_id:
        raise CallLookupError("agent")
    if info.call_status not in ALLOWED_STATUS:
        raise CallLookupError("status")


def token_matches(got: str, expected: str) -> bool:
    left = got.encode("utf-8")
    right = expected.encode("utf-8")
    if len(left) != len(right):
        hmac.compare_digest(b"\x00", b"\x01")
        return False
    return hmac.compare_digest(left, right)


def require_token(token: str) -> str:
    if len(token) < _TOKEN_MIN:
        raise ValueError("RETELL_WS_TOKEN must be at least 32 characters")
    return token


def valid_call_id(call_id: str) -> bool:
    return bool(_CALL_ID.fullmatch(call_id))


# --------------------------------------------------------------------------- pydantic events


class _Extra(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Utterance(_Extra):
    role: Literal["agent", "user"]
    content: str


class PingPongRequest(_Extra):
    interaction_type: Literal["ping_pong"]
    timestamp: int


class CallDetailsRequest(_Extra):
    interaction_type: Literal["call_details"]
    call: dict[str, Any]


class UpdateOnlyRequest(_Extra):
    interaction_type: Literal["update_only"]
    transcript: list[Utterance] = []
    turntaking: Literal["agent_turn", "user_turn"] | None = None


class ResponseRequiredRequest(_Extra):
    interaction_type: Literal["response_required", "reminder_required"]
    transcript: list[Utterance] = []
    response_id: int


class ConfigResponse(BaseModel):
    response_type: Literal["config"] = "config"
    config: dict[str, bool]


class PingPongResponse(BaseModel):
    response_type: Literal["ping_pong"] = "ping_pong"
    timestamp: int


class ResponseResponse(BaseModel):
    response_type: Literal["response"] = "response"
    response_id: int
    content: str
    content_complete: bool
    no_interruption_allowed: bool | None = None
    end_call: bool | None = None
    transfer_number: str | None = None


Inbound = PingPongRequest | CallDetailsRequest | UpdateOnlyRequest | ResponseRequiredRequest


def parse_inbound(raw: Mapping[str, object]) -> Inbound | None:
    kind = raw.get("interaction_type")
    try:
        if kind == "ping_pong":
            return PingPongRequest.model_validate(raw)
        if kind == "call_details":
            return CallDetailsRequest.model_validate(raw)
        if kind == "update_only":
            return UpdateOnlyRequest.model_validate(raw)
        if kind in ("response_required", "reminder_required"):
            return ResponseRequiredRequest.model_validate(raw)
    except ValidationError:
        return None
    return None


def last_user_text(transcript: object) -> str | None:
    if not isinstance(transcript, list):
        return None
    last: str | None = None
    for item in transcript:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            last = content
    return last


class CallSession:
    def __init__(
        self,
        websocket: WebSocket,
        *,
        loop: TurnLoop,
        state: CallState,
        lock: asyncio.Lock,
        call_id: str,
        tenant_id: str,
    ) -> None:
        self._ws = websocket
        self.loop = loop
        self.state = state
        self.lock = lock
        self.call_id = call_id
        self.tenant_id = tenant_id
        self._out = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._response_id: int | None = None
        self._consumed_text: str | None = None
        self._consumed_turn: int | None = None
        self._interrupt_id = 0

    def __repr__(self) -> str:
        return f"CallSession(call_id={self.call_id!r}, tenant_id={self.tenant_id!r})"

    def __str__(self) -> str:
        return repr(self)

    async def supersede(self) -> None:
        await self._cancel_current()
        try:
            await self._ws.close(code=_CLOSE_GOING_AWAY)
        except Exception:
            return

    async def run(self, *, reconnect: bool) -> None:
        await self._send(
            {
                "response_type": "config",
                "config": {"auto_reconnect": True, "call_details": True},
            }
        )
        if reconnect:
            await self._send_response(0, "", complete=True)
        else:
            async with self.lock:
                self._response_id = 0
                await self._stream_chunks(0, self.loop.connect(self.state), interrupt=False)
                self._response_id = None
        try:
            while True:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    await self._ws.close(code=_CLOSE_BAD_DATA)
                    return
                text = message.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    raw: object = json.loads(text)
                except json.JSONDecodeError:
                    log_ids("ws_bad_json", call_id=self.call_id)
                    continue
                if not isinstance(raw, dict):
                    continue
                await self._on_message(raw)
        except WebSocketDisconnect:
            return
        finally:
            await self._cancel_current()

    async def _on_message(self, raw: dict[str, object]) -> None:
        kind = raw.get("interaction_type")
        if not isinstance(kind, str):
            return
        if kind == "ping_pong":
            stamp = raw.get("timestamp")
            if isinstance(stamp, int):
                await self._send({"response_type": "ping_pong", "timestamp": stamp})
            return
        if kind == "call_details":
            log_ids("ws_event", call_id=self.call_id, interaction_type="call_details")
            return
        if kind == "update_only":
            log_ids(
                "ws_event",
                call_id=self.call_id,
                interaction_type="update_only",
                level=logging.DEBUG,
            )
            await self._on_update(raw)
            return
        if kind in ("response_required", "reminder_required"):
            response_id = raw.get("response_id")
            if not isinstance(response_id, int):
                log_ids("ws_bad_response_id", call_id=self.call_id)
                return
            log_ids(
                "ws_event",
                call_id=self.call_id,
                interaction_type=kind,
                response_id=response_id,
            )
            utterance = last_user_text(raw.get("transcript"))
            await self._start_response(response_id, utterance, reminder=kind == "reminder_required")
            return
        log_ids("ws_unknown_event", call_id=self.call_id)

    async def _on_update(self, raw: dict[str, object]) -> None:
        if isinstance(self.state.escalation, Committed):
            return
        if self._task is None or self._task.done():
            return
        utterance = last_user_text(raw.get("transcript"))
        if utterance is None:
            return
        result = self.loop.engine.evaluate(utterance, self.state, now=self.loop.clock())
        if isinstance(result, Continue | Arm):
            return
        if not isinstance(result, Commit | Reply):
            return
        await self._cancel_current()
        async with self.lock:
            chunk = await self.loop.handle_mid_turn_update(self.state, utterance)
            self._consumed_text = utterance
            self._consumed_turn = self.state.turn_count
            if chunk is not None:
                await self._send_chunk(chunk, first=True, interrupt=True)

    async def _start_response(
        self, response_id: int, utterance: str | None, *, reminder: bool
    ) -> None:
        await self._cancel_current()
        self._response_id = response_id
        self._task = asyncio.create_task(
            self._run_response(response_id, utterance, reminder=reminder)
        )

    async def _run_response(
        self, response_id: int, utterance: str | None, *, reminder: bool
    ) -> None:
        try:
            async with self.lock:
                if self._response_id != response_id:
                    return
                if reminder:
                    chunks: AsyncIterator[OutputChunk] = self.loop.handle_reminder(
                        self.state, response_id=str(response_id)
                    )
                elif (
                    utterance is not None
                    and utterance == self._consumed_text
                    and self.state.turn_count == self._consumed_turn
                ):
                    chunks = self.loop.handle_consumed_response(self.state, utterance)
                elif utterance is None:
                    await self._send_response(response_id, "", complete=True)
                    return
                else:
                    chunks = self.loop.handle_caller_turn(
                        self.state, utterance, response_id=str(response_id)
                    )
                await self._stream_chunks(response_id, chunks, interrupt=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            log_ids("ws_turn_error", call_id=self.call_id, response_id=response_id)
            if self._response_id == response_id:
                await self._send_response(response_id, HEDGE, complete=True)

    async def _stream_chunks(
        self,
        response_id: int,
        chunks: AsyncIterator[OutputChunk],
        *,
        interrupt: bool,
    ) -> None:
        sent = False
        first = True
        async for chunk in chunks:
            if not interrupt and self._response_id != response_id:
                return
            sent = True
            await self._send_chunk(chunk, response_id=response_id, first=first, interrupt=interrupt)
            first = False
        if not sent and not interrupt and self._response_id == response_id:
            await self._send_response(response_id, "", complete=True)

    async def _send_chunk(
        self,
        chunk: OutputChunk,
        *,
        response_id: int = 0,
        first: bool,
        interrupt: bool,
    ) -> None:
        if interrupt:
            self._interrupt_id += 1
            payload: dict[str, object] = {
                "response_type": "agent_interrupt",
                "interrupt_id": self._interrupt_id,
                "content": chunk.text,
                "content_complete": True,
                "no_interruption_allowed": True,
            }
            if chunk.end_call:
                payload["end_call"] = True
            if chunk.transfer_number:
                payload["transfer_number"] = chunk.transfer_number
            await self._send(payload)
            return
        payload = {
            "response_type": "response",
            "response_id": response_id,
            "content": chunk.text,
            "content_complete": chunk.content_complete,
        }
        if chunk.no_interruption_allowed:
            payload["no_interruption_allowed"] = True
        if chunk.end_call:
            payload["end_call"] = True
        if first and chunk.transfer_number:
            payload["transfer_number"] = chunk.transfer_number
        await self._send(payload)

    async def _send_response(self, response_id: int, content: str, *, complete: bool) -> None:
        await self._send(
            {
                "response_type": "response",
                "response_id": response_id,
                "content": content,
                "content_complete": complete,
            }
        )

    async def _send(self, payload: Mapping[str, object]) -> None:
        async with self._out:
            await self._ws.send_json(dict(payload))

    async def _cancel_current(self) -> None:
        task = self._task
        self._task = None
        self._response_id = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return
