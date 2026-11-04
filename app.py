"""Retell Custom LLM websocket server. Does not live in core/."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from retell import AsyncRetell

from core.crypto import InMemoryKeyStore, KeyStore
from core.events import EventDraft, EventLog, InMemoryEventLog, log_ids
from core.guardrail import load_output_pack
from core.llm import Model
from core.numbers import find_numbers
from core.policy import load_policy_engine
from core.state import InMemorySessionStore, SessionStore, new_call_state
from core.tenant import Tenant, TenantRegistry, extra_price_words
from core.turn import TurnLoop
from integrations.booking import MockBookingAdapter
from integrations.llm_anthropic import AnthropicLLM
from integrations.retell import (
    CallLookup,
    CallLookupError,
    CallSession,
    RetellApiLookup,
    check_call,
    require_token,
    token_matches,
    valid_call_id,
)

ROOT = Path(__file__).resolve().parent
_WS_PATH_RE = re.compile(r"/llm-websocket/\S*")
_WS_PATH_REDACTED = "/llm-websocket/<redacted>"
_WS_LOGGERS = ("", "uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
_ENV_KEYS = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_EFFORT",
        "RETELL_WS_TOKEN",
        "RETELL_AGENT_ID",
        "RETELL_API_KEY",
        "BOOKING_SECRET",
        "PORT",
    }
)
_CLOSE_POLICY = 1008
Clock = Callable[[], datetime]


def load_dotenv(path: Path) -> None:
    """Load known keys from a local file. Does not override existing env."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in _ENV_KEYS or name in os.environ:
            continue
        os.environ[name] = value.strip().strip("'").strip('"')


@dataclass(frozen=True)
class Settings:
    ws_token: str
    agent_id: str
    retell_api_key: str
    booking_secret: bytes

    def __repr__(self) -> str:
        return f"Settings(agent_id={self.agent_id!r}, token_len={len(self.ws_token)})"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_env(cls) -> Settings:
        token = require_token(os.environ.get("RETELL_WS_TOKEN", ""))
        agent_id = os.environ.get("RETELL_AGENT_ID", "").strip()
        if not agent_id:
            raise ValueError("RETELL_AGENT_ID is required")
        api_key = os.environ.get("RETELL_API_KEY", "").strip()
        if not api_key:
            raise ValueError("RETELL_API_KEY is required")
        raw_secret = os.environ.get("BOOKING_SECRET", "")
        secret = raw_secret.encode("utf-8") if raw_secret else secrets.token_bytes(32)
        if len(secret) < 32:
            raise ValueError("BOOKING_SECRET must be at least 32 bytes")
        return cls(ws_token=token, agent_id=agent_id, retell_api_key=api_key, booking_secret=secret)


class Runtime:
    def __init__(
        self,
        settings: Settings,
        lookup: CallLookup,
        tenants: TenantRegistry,
        store: SessionStore,
        keys: KeyStore,
        log: EventLog,
        loops: dict[str, TurnLoop],
    ) -> None:
        self.settings = settings
        self.lookup = lookup
        self.tenants = tenants
        self.store = store
        self.keys = keys
        self.log = log
        self.loops = loops
        self.live: dict[str, tuple[asyncio.Lock, CallSession]] = {}
        self._live_guard = asyncio.Lock()

    def __repr__(self) -> str:
        return f"Runtime(agent_id={self.settings.agent_id!r}, tenants={self.tenants.ids})"

    def __str__(self) -> str:
        return repr(self)

    async def claim_lock(self, call_id: str) -> tuple[asyncio.Lock, CallSession | None]:
        async with self._live_guard:
            entry = self.live.get(call_id)
            if entry is None:
                return asyncio.Lock(), None
            return entry[0], entry[1]

    async def register(self, call_id: str, lock: asyncio.Lock, session: CallSession) -> None:
        async with self._live_guard:
            self.live[call_id] = (lock, session)

    async def drop(self, call_id: str, session: CallSession) -> None:
        async with self._live_guard:
            entry = self.live.get(call_id)
            if entry is not None and entry[1] is session:
                del self.live[call_id]

    @classmethod
    def from_env(cls) -> Runtime:
        settings = Settings.from_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is required")
        today = datetime.now(UTC).date()
        tenants = TenantRegistry.load(ROOT / "tenants", today=today)
        keys = InMemoryKeyStore()
        log = InMemoryEventLog(keys)
        store = InMemorySessionStore()
        client = AsyncRetell(api_key=settings.retell_api_key)
        lookup: CallLookup = RetellApiLookup(client)
        model: Model = AnthropicLLM()
        loops = {
            tenant.id: _build_loop(tenant, store, log, model, settings.booking_secret)
            for tenant in (tenants.get(tid) for tid in tenants.ids)
            if tenant is not None
        }
        return cls(settings, lookup, tenants, store, keys, log, loops)


def _build_loop(
    tenant: Tenant,
    store: SessionStore,
    log: EventLog,
    model: Model,
    secret: bytes,
    *,
    clock: Clock | None = None,
) -> TurnLoop:
    tz = tenant.tz

    def now() -> datetime:
        return datetime.now(tz)

    words = extra_price_words(tenant)
    extract = partial(
        find_numbers,
        extra_price_words=words.extra_price_words,
        service_words=words.service_words,
    )
    return TurnLoop(
        tenant,
        store,
        log,
        model,
        MockBookingAdapter.from_yaml(
            ROOT / "tenants" / f"{tenant.id}.mock.yaml", tz=tenant.tz, clock=clock or now
        ),
        secret,
        clock=clock or now,
        engine=load_policy_engine(tenant),
        pack=load_output_pack(ROOT / "packs" / "medspa.output.yaml"),
        extract=extract,
    )


class _RedactWsPathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_ws_path(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(_redact_ws_path(arg) for arg in args)
        elif isinstance(args, Mapping):
            record.args = {str(key): _redact_ws_path(val) for key, val in args.items()}
        return True


_WS_PATH_FILTER = _RedactWsPathFilter()


def _redact_ws_path(value: object) -> object:
    if isinstance(value, str):
        return _WS_PATH_RE.sub(_WS_PATH_REDACTED, value)
    return value


def install_ws_path_filter() -> None:
    for name in _WS_LOGGERS:
        logger = logging.getLogger(name)
        if _WS_PATH_FILTER not in logger.filters:
            logger.addFilter(_WS_PATH_FILTER)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    install_ws_path_filter()
    yield


def create_app(runtime: Runtime | None = None) -> FastAPI:
    if runtime is None:
        runtime = Runtime.from_env()
    install_ws_path_filter()
    app = FastAPI(lifespan=_lifespan)
    app.state.runtime = runtime

    @app.get("/health")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/llm-websocket/{token}/{tenant_id}/{call_id}")
    async def llm_websocket(websocket: WebSocket, token: str, tenant_id: str, call_id: str) -> None:
        await _handle_socket(websocket, runtime, token, tenant_id, call_id)

    return app


async def _handle_socket(
    websocket: WebSocket,
    runtime: Runtime,
    token: str,
    tenant_id: str,
    call_id: str,
) -> None:
    if not valid_call_id(call_id):
        log_ids("ws_reject", reason="invalid")
        await websocket.close(code=_CLOSE_POLICY)
        return
    if not token_matches(token, runtime.settings.ws_token):
        await _reject(websocket, call_id=call_id, reason="token")
        return
    tenant = runtime.tenants.get(tenant_id)
    if tenant is None:
        await _reject(websocket, call_id=call_id, reason="tenant")
        return
    loop = runtime.loops.get(tenant.id)
    if loop is None:
        await _reject(websocket, call_id=call_id, reason="tenant")
        return
    try:
        info = await runtime.lookup.get_call(call_id)
        check_call(info, runtime.settings.agent_id)
    except CallLookupError as exc:
        await _reject(websocket, call_id=call_id, reason=exc.reason)
        return
    await websocket.accept()
    log_ids("ws_open", call_id=call_id, tenant_id=tenant.id)
    lock, previous = await runtime.claim_lock(call_id)
    if previous is not None:
        await previous.supersede()
    stored = await runtime.store.get(call_id)
    reconnect = stored is not None
    if stored is None:
        data_key_id = await runtime.keys.create(call_id)
        state = new_call_state(
            tenant=tenant,
            call_id=call_id,
            data_key_id=data_key_id,
            now=loop.clock(),
            extract=loop.extract,
        )
        if info.from_number:
            state.caller_number = info.from_number
        await runtime.store.create(state)
        payload: dict[str, str | int | bool | list[str]] = {}
        if state.caller_number:
            payload["caller_number"] = state.caller_number
        await runtime.log.append(
            EventDraft(
                call_id=call_id,
                tenant_id=tenant.id,
                at=loop.clock(),
                kind="call_started",
                payload=payload,
            )
        )
    else:
        state = stored
    session = CallSession(
        websocket,
        loop=loop,
        state=state,
        lock=lock,
        call_id=call_id,
        tenant_id=tenant.id,
    )
    await runtime.register(call_id, lock, session)
    try:
        await session.run(reconnect=reconnect)
    finally:
        await runtime.drop(call_id, session)
        log_ids("ws_close", call_id=call_id, tenant_id=tenant.id)


async def _reject(websocket: WebSocket, *, call_id: str, reason: str) -> None:
    log_ids("ws_reject", call_id=call_id, reason=reason)
    await websocket.close(code=_CLOSE_POLICY)


def serve(app: FastAPI, *, host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port, access_log=False)


def main() -> None:
    load_dotenv(ROOT / ".env")
    app = create_app()
    port_raw = os.environ.get("PORT", "8000")
    port = int(port_raw) if port_raw.isdigit() else 8000
    serve(app, port=port)


if __name__ == "__main__":
    main()
