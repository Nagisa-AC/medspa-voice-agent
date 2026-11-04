from __future__ import annotations

import ast
import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from websockets.asyncio.client import connect

from app import Settings, create_app, install_ws_path_filter
from tests.integrations.test_retell import CALL, TENANT, TOKEN, make_runtime, ws_url

ROOT = Path(__file__).resolve().parents[1]


def test_health_ok() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_missing_ws_token_refuses_to_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETELL_WS_TOKEN", raising=False)
    monkeypatch.setenv("RETELL_AGENT_ID", "agent")
    monkeypatch.setenv("RETELL_API_KEY", "k" * 32)
    with pytest.raises(ValueError, match="RETELL_WS_TOKEN"):
        Settings.from_env()


def test_short_ws_token_refuses_to_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETELL_WS_TOKEN", "short")
    monkeypatch.setenv("RETELL_AGENT_ID", "agent")
    monkeypatch.setenv("RETELL_API_KEY", "k" * 32)
    with pytest.raises(ValueError, match="32"):
        Settings.from_env()


def test_missing_agent_id_refuses_to_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETELL_WS_TOKEN", TOKEN)
    monkeypatch.delenv("RETELL_AGENT_ID", raising=False)
    monkeypatch.setenv("RETELL_API_KEY", "k" * 32)
    with pytest.raises(ValueError, match="RETELL_AGENT_ID"):
        Settings.from_env()


def test_missing_api_key_refuses_to_listen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETELL_WS_TOKEN", TOKEN)
    monkeypatch.setenv("RETELL_AGENT_ID", "agent")
    monkeypatch.delenv("RETELL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="RETELL_API_KEY"):
        Settings.from_env()


def test_settings_repr_hides_secrets() -> None:
    settings = Settings(
        ws_token=TOKEN, agent_id="agent_glow", retell_api_key="secret-key", booking_secret=b"n" * 32
    )
    assert TOKEN not in repr(settings)
    assert "secret-key" not in repr(settings)
    assert "secret-key" not in str(settings)


def test_serve_disables_access_log() -> None:
    source = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(source):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "access_log" and isinstance(keyword.value, ast.Constant):
                found = keyword.value.value is False
    assert found


def test_core_does_not_import_retell_fastapi_or_app() -> None:
    forbidden = {"retell", "fastapi", "app", "uvicorn", "integrations.retell"}
    imported: set[str] = set()
    for path in Path("core").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)


def test_env_example_has_retell_placeholders() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")
    assert "RETELL_WS_TOKEN=" in text
    assert "RETELL_AGENT_ID=" in text
    assert "RETELL_API_KEY=" in text
    assert "sk-" not in text


def test_health_does_not_require_websocket() -> None:
    runtime, _lookup = make_runtime()
    client = TestClient(create_app(runtime))
    assert client.get("/health").json() == {"ok": True}
    assert ws_url().startswith("/llm-websocket/")


def test_ws_path_filter_redacts_token_in_record_args() -> None:
    install_ws_path_filter()
    path = f"/llm-websocket/{TOKEN}/glow_aesthetics/{CALL}"
    record = logging.LogRecord(
        name="uvicorn.error",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "WebSocket %s" [accepted]',
        args=("127.0.0.1:9", path),
        exc_info=None,
    )
    assert logging.getLogger("uvicorn.error").filter(record)
    message = record.getMessage()
    assert TOKEN not in message
    assert "/llm-websocket/<redacted>" in message
    assert "[accepted]" in message


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(port: int, proc: subprocess.Popen[str]) -> None:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(50):
        if proc.poll() is not None:
            out, _ = proc.communicate()
            raise AssertionError(f"uvicorn exited {proc.returncode}: {out}")
        try:
            with urllib.request.urlopen(url, timeout=0.2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(0.1)
    proc.kill()
    out, _ = proc.communicate()
    raise AssertionError(f"health never came up: {out}")


def test_uvicorn_accept_log_never_contains_ws_token() -> None:
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "from app import create_app; "
                "from tests.integrations.test_retell import make_runtime; "
                "import sys, uvicorn; "
                "uvicorn.run(create_app(make_runtime()[0]), host='127.0.0.1', "
                "port=int(sys.argv[1]), access_log=False, log_level='info')"
            ),
            str(port),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_health(port, proc)

        async def _hit() -> None:
            uri = f"ws://127.0.0.1:{port}/llm-websocket/{TOKEN}/{TENANT}/{CALL}"
            async with connect(uri, open_timeout=5):
                await asyncio.sleep(0.3)

        asyncio.run(_hit())
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
    assert TOKEN not in out, out
    assert "accepted" in out.lower(), out
