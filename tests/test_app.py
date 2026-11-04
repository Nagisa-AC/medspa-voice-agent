from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import Settings, create_app
from tests.integrations.test_retell import TOKEN, make_runtime, ws_url


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
