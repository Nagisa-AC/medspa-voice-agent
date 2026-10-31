from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

import core.prompt
from core.escalation import EMERGENCY_LINE
from core.llm import Message, ToolCall
from core.prompt import (
    TEMPLATE_PATH,
    TOOLS,
    BuiltPrompt,
    PromptFormatError,
    build,
    load_template,
    tool_result_messages,
)
from core.state import AgentLine, CallerLine, CallState, new_call_state
from core.tenant import Tenant, load_tenant_file

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 10, 29)
NOW = datetime(2026, 11, 3, 16, 0, tzinfo=UTC)
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)
SPA = load_tenant_file(
    Path(__file__).parent / "fixtures" / "tenants" / "test_spa.yaml", today=TODAY
)
FIXTURES = Path(__file__).parent / "fixtures" / "prompts"
DEFLECT = (
    "That's a great question for our injector. I can book you a consult, "
    "or have them call you back."
)
FILLER_SAY = "Dermal Filler typically runs $700 to $950 per syringe."
PRICE_LEAKS = ("$13", "$15", "$700", "$950", "per unit", "per syringe")
PHONE = "+12035550148"
EXCEPTION = (
    "Never state a price, a duration, or a medical fact yourself, except exactly as "
    "written in Facts and Policies below."
)
FAQ_MARKER = "[answered from the spa's FAQ]"
PRICE_MARKER = "[price quoted for Dermal Filler]"
HEDGE = "I can help with that."
AVAIL = "I have Wednesday the 4th at 2:00 PM."
BOOK_SAY = (
    "You're all set for Dermal Filler on Wednesday the 4th at 2:00 PM. "
    "Just so you know, there's a $100 deposit for this treatment."
)
TOX_SAY = GLOW.faq[0].say


def fresh(tenant: Tenant = GLOW, **fields: Any) -> CallState:
    state = new_call_state(
        tenant=tenant,
        call_id="call_1",
        data_key_id="key_1",
        now=NOW,
        extract=lambda _text: (),
    )
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def non_system(built: BuiltPrompt) -> str:
    return "\n".join(message.content for message in built.messages if message.role != "system")


def all_content(built: BuiltPrompt) -> str:
    return "\n".join(message.content for message in built.messages)


def as_map(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


# --------------------------------------------------------------------------- snapshot and fill


def test_glow_snapshot_matches_fresh_build() -> None:
    assert build(GLOW, fresh()).dump() == (FIXTURES / "glow.txt").read_text(encoding="utf-8")


def test_glow_system_has_names_and_seeded_policy_not_prices() -> None:
    built = build(GLOW, fresh())
    system = built.messages[0].content
    assert built.messages[0].role == "system"
    assert system.startswith("You are the virtual receptionist for Glow Aesthetics.")
    assert "Botox, Dermal Filler, and Weight Loss Program" in system
    assert "- hours: Monday through Friday 9 to 6, Saturday 10 to 4, closed Sunday." in system
    assert "- address: 1420 Chapel Street, Suite 200, New Haven." in system
    assert "- parking: Free lot behind the building, enter from Howe Street." in system
    assert "- cancellation: 24 hours notice, otherwise there's a $50 fee." in system
    assert "- minors: Patients under 18 need a parent or guardian present." in system
    assert EXCEPTION in " ".join(system.split())
    assert "$50" in system
    for leak in PRICE_LEAKS:
        assert leak not in system
    assert TOX_SAY not in system
    assert EMERGENCY_LINE not in system
    assert DEFLECT not in system
    assert GLOW.callback_promise not in system
    assert GLOW.greeting not in system


def test_template_file_has_no_spoken_backend_lines() -> None:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    assert EXCEPTION in " ".join(text.split())
    for leak in (
        *PRICE_LEAKS,
        TOX_SAY,
        EMERGENCY_LINE,
        DEFLECT,
        GLOW.callback_promise,
        SPA.callback_promise,
        "A test answer.",
    ):
        assert leak not in text


def test_fill_follows_the_tenant() -> None:
    glow = build(GLOW, fresh()).messages[0].content
    spa = build(SPA, fresh(SPA)).messages[0].content
    assert "Glow Aesthetics" in glow
    assert "Test Spa" in spa
    assert "Alpha Treatment, Beta Treatment, and Gamma Program" in spa
    assert "A day's notice, please." in spa
    assert "Glow Aesthetics" not in spa
    assert "$50" not in spa
    assert "Botox" not in spa
    source = Path(core.prompt.__file__).read_text(encoding="utf-8")
    assert "if tenant.id" not in source
    assert "if tenant.id ==" not in source


# --------------------------------------------------------------------------- history tags


def test_price_marker_hides_the_spoken_quote() -> None:
    state = fresh()
    state.transcript.extend(
        [
            CallerLine("how much is lip filler", NOW),
            AgentLine(
                FILLER_SAY,
                "backend",
                NOW,
                history="marker",
                marker="price",
                label="Dermal Filler",
            ),
        ]
    )
    built = build(GLOW, state)
    history = non_system(built)
    assert [m.role for m in built.messages] == ["system", "user", "assistant"]
    assert built.messages[2].content == PRICE_MARKER
    assert PRICE_MARKER in history
    assert FILLER_SAY not in history
    for leak in ("$", "700", "950", "typically"):
        assert leak not in history


def test_faq_marker_hides_the_spoken_answer() -> None:
    state = fresh()
    state.transcript.append(AgentLine(TOX_SAY, "backend", NOW, history="marker", marker="faq"))
    built = build(GLOW, state)
    history = non_system(built)
    assert FAQ_MARKER in history
    assert TOX_SAY not in history
    assert "three to four months" not in history
    assert TOX_SAY not in all_content(built)


def test_include_lines_are_copied() -> None:
    state = fresh()
    state.transcript.extend(
        [
            AgentLine(HEDGE, "backend", NOW),
            AgentLine(DEFLECT, "backend", NOW),
            AgentLine(GLOW.ai_identity_reply, "backend", NOW),
            AgentLine(AVAIL, "backend", NOW),
            AgentLine("Happy to help.", "model", NOW),
        ]
    )
    history = non_system(build(GLOW, state))
    assert HEDGE in history
    assert DEFLECT in history
    assert GLOW.ai_identity_reply in history
    assert AVAIL in history
    assert "Happy to help." in history


def test_omit_hides_book_911_and_callback() -> None:
    state = fresh()
    state.transcript.extend(
        [
            AgentLine(EMERGENCY_LINE, "backend", NOW, history="omit"),
            AgentLine(GLOW.callback_promise, "backend", NOW, history="omit"),
            AgentLine(BOOK_SAY, "backend", NOW, history="omit"),
        ]
    )
    built = build(GLOW, state)
    assert len(built.messages) == 1
    joined = all_content(built)
    assert EMERGENCY_LINE not in joined
    assert GLOW.callback_promise not in joined
    assert BOOK_SAY not in joined
    assert "$100" not in joined


def test_tag_not_substring_hides_an_included_911_line() -> None:
    state = fresh()
    state.transcript.append(AgentLine(EMERGENCY_LINE, "backend", NOW, history="include"))
    assert EMERGENCY_LINE in non_system(build(GLOW, state))


def test_caller_redaction_is_passed_through() -> None:
    state = fresh()
    state.transcript.append(CallerLine("[redacted:clinical_question]", NOW))
    built = build(GLOW, state)
    assert built.messages[1] == Message(role="user", content="[redacted:clinical_question]")


# --------------------------------------------------------------------------- extra / tool results


def test_tool_result_messages_are_model_payload_only() -> None:
    call = ToolCall("c1", "quote_price", {"service_text": "filler"})
    payload = {
        "found": True,
        "service_id": "filler",
        "already_spoken": True,
        "consult_required": False,
        "deposit_required": True,
        "matches": ("tox", "filler"),
    }
    assistant, tool = tool_result_messages(call, payload)
    assert assistant == Message(role="assistant", tool_calls=(call,))
    assert tool.role == "tool"
    assert tool.name == "quote_price"
    assert tool.tool_call_id == "c1"
    data = json.loads(tool.content)
    assert data["matches"] == ["tox", "filler"]
    assert "say" not in data
    assert data["found"] is True
    for leak in ("$", "700", "typically", "say"):
        assert leak not in tool.content


def test_extra_appends_after_history() -> None:
    state = fresh()
    state.transcript.append(CallerLine("how much is filler", NOW))
    call = ToolCall("c1", "quote_price", {"service_text": "filler"})
    extra = tool_result_messages(
        call,
        {"found": True, "service_id": "filler", "already_spoken": True},
    )
    built = build(GLOW, state, extra=extra)
    assert built.messages[1].role == "user"
    assert built.messages[-2:] == extra


# --------------------------------------------------------------------------- schemas


def test_four_schemas_and_book_requires_confirmed_boolean() -> None:
    assert tuple(tool.name for tool in TOOLS) == (
        "quote_price",
        "lookup_faq",
        "check_availability",
        "book",
    )
    built = build(GLOW, fresh())
    assert built.tools is TOOLS
    book = TOOLS[3]
    params = as_map(book.parameters)
    properties = as_map(params["properties"])
    confirmed = as_map(properties["confirmed"])
    assert params["additionalProperties"] is False
    assert confirmed["type"] == "boolean"
    required = params["required"]
    assert isinstance(required, tuple | list)
    assert "confirmed" in required
    assert set(required) == {"slot_start_iso", "patient_name", "patient_phone", "confirmed"}
    avail = as_map(TOOLS[2].parameters)
    avail_props = as_map(avail["properties"])
    earliest = as_map(avail_props["earliest_iso"])
    earliest_type = earliest["type"]
    assert isinstance(earliest_type, tuple | list)
    assert set(earliest_type) == {"string", "null"}
    assert avail_props["limit"] == {"type": "integer", "minimum": 1, "maximum": 3}
    avail_required = avail["required"]
    assert isinstance(avail_required, tuple | list)
    assert "limit" not in avail_required
    for tool in TOOLS:
        assert tool.parameters["additionalProperties"] is False
        for leak in (*PRICE_LEAKS, TOX_SAY, "typically", "$"):
            assert leak not in tool.description


def test_parameters_json_dumps_and_round_trips() -> None:
    for tool in TOOLS:
        payload = tool.parameters_json()
        assert type(payload) is dict
        encoded = json.dumps(payload)
        assert json.loads(encoded) == payload


# --------------------------------------------------------------------------- template load


def test_load_rejects_missing_and_unknown_placeholders(tmp_path: Path) -> None:
    missing = tmp_path / "missing.md"
    missing.write_text("Hello {spa_name} {service_names} {facts}\n")
    with pytest.raises(PromptFormatError, match="missing"):
        load_template(missing)
    extra = tmp_path / "extra.md"
    extra.write_text("Hello {spa_name} {service_names} {facts} {policies} {extra}\n")
    with pytest.raises(PromptFormatError, match="unknown"):
        load_template(extra)


def test_load_rejects_invalid_and_formatted_placeholders(tmp_path: Path) -> None:
    broken = tmp_path / "broken.md"
    broken.write_text("Hello {spa_name\n")
    with pytest.raises(PromptFormatError, match="invalid placeholder"):
        load_template(broken)
    spec = tmp_path / "spec.md"
    spec.write_text("{spa_name:>10} {service_names} {facts} {policies}\n")
    with pytest.raises(PromptFormatError, match="bare"):
        load_template(spec)


# --------------------------------------------------------------------------- privacy and imports


def test_repr_hides_quote_and_phone() -> None:
    state = fresh()
    state.transcript.append(
        AgentLine(
            FILLER_SAY,
            "backend",
            NOW,
            history="marker",
            marker="price",
            label="Dermal Filler",
        )
    )
    extra = tool_result_messages(
        ToolCall("c1", "book", {"patient_phone": PHONE}),
        {"phone": PHONE},
    )
    built = build(GLOW, state, extra=extra)
    shown = repr(built) + str(built)
    assert shown == "BuiltPrompt(messages=4, tools=4)" * 2
    assert FILLER_SAY not in shown
    assert PHONE not in shown
    assert "$700" not in shown
    for message in built.messages:
        text = repr(message) + str(message)
        assert FILLER_SAY not in text
        assert PHONE not in text


def test_prompt_module_imports_are_closed() -> None:
    tree = ast.parse(Path(core.prompt.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {
        "__future__",
        "collections.abc",
        "core.llm",
        "core.state",
        "core.tenant",
        "dataclasses",
        "functools",
        "json",
        "pathlib",
        "string",
        "types",
        "typing",
    }
    assert "tools" not in imported
    assert "core.turn" not in imported
    assert "core.events" not in imported
    assert "core.policy" not in imported
    assert "core.escalation" not in imported
    assert "core.guardrail" not in imported
    assert "retell" not in imported
