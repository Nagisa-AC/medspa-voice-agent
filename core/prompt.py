"""Build the model prompt from the tenant template and the call transcript.

Does not call the model, run tools, or speak. Task 19 owns the turn loop, events,
say-direct, and when a line is appended. A model sentence blocked by the
guardrail is never appended to the transcript; only its replacement is.

History tags (set at write time, read here; the builder never re-parses spoken text):

- quote_price say → marker, kind=price, label=service.name (or "a treatment")
- lookup_faq say → marker, kind=faq
- check_availability say, hedge, clinical deflect, identity reply → include
- book say, 911, callback promise → omit
- greeting → include or omit (not required in history)
- model sentence that passed the guardrail → include
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Final

from core.llm import Message, ToolCall, ToolSchema
from core.state import AgentLine, CallerLine, CallState
from core.tenant import Tenant

TEMPLATE_PATH: Final = Path(__file__).resolve().parent.parent / "prompts" / "medspa.md"
_REQUIRED = frozenset({"spa_name", "service_names", "facts", "policies"})
_PRICE_MARKER = "[price quoted for {label}]"
_FAQ_MARKER = "[answered from the spa's FAQ]"


class PromptFormatError(Exception):
    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


# --------------------------------------------------------------------------- tools


_QUOTE_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": MappingProxyType({"service_text": MappingProxyType({"type": "string"})}),
        "required": ("service_text",),
        "additionalProperties": False,
    }
)
_FAQ_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": MappingProxyType({"question_text": MappingProxyType({"type": "string"})}),
        "required": ("question_text",),
        "additionalProperties": False,
    }
)
_AVAIL_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": MappingProxyType(
            {
                "service_text": MappingProxyType({"type": "string"}),
                "earliest_iso": MappingProxyType({"type": ("string", "null")}),
                "limit": MappingProxyType({"type": "integer", "minimum": 1, "maximum": 3}),
            }
        ),
        "required": ("service_text",),
        "additionalProperties": False,
    }
)
_BOOK_PARAMS: Mapping[str, object] = MappingProxyType(
    {
        "type": "object",
        "properties": MappingProxyType(
            {
                "slot_start_iso": MappingProxyType({"type": "string"}),
                "patient_name": MappingProxyType({"type": "string"}),
                "patient_phone": MappingProxyType({"type": "string"}),
                "confirmed": MappingProxyType({"type": "boolean"}),
            }
        ),
        "required": ("slot_start_iso", "patient_name", "patient_phone", "confirmed"),
        "additionalProperties": False,
    }
)

TOOLS: tuple[ToolSchema, ...] = (
    ToolSchema(
        name="quote_price",
        description=(
            "Look up what a treatment costs. Call this instead of stating a price. "
            "The spoken line is delivered automatically."
        ),
        parameters=_QUOTE_PARAMS,
    ),
    ToolSchema(
        name="lookup_faq",
        description=(
            "Answer from the spa's approved FAQ. Call this for how long something "
            "lasts, pain, recovery, downtime, results, or anything about the "
            "caller's body or health."
        ),
        parameters=_FAQ_PARAMS,
    ),
    ToolSchema(
        name="check_availability",
        description=("List open appointment times. Offer only the times this tool returns."),
        parameters=_AVAIL_PARAMS,
    ),
    ToolSchema(
        name="book",
        description=(
            "Book a held time after the caller says yes. Read back the treatment, "
            "the day, and the time first. confirmed must be true."
        ),
        parameters=_BOOK_PARAMS,
    ),
)


# --------------------------------------------------------------------------- prompt


@dataclass(frozen=True)
class BuiltPrompt:
    messages: tuple[Message, ...]
    tools: tuple[ToolSchema, ...]

    def dump(self) -> str:
        system = self.messages[0].content.rstrip("\n")
        names = "\n".join(tool.name for tool in self.tools)
        return f"### system\n{system}\n\n### tools\n{names}\n"

    def __repr__(self) -> str:
        return f"BuiltPrompt(messages={len(self.messages)}, tools={len(self.tools)})"

    def __str__(self) -> str:
        return repr(self)


@cache
def load_template(path: Path) -> str:
    source = str(path)
    text = path.read_text(encoding="utf-8")
    names = _placeholder_names(text, source)
    if names != _REQUIRED:
        missing = sorted(_REQUIRED - names)
        extra = sorted(names - _REQUIRED)
        parts: list[str] = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unknown {extra}")
        raise PromptFormatError(source, "; ".join(parts))
    return text


def build(
    tenant: Tenant,
    state: CallState,
    *,
    extra: Sequence[Message] = (),
    template: str | None = None,
    template_path: Path | None = None,
) -> BuiltPrompt:
    text = template if template is not None else load_template(template_path or TEMPLATE_PATH)
    system = _fill(text, tenant)
    return BuiltPrompt(
        messages=(Message(role="system", content=system), *_history(state), *extra),
        tools=TOOLS,
    )


def tool_result_messages(call: ToolCall, payload: Mapping[str, object]) -> tuple[Message, Message]:
    content = json.dumps(_jsonable(payload), sort_keys=True, default=str)
    return (
        Message(role="assistant", tool_calls=(call,)),
        Message(role="tool", name=call.name, tool_call_id=call.id, content=content),
    )


def _history(state: CallState) -> tuple[Message, ...]:
    messages: list[Message] = []
    for line in state.transcript:
        if isinstance(line, CallerLine):
            messages.append(Message(role="user", content=line.text))
            continue
        content = _agent_content(line)
        if content is not None:
            messages.append(Message(role="assistant", content=content))
    return tuple(messages)


def _agent_content(line: AgentLine) -> str | None:
    if line.history == "omit":
        return None
    if line.history == "include":
        return line.text
    if line.marker == "price":
        return _PRICE_MARKER.format(label=line.label)
    return _FAQ_MARKER


def _fill(template: str, tenant: Tenant) -> str:
    return template.format(
        spa_name=tenant.name,
        service_names=_english_list(tuple(service.name for service in tenant.services)),
        facts=_bullets(tenant.facts),
        policies=_bullets(tenant.policies),
    )


def _english_list(names: Sequence[str]) -> str:
    if not names:
        raise ValueError("need at least one service name")
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _bullets(items: Mapping[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in items.items())


def _placeholder_names(text: str, source: str) -> set[str]:
    try:
        parsed = tuple(Formatter().parse(text))
    except ValueError as exc:
        raise PromptFormatError(source, f"invalid placeholder ({exc})") from None
    names: set[str] = set()
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if spec or conversion:
            raise PromptFormatError(source, f"placeholder {name!r} must be bare")
        if not name:
            raise PromptFormatError(source, "empty placeholder")
        names.add(name)
    return names


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value
