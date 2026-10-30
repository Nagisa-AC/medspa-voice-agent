"""lookup_faq: spoken owner-approved answer for the caller, structured result for the model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from core.state import CallState, NumberExtractor, admit_numbers
from core.tenant import FaqEntry, Tenant, phrase_spans


@dataclass(frozen=True)
class LookupFaqResult:
    say: str
    model: Mapping[str, object]


def lookup_faq(
    tenant: Tenant,
    state: CallState,
    question_text: str,
    *,
    deflect: str,
    extract: NumberExtractor,
) -> LookupFaqResult:
    """Resolve one FAQ entry. Admit the say. Model payload has no answer text."""
    if not deflect.strip():
        raise ValueError("deflect must be a non-empty pack line")
    matches = _matched_entries(tenant, question_text)
    if len(matches) == 1:
        entry = matches[0]
        say = entry.say
        model = MappingProxyType({"found": True, "faq_id": entry.id, "already_spoken": True})
    else:
        say = deflect
        model = MappingProxyType({"found": False, "already_spoken": True})
    admit_numbers(state, say, extract)
    return LookupFaqResult(say=say, model=model)


def _matched_entries(tenant: Tenant, question_text: str) -> tuple[FaqEntry, ...]:
    spans = phrase_spans(question_text, [entry.topics for entry in tenant.faq])
    order: list[int] = []
    for _, _, index in spans:
        if index not in order:
            order.append(index)
    return tuple(tenant.faq[i] for i in order)
