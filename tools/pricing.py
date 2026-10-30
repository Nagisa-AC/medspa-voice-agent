"""quote_price: spoken line for the caller, structured result for the model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType

from core.state import CallState, NumberExtractor, admit_numbers
from core.tenant import Ambiguous, Match, NoMatch, Service, Tenant, resolve_service

RANGE = "{name} typically runs {min} to {max} {unit}."
FLAT = "{name} is {price} {unit}."
UNQUOTABLE = "I don't have a number for {name} over the phone. I can book you a consult."
NO_MATCH = "Let me have someone follow up with an exact number."
AMBIGUOUS = "Which were you asking about, {names}?"


@dataclass(frozen=True)
class QuotePriceResult:
    say: str
    model: Mapping[str, object]


def format_money(value: Decimal) -> str:
    """Dollar string: no trailing .00, two cents digits otherwise, thousands with commas."""
    quantized = value.quantize(Decimal("0.01"))
    if quantized == quantized.to_integral_value():
        return f"${int(quantized):,}"
    cents = int((quantized * 100).to_integral_value())
    dollars, remainder = divmod(abs(cents), 100)
    sign = "-" if cents < 0 else ""
    return f"{sign}${dollars:,}.{remainder:02d}"


def quote_price(
    tenant: Tenant,
    state: CallState,
    service_text: str,
    *,
    extract: NumberExtractor,
) -> QuotePriceResult:
    """Resolve a service and build a spoken quote. Admit the say. Model payload has no prices."""
    resolution = resolve_service(tenant, service_text)
    match resolution:
        case Match(service) if service.quotable:
            say = _spoken_quote(service)
            model = _found_model(service)
        case Match(service):
            say = UNQUOTABLE.format(name=service.name)
            model = _found_model(service)
        case Ambiguous(services):
            say = AMBIGUOUS.format(names=_join_names(tuple(item.name for item in services)))
            model = MappingProxyType(
                {
                    "found": False,
                    "already_spoken": True,
                    "matches": tuple(item.id for item in services),
                }
            )
        case NoMatch():
            say = NO_MATCH
            model = MappingProxyType({"found": False, "already_spoken": True})
    admit_numbers(state, say, extract)
    return QuotePriceResult(say=say, model=model)


def _spoken_quote(service: Service) -> str:
    low = service.price_min
    high = service.price_max
    unit = service.price_unit
    if low is None or high is None or unit is None:
        raise RuntimeError(f"quotable service {service.id} is missing price fields")
    if low == high:
        return FLAT.format(name=service.name, price=format_money(low), unit=unit)
    return RANGE.format(name=service.name, min=format_money(low), max=format_money(high), unit=unit)


def _found_model(service: Service) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "found": True,
            "service_id": service.id,
            "already_spoken": True,
            "consult_required": service.consult_required,
            "deposit_required": service.deposit_required,
        }
    )


def _join_names(names: Sequence[str]) -> str:
    if len(names) == 2:
        return f"{names[0]} or {names[1]}"
    return f"{', '.join(names[:-1])}, or {names[-1]}"
