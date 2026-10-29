import copy
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from core.tenant import (
    Ambiguous,
    Match,
    NoMatch,
    Resolution,
    Tenant,
    load_tenant_file,
    parse_tenant,
    read_yaml,
    resolve_service,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).parent / "fixtures" / "tenants" / "test_spa.yaml"
TODAY = date(2026, 10, 29)
GLOW = load_tenant_file(ROOT / "tenants" / "glow_aesthetics.yaml", today=TODAY)


def ids(result: Resolution) -> tuple[str, ...] | str | None:
    match result:
        case Match(service):
            return service.id
        case Ambiguous(services):
            return tuple(s.id for s in services)
        case NoMatch():
            return None


GLOW_CASES: list[tuple[str, tuple[str, ...] | str | None]] = [
    ("how much is lip filler", "filler"),
    ("botox and filler", ("tox", "filler")),
    ("the lip thing", None),
    ("Botox", "tox"),
    ("BOTOX?!", "tox"),
    ("dermal filler please", "filler"),
    ("lip-filler", "filler"),
    ("I want lip filler, you know, filler", "filler"),
    ("dysport or xeomin", "tox"),
    ("wrinkle   relaxer", "tox"),
    ("relaxer", None),
    ("loss weight", None),
    ("detox drink", None),
    ("lip fillers", "filler"),
    ("how much are fillers", "filler"),
    ("wrinkle relaxers", "tox"),
    ("botoxx", None),
    ("botox's price", "tox"),
    ("is glp-1 covered", "weightloss"),
    ("semaglutide", "weightloss"),
    ("filler then botox", ("filler", "tox")),
    ("botox, filler, botox", ("tox", "filler")),
    ("botox and weight loss", ("tox", "weightloss")),
    ("not botox, just filler", ("tox", "filler")),
    ("Juvéderm", None),
    ("juvederm", "filler"),
    ("", None),
    ("???", None),
    ("   ", None),
    ("tox", "tox"),
    ("weightloss", None),
]


@pytest.mark.parametrize(
    ("text", "expected"), GLOW_CASES, ids=[c[0] or "<empty>" for c in GLOW_CASES]
)
def test_glow(text: str, expected: tuple[str, ...] | str | None) -> None:
    assert ids(resolve_service(GLOW, text)) == expected


@pytest.mark.parametrize(
    ("service_id", "phrase"),
    [(s.id, p) for s in GLOW.services for p in (s.name, *s.aliases)],
)
def test_every_glow_phrase_resolves_to_its_service(service_id: str, phrase: str) -> None:
    assert ids(resolve_service(GLOW, f"how much is {phrase}")) == service_id


def test_no_match_carries_no_text() -> None:
    assert resolve_service(GLOW, "something private") == NoMatch()


def tenant_with_aliases(**aliases: list[str]) -> Tenant:
    data: Any = copy.deepcopy(read_yaml(BASE))
    for service in data["services"]:
        if service["id"] in aliases:
            service["aliases"] = aliases[service["id"]]
    return parse_tenant(data, source="test_spa.yaml", today=TODAY)


NESTED = tenant_with_aliases(alpha=["lip filler"], beta=["filler"])
OVERLAP = tenant_with_aliases(alpha=["lip filler"], beta=["filler injection"])


@pytest.mark.parametrize(
    ("tenant", "text", "expected"),
    [
        (NESTED, "lip filler", "alpha"),
        (NESTED, "filler", "beta"),
        (NESTED, "lip filler and filler", ("alpha", "beta")),
        (OVERLAP, "lip filler injection", ("alpha", "beta")),
    ],
)
def test_nested_and_overlapping_phrases(
    tenant: Tenant, text: str, expected: tuple[str, ...] | str
) -> None:
    assert ids(resolve_service(tenant, text)) == expected


def test_lashes_does_not_match_lash() -> None:
    # Documented limit: only a trailing "s" is folded, so "lashes" becomes "lashe".
    tenant = tenant_with_aliases(alpha=["lash"])
    assert resolve_service(tenant, "lashes") == NoMatch()
    assert ids(resolve_service(tenant, "lash lift")) == "alpha"
