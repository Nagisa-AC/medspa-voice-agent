import base64
import traceback

import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.crypto import (
    KEY_BITS,
    NONCE_BYTES,
    CallKeyExistsError,
    Ciphertext,
    InMemoryKeyStore,
    InvalidCiphertextError,
    Shredded,
    ShreddedKeyError,
    UnknownCallKeyError,
    bind_aad,
)

KNOWN_KEY = b"\x11" * 32


def aad(
    *,
    tenant_id: str = "tenant_a",
    call_id: str = "call_1",
    kind: str = "caller_utterance",
    field: str = "text",
) -> bytes:
    return bind_aad(tenant_id=tenant_id, call_id=call_id, kind=kind, field=field)


async def live_store(call_id: str = "call_1") -> InMemoryKeyStore:
    store = InMemoryKeyStore()
    await store.create(call_id)
    return store


async def test_create_returns_opaque_id_not_equal_to_call_id() -> None:
    store = InMemoryKeyStore()
    key_id = await store.create("call_1")
    assert key_id
    assert key_id != "call_1"
    assert len(key_id) == 32


async def test_create_uses_256_bit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    real = AESGCM.generate_key

    def spy(*, bit_length: int) -> bytes:
        seen.append(bit_length)
        return real(bit_length=bit_length)

    monkeypatch.setattr(AESGCM, "generate_key", staticmethod(spy))
    store = InMemoryKeyStore()
    await store.create("call_1")
    assert seen == [KEY_BITS]


async def test_duplicate_create_raises() -> None:
    store = await live_store()
    with pytest.raises(CallKeyExistsError) as info:
        await store.create("call_1")
    assert info.value.call_id == "call_1"


async def test_round_trip_encrypt_decrypt() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    assert len(blob.nonce) == NONCE_BYTES
    assert await store.decrypt("call_1", blob, aad=aad()) == b"hello"


async def test_nonce_is_random_per_encrypt() -> None:
    store = await live_store()
    first = await store.encrypt("call_1", b"same", aad=aad())
    second = await store.encrypt("call_1", b"same", aad=aad())
    assert first.nonce != second.nonce
    assert first.data != second.data


async def test_wrong_associated_data_fails() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    mismatches = [
        aad(call_id="call_2"),
        aad(kind="agent_sentence"),
        aad(field="patient_name"),
        aad(tenant_id="tenant_b"),
    ]
    for other in mismatches:
        with pytest.raises(InvalidCiphertextError):
            await store.decrypt("call_1", blob, aad=other)


async def test_ciphertext_moved_to_another_tenant_fails() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad(tenant_id="tenant_a"))
    with pytest.raises(InvalidCiphertextError):
        await store.decrypt("call_1", blob, aad=aad(tenant_id="tenant_b"))


async def test_tampered_ciphertext_fails() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    flipped = blob.data[:-1] + bytes([blob.data[-1] ^ 1])
    tampered = Ciphertext(key_id=blob.key_id, nonce=blob.nonce, data=flipped)
    with pytest.raises(InvalidCiphertextError):
        await store.decrypt("call_1", tampered, aad=aad())


async def test_shred_returns_marker_not_exception() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    await store.shred("call_1")
    assert await store.decrypt("call_1", blob, aad=aad()) == Shredded()
    with pytest.raises(ShreddedKeyError):
        await store.encrypt("call_1", b"hello", aad=aad())


async def test_shred_is_idempotent() -> None:
    store = await live_store()
    await store.shred("call_1")
    await store.shred("call_1")
    await store.shred("missing")


async def test_decrypt_unknown_call_raises() -> None:
    store = InMemoryKeyStore()
    blob = Ciphertext(key_id="kid", nonce=b"\x00" * 12, data=b"\x00" * 16)
    with pytest.raises(UnknownCallKeyError):
        await store.decrypt("nope", blob, aad=aad())


async def test_create_after_shred_raises() -> None:
    store = await live_store()
    await store.shred("call_1")
    with pytest.raises(CallKeyExistsError):
        await store.create("call_1")


async def test_key_material_stays_out_of_reprs_and_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        AESGCM,
        "generate_key",
        staticmethod(lambda *, bit_length: KNOWN_KEY),
    )
    store = InMemoryKeyStore()
    await store.create("call_1")
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    try:
        await store.create("call_1")
    except CallKeyExistsError as exc:
        text = "".join(traceback.format_exception(exc)) + repr(store) + str(exc) + repr(blob)
    else:
        raise AssertionError("expected CallKeyExistsError")
    assert KNOWN_KEY not in text.encode("utf-8")
    assert KNOWN_KEY.hex() not in text
    assert base64.b64encode(KNOWN_KEY).decode("ascii") not in text
    assert "InMemoryKeyStore" in repr(store)
    assert "call_1" in repr(store)


async def test_ciphertext_repr_has_no_bytes_payload() -> None:
    store = await live_store()
    blob = await store.encrypt("call_1", b"hello", aad=aad())
    text = repr(blob)
    assert blob.nonce.hex() not in text
    assert base64.b64encode(blob.data).decode("ascii") not in text


async def test_bind_aad_rejects_separator() -> None:
    with pytest.raises(ValueError, match="separator"):
        bind_aad(tenant_id="a\x1fb", call_id="c", kind="k", field="f")
