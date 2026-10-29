"""Per-call AES-GCM keys. Key material never leaves the store."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Protocol, final

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_BITS = 256
NONCE_BYTES = 12
_AAD_VERSION = "v1"
_AAD_SEP = "\x1f"


class CallKeyExistsError(Exception):
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(f"call {call_id!r} already has a data key")


class UnknownCallKeyError(Exception):
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(f"no data key for call {call_id!r}")


class ShreddedKeyError(Exception):
    def __init__(self, call_id: str) -> None:
        self.call_id = call_id
        super().__init__(f"data key for call {call_id!r} has been shredded")


class InvalidCiphertextError(Exception):
    """Tampered ciphertext, wrong nonce, or associated data that does not bind."""


@dataclass(frozen=True)
class Shredded:
    """A sensitive field whose key is gone. Reports must survive this."""


@dataclass(frozen=True)
class Ciphertext:
    key_id: str
    nonce: bytes
    data: bytes

    def __post_init__(self) -> None:
        if len(self.nonce) != NONCE_BYTES:
            raise ValueError("nonce must be 12 bytes")

    def __repr__(self) -> str:
        return (
            f"Ciphertext(key_id={self.key_id!r}, nonce={len(self.nonce)}B, data={len(self.data)}B)"
        )

    def __str__(self) -> str:
        return repr(self)


def bind_aad(*, tenant_id: str, call_id: str, kind: str, field: str) -> bytes:
    """Bind a blob to one tenant, call, kind and field. Parts must not contain the separator."""
    parts = (_AAD_VERSION, tenant_id, call_id, kind, field)
    if any(_AAD_SEP in part for part in parts):
        raise ValueError("associated-data part contains the separator")
    return _AAD_SEP.join(parts).encode("utf-8")


class KeyStore(Protocol):
    async def create(self, call_id: str) -> str:
        """Mint a data key. Returns an opaque key id. Raises if the call already has one."""
        ...

    async def shred(self, call_id: str) -> None:
        """Drop the call's key. Idempotent."""
        ...

    async def encrypt(self, call_id: str, plaintext: bytes, *, aad: bytes) -> Ciphertext: ...

    async def decrypt(
        self, call_id: str, ciphertext: Ciphertext, *, aad: bytes
    ) -> bytes | Shredded: ...


class _KeyEntry:
    __slots__ = ("key_id", "material")

    def __init__(self, key_id: str, material: bytearray) -> None:
        self.key_id = key_id
        self.material = material

    def __repr__(self) -> str:
        return f"_KeyEntry(key_id={self.key_id!r})"


@final
class InMemoryKeyStore:
    def __init__(self) -> None:
        self._live: dict[str, _KeyEntry] = {}
        self._shredded_calls: set[str] = set()
        self._shredded_key_ids: set[str] = set()

    def __repr__(self) -> str:
        return (
            f"InMemoryKeyStore(live={sorted(self._live)}, shredded={sorted(self._shredded_calls)})"
        )

    async def create(self, call_id: str) -> str:
        if not call_id:
            raise ValueError("call_id must not be empty")
        if call_id in self._live or call_id in self._shredded_calls:
            raise CallKeyExistsError(call_id)
        key_id = uuid.uuid4().hex
        material = bytearray(AESGCM.generate_key(bit_length=KEY_BITS))
        self._live[call_id] = _KeyEntry(key_id, material)
        return key_id

    async def shred(self, call_id: str) -> None:
        # Best-effort only: CPython may keep copies of key bytes. This in-memory store
        # is not a KMS; real deletion is dropping the key from the later durable store.
        entry = self._live.pop(call_id, None)
        if entry is None:
            return
        for i in range(len(entry.material)):
            entry.material[i] = 0
        self._shredded_calls.add(call_id)
        self._shredded_key_ids.add(entry.key_id)

    async def encrypt(self, call_id: str, plaintext: bytes, *, aad: bytes) -> Ciphertext:
        entry = self._require_live(call_id)
        nonce = os.urandom(NONCE_BYTES)
        data = AESGCM(bytes(entry.material)).encrypt(nonce, plaintext, aad)
        return Ciphertext(key_id=entry.key_id, nonce=nonce, data=data)

    async def decrypt(
        self, call_id: str, ciphertext: Ciphertext, *, aad: bytes
    ) -> bytes | Shredded:
        if call_id in self._shredded_calls:
            return Shredded()
        entry = self._require_live(call_id)
        if ciphertext.key_id != entry.key_id:
            raise InvalidCiphertextError(call_id)
        if len(ciphertext.nonce) != NONCE_BYTES:
            raise InvalidCiphertextError(call_id)
        try:
            return AESGCM(bytes(entry.material)).decrypt(ciphertext.nonce, ciphertext.data, aad)
        except InvalidTag as exc:
            raise InvalidCiphertextError(call_id) from exc

    def _require_live(self, call_id: str) -> _KeyEntry:
        if call_id in self._shredded_calls:
            raise ShreddedKeyError(call_id)
        entry = self._live.get(call_id)
        if entry is None:
            raise UnknownCallKeyError(call_id)
        return entry
