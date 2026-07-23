from __future__ import annotations

from hashlib import blake2b


class _XXH3_128:
    """Compat shim para ambientes Windows onde o módulo nativo xxhash é bloqueado.

    Mantém um comportamento determinístico e omite a dependência do módulo binário
    ``_xxhash`` usado por alguns pacotes. Para suportar o contrato de `datasets`,
    implementamos também um `xxh64` que devolve um objeto com `digest()`.
    """

    def __init__(self, payload: bytes) -> None:
        self._digest = blake2b(payload, digest_size=16).digest()

    def digest(self) -> bytes:
        return self._digest


class _XXH64:
    def __init__(self, payload: bytes = b"") -> None:
        self._buffer = bytearray(payload)

    def update(self, payload: bytes) -> None:
        self._buffer.extend(payload)

    def digest(self) -> bytes:
        return blake2b(bytes(self._buffer), digest_size=8).digest()

    def hexdigest(self) -> str:
        return self.digest().hex()


def xxh3_128(payload: bytes) -> _XXH3_128:
    return _XXH3_128(payload)


def xxh64(payload: bytes = b"") -> _XXH64:
    return _XXH64(payload)
