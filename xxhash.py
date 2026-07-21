from __future__ import annotations

from hashlib import blake2b


class _XXH3_128:
    """Compat shim para ambientes Windows onde o módulo nativo xxhash é bloqueado.

    O LangSmith usa apenas o método ``digest()`` para gerar um valor determinístico
    em ``uuid7_deterministic``. Para manter o comportamento sem depender do módulo
    binário ``_xxhash``, usamos um hash Python puro e estável.
    """

    def __init__(self, payload: bytes) -> None:
        self._digest = blake2b(payload, digest_size=16).digest()

    def digest(self) -> bytes:
        return self._digest


def xxh3_128(payload: bytes) -> _XXH3_128:
    return _XXH3_128(payload)
