from __future__ import annotations

import hashlib
import sys
import types


try:
    import xxhash as _xxhash  # type: ignore
except Exception:
    # Windows sometimes blocks the native xxhash binary (.pyd/.dll) even when the
    # Python package is otherwise installed. LangSmith only needs xxhash for a
    # deterministic UUID helper. For local development we expose a tiny fallback
    # implementation with the same public API used by LangSmith.
    stub = types.ModuleType("xxhash")

    class _XXH3_128:
        def __init__(self, payload: bytes) -> None:
            self._digest = hashlib.blake2b(payload, digest_size=16).digest()

        def digest(self) -> bytes:
            return self._digest

    def xxh3_128(payload: bytes) -> _XXH3_128:
        return _XXH3_128(payload)

    stub.xxh3_128 = xxh3_128
    stub.__version__ = "0.0.0-fallback"
    sys.modules["xxhash"] = stub
