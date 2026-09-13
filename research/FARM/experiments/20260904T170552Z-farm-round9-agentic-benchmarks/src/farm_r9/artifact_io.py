"""Atomic artifact I/O, hashing, and secret-leak checks."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|bearer|password|passwd|secret|access[_-]?token)",
    re.IGNORECASE,
)
_LIKELY_SECRET_VALUE = re.compile(r"(?:[A-Za-z0-9_-]{24,}\.[A-Za-z0-9_-]{16,}|Bearer\s+\S+)")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record is not an object")
            rows.append(value)
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json_atomic(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")
    _atomic_write(path, payload)


def assert_no_secrets(value: Any, *, path: str = "$") -> None:
    """Reject credential-shaped keys/values before an artifact is persisted.

    Opaque references such as ``secret_ref: account_alias`` are explicitly
    allowed; their values name a vault entry rather than containing a secret.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _SECRET_KEY.search(key_text) and key_text not in {
                "secret_ref", "secret_reference", "security_violation"
            }:
                if child not in (None, "", False, "[REDACTED]"):
                    raise ValueError(f"credential-bearing field forbidden at {path}.{key_text}")
            assert_no_secrets(child, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_secrets(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and _LIKELY_SECRET_VALUE.search(value):
        raise ValueError(f"credential-shaped value forbidden at {path}")


def ordered_ids_sha256(rows: Iterable[Mapping[str, Any]], id_key: str = "case_id") -> str:
    identifiers = []
    for row in rows:
        value = row.get(id_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing nonempty {id_key}")
        identifiers.append(value)
    return sha256_text("\n".join(identifiers) + "\n")
