"""Stable content hashing for Task 3 immutable artefacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def content_hash(*parts: str) -> str:
    """SHA-256 hex digest over concatenated parts."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def hash_model(model: BaseModel, *, exclude: set[str] | None = None) -> str:
    """Hash a Pydantic model dump, optionally excluding volatile fields."""
    data = model.model_dump(mode="json", exclude=exclude or set())
    return content_hash(stable_json_dumps(data))
