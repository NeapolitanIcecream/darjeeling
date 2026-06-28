from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def stable_json(value: Any) -> str:
    def default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"unsupported JSON value: {type(obj).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(path: Path) -> str:
    entries: list[tuple[str, str]] = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and not item.is_symlink():
            entries.append((item.relative_to(path).as_posix(), file_digest(item)))
        elif item.is_symlink():
            entries.append((item.relative_to(path).as_posix(), f"symlink:{os.readlink(item)}"))
    return stable_hash(entries)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def scoped_hash(scope: str, value: str) -> str:
    return hashlib.sha256(f"{scope}\0{value}".encode()).hexdigest()


def bounded_code(value: str, *, max_len: int = 80) -> bool:
    if not value or len(value) > max_len:
        return False
    return all(ch.isalnum() or ch in {"_", "-", "."} for ch in value)


def safe_public_error(error_type: str | None) -> str:
    messages = {
        "deadline_exceeded": "The request deadline was exceeded.",
        "l4_fallback_failure": "The fallback path could not produce a valid response.",
        "no_valid_output": "No valid output was produced.",
        "runtime_error": "The request failed at runtime.",
    }
    return messages.get(error_type or "runtime_error", messages["runtime_error"])
