import os
import sys
from pathlib import Path
from typing import Optional

_configured_paths = set()


def _normalize_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def configure_vagen_path(path: Optional[str]) -> None:
    """Register the VAGEN path (from config) once and set env var."""
    if not path:
        return
    normalized = _normalize_path(path)
    if normalized not in _configured_paths:
        _configured_paths.add(normalized)
        if normalized not in sys.path:
            sys.path.insert(0, normalized)
    os.environ.setdefault("VAGEN_PATH", normalized)


def ensure_vagen_path_from_env() -> Optional[str]:
    """Ensure the VAGEN path from env is on sys.path."""
    path = os.environ.get("VAGEN_PATH")
    if not path:
        return None
    normalized = _normalize_path(path)
    if normalized not in sys.path:
        sys.path.insert(0, normalized)
    return normalized

