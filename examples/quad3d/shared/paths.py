"""Resolve project and example artifacts independently of the working directory."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"quad3d", "rope"}:
        return (PROJECT_ROOT / "examples" / path).resolve()
    return (EXAMPLE_ROOT / path).resolve()
