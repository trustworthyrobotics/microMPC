"""Project-relative paths for example artifacts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] in {"quadrotor", "rope_manipulation"}:
        path = Path("examples") / path
    return (PROJECT_ROOT / path).resolve()
