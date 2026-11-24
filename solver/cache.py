"""Atomic storage for deterministic NumPy solver caches."""

import os
import tempfile
from pathlib import Path

import numpy as np


def save_array_cache(path: Path, **arrays) -> None:
    """Replace a cache atomically, including when multiple processes build it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
