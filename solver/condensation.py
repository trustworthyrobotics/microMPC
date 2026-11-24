"""Affine prediction maps and cached condensed proximal systems (NumPy only)."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import save_array_cache

PROJECTED_MAP_DTYPE = np.float64
CACHE_SCHEMA_VERSION = 1
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "solver"


def _hash_arrays(*arrays):
    digest = hashlib.sha256()
    digest.update(f"u-kkt-pd-cache-v{CACHE_SCHEMA_VERSION}".encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array))
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:20]


def _read_metadata(npz_file):
    return json.loads(str(npz_file["metadata"].item()))


def build_prediction_maps(A, B, C, c, horizon):
    """Build physical affine prediction maps in cache precision.

    x_1,...,x_H = obs_x z_0 + c + gamma_x U. Full latent trajectory
    matrices are deliberately never materialized.
    """
    A = np.asarray(A, dtype=PROJECTED_MAP_DTYPE)
    B = np.asarray(B, dtype=PROJECTED_MAP_DTYPE)
    C = np.asarray(C, dtype=PROJECTED_MAP_DTYPE)
    c = np.asarray(c, dtype=PROJECTED_MAP_DTYPE)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.ndim != 2 or B.shape[0] != A.shape[0] or B.shape[1] < 1:
        raise ValueError("B must have shape [latent_dim, control_dim]")
    if C.ndim != 2 or C.shape[1] != A.shape[0] or c.shape != (C.shape[0],):
        raise ValueError("decoder dimensions do not match the dynamics")
    if not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not all(np.isfinite(value).all() for value in (A, B, C, c)):
        raise ValueError("dynamics and decoder must be finite")
    nx, nz = C.shape
    nu = B.shape[1]

    obs_x = np.empty((horizon, nx, nz), dtype=PROJECTED_MAP_DTYPE)
    markov_x = np.empty((horizon, nx, nu), dtype=PROJECTED_MAP_DTYPE)

    ca_power = C @ A
    a_power_b = B.copy()
    for step in range(horizon):
        obs_x[step] = ca_power
        markov_x[step] = C @ a_power_b
        ca_power = ca_power @ A
        a_power_b = A @ a_power_b

    gamma_x = np.zeros((horizon, nx, horizon, nu), dtype=PROJECTED_MAP_DTYPE)
    for state_step in range(horizon):
        for control_step in range(state_step + 1):
            gamma_x[state_step, :, control_step, :] = markov_x[
                state_step - control_step
            ]

    return {
        "obs_x": obs_x,
        "gamma_x": gamma_x,
        "decoder_bias": c,
    }


def load_or_build_prediction_maps(A, B, C, c, horizon, cache_dir=CACHE_DIR):
    """Load any compatible cache with at least ``horizon`` steps."""
    model_key = _hash_arrays(
        np.asarray(A, dtype=PROJECTED_MAP_DTYPE),
        np.asarray(B, dtype=PROJECTED_MAP_DTYPE),
        np.asarray(C, dtype=PROJECTED_MAP_DTYPE),
        np.asarray(c, dtype=PROJECTED_MAP_DTYPE),
    )
    cache_dir = Path(cache_dir)
    candidates = []
    for path in cache_dir.glob(f"projected_maps_{model_key}_h*.npz"):
        try:
            with np.load(path, allow_pickle=False) as cached:
                metadata = _read_metadata(cached)
                cached_horizon = int(metadata["horizon"])
                if metadata["model_key"] == model_key and cached_horizon >= horizon:
                    candidates.append((cached_horizon, path))
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            continue

    if candidates:
        _, cache_path = min(candidates)
        with np.load(cache_path, allow_pickle=False) as cached:
            maps = {
                "obs_x": cached["obs_x"][:horizon].copy(),
                "gamma_x": cached["gamma_x"][:horizon, :, :horizon, :].copy(),
                "decoder_bias": cached["decoder_bias"].copy(),
            }
        return maps, cache_path, True, model_key

    maps = build_prediction_maps(A, B, C, c, horizon)
    metadata = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "model_key": model_key,
            "horizon": int(horizon),
        },
        sort_keys=True,
    )
    cache_path = cache_dir / f"projected_maps_{model_key}_h{horizon}.npz"
    save_array_cache(cache_path, metadata=np.array(metadata), **maps)
    return maps, cache_path, False, model_key


def load_or_build_qp_inverse(
    g_y, q_y, r_bar, rho, model_key, horizon, cache_dir=CACHE_DIR
):
    config_key = _hash_arrays(
        np.array([horizon, rho], dtype=g_y.dtype), g_y, q_y, r_bar
    )
    cache_path = Path(cache_dir) / f"qp_inverse_{model_key}_{config_key}.npz"
    expected_shape = (r_bar.shape[0], r_bar.shape[0])
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = _read_metadata(cached)
                k_inv = cached["k_inv"].copy()
            if metadata["config_key"] == config_key and k_inv.shape == expected_shape:
                return k_inv, cache_path, True
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            pass

    hessian = g_y.T @ q_y @ g_y + r_bar
    k_matrix = hessian + float(rho) * np.eye(hessian.shape[0], dtype=g_y.dtype)
    k_inv = np.linalg.inv(k_matrix).astype(g_y.dtype)
    metadata = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "config_key": config_key,
            "horizon": int(horizon),
        },
        sort_keys=True,
    )
    save_array_cache(cache_path, metadata=np.array(metadata), k_inv=k_inv)
    return k_inv, cache_path, False


def load_or_build_qp_factor(
    g_y, q_y, r_bar, rho, model_key, horizon, cache_dir=CACHE_DIR
):
    """Load or build the lower Cholesky factor of the PD system matrix."""
    config_key = _hash_arrays(
        np.array([horizon, rho], dtype=g_y.dtype), g_y, q_y, r_bar
    )
    cache_path = Path(cache_dir) / f"qp_cholesky_{model_key}_{config_key}.npz"
    expected_shape = (r_bar.shape[0], r_bar.shape[0])
    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                metadata = _read_metadata(cached)
                k_cholesky = cached["k_cholesky"].copy()
            if (
                metadata["config_key"] == config_key
                and k_cholesky.shape == expected_shape
            ):
                return k_cholesky, cache_path, True
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            pass

    hessian = g_y.T @ q_y @ g_y + r_bar
    k_matrix = hessian + float(rho) * np.eye(hessian.shape[0], dtype=g_y.dtype)
    k_cholesky = np.linalg.cholesky(k_matrix).astype(g_y.dtype)
    metadata = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "config_key": config_key,
            "horizon": int(horizon),
            "factorization": "cholesky_lower",
        },
        sort_keys=True,
    )
    save_array_cache(cache_path, metadata=np.array(metadata), k_cholesky=k_cholesky)
    return k_cholesky, cache_path, False
