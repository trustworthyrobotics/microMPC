"""Runtime setup that must run before importing JAX or its native backend."""

import logging
import os


def configure_jax_logging():
    """Silence native compiler diagnostics and optional Python compilation logs."""
    # Level 3 suppresses INFO/WARNING/ERROR diagnostics, including recoverable
    # xtile/Triton compilation failures. Fatal failures and Python exceptions
    # still propagate normally.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    os.environ["JAX_LOG_COMPILES"] = "false"
    os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "false"
    logging.getLogger("jax").setLevel(logging.ERROR)
    logging.getLogger("jaxlib").setLevel(logging.ERROR)
