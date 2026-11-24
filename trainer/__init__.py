"""Shared Koopman training objective and dataset aggregation schedule."""

from .replay import should_collect_dagger
from .training import KoopmanTrainer, LossWeights

__all__ = ["KoopmanTrainer", "LossWeights", "should_collect_dagger"]
