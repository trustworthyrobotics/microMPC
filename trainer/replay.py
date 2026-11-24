"""Scheduling for dataset aggregation."""


def should_collect_dagger(
    epoch: int, total_epochs: int, start: int, interval: int
) -> bool:
    """Collect after the requested completed epoch, but never after the final epoch."""
    if interval < 1:
        raise ValueError("dagger_interval must be positive.")
    return epoch < total_epochs and epoch >= start and (epoch - start) % interval == 0
