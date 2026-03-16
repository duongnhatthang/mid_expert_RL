"""Discount factor configuration and utilities."""


def compute_gamma_from_horizon(horizon: int) -> float:
    """
    Compute gamma using the engineering trick: H = 1/(1-gamma) => gamma = 1 - 1/H.

    Args:
        horizon: Episode length H

    Returns:
        Discount factor gamma
    """
    if horizon <= 0:
        raise ValueError(f"Horizon must be positive, got {horizon}")
    return 1.0 - 1.0 / horizon
