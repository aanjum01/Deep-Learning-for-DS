"""Gradual forgetting: pull weights back toward their initial values (plan phase-6b2; [kumar2024, §3, App. A.3] L2 Init)."""


def fade_toward_init(net, W1_0, w2_0, rho):
    """In place: theta <- theta - rho (theta - theta_0) for both layers of every network in an MLPStack."""
    if not 0 <= rho < 1:
        raise ValueError(f"fade rate rho = {rho} must be in [0, 1)")
    if rho:
        net.W1 -= rho * (net.W1 - W1_0)
        net.w2 -= rho * (net.w2 - w2_0)
