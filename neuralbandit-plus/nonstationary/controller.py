"""Controllers for NeuralBandit1Ctrl's alarm hook (plan phase-6c §2): controller(policy) -> {"rho", "length", ...} or None.

Named in CONTROLLERS so configs pick one by name (and the name is part of the saved-run key). 6d adds the LLM controller here.
"""


def rule_controller(policy):
    """A/B (Q5/R1): every alarm -> the policy's fixed boost (configs: 1/20k for 20k steps, one memory length)."""
    return {"rho": policy.boost_rho, "length": policy.boost_length}


CONTROLLERS = {"rule": rule_controller}
