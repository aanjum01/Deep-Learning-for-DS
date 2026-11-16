"""
Shared contract for every contextual-bandit algorithm in NeuralBandit++.

Every algorithm (Random, LinUCB, NeuralBandit 1/2/3, NeuralUCB, ...)
subclasses `Bandit`, so the experiment runner can treat
them all identically and comparisons stay fair.

Interaction loop (Allesiardo et al., 2014, Definition 1):

    for t = 1, ..., T:
        x_t, candidates = environment reveals the context
        a_t = policy.select(x_t, candidates)
        r_t = environment reveals the reward of a_t ONLY
        policy.update(x_t, a_t, r_t, candidates)

Two action settings share the same API:
  * Fixed arms  (Forest Cover, MIND category-arm): candidates is None and
    the actions are 0 .. n_actions-1.
  * Candidate items (MIND shared-network): candidates is a 2-D array with
    one row of features per item; the returned action is a row index.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Bandit(ABC):
    """Abstract base class for contextual-bandit policies.

    Subclasses must implement:
        _init_state()  build/clear all learned parameters
        select(...)    choose an action and set self.last_probs
        update(...)    learn from the reward of the chosen action

    Subclass constructor pattern:
        super().__init__(n_actions, seed)
        <store hyperparameters>
        self.reset()                  # builds the initial learned state
    """

    def __init__(self, n_actions: int | None = None, seed: int = 0) -> None:
        if n_actions is not None and n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}")
        self.n_actions = n_actions
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Probability assigned to each action at the most recent select().
        # Needed for importance weighting (NeuralBandit Eq. 1), EXP3 (Eq. 3)
        # and logging. None until the first select().
        self.last_probs: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Required by every algorithm
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _init_state(self) -> None:
        """Create (or re-create) all learned parameters from scratch."""

    @abstractmethod
    def select(self, context: np.ndarray,
               candidates: np.ndarray | None = None,
               avail: np.ndarray | None = None) -> int:
        """Return the index of the chosen action and set self.last_probs.

        context:    1-D feature vector for this round.
        candidates: None for fixed arms, or a (n_candidates, d) array.
        avail:      None (every arm available, Forest Cover) or a bool mask of the fixed arms
                    available this round (MIND: categories shown; arm set A_t [li2010, §2.1]).
                    Only available arms may be chosen; last_probs is 0 on the others.
        """

    @abstractmethod
    def update(self, context: np.ndarray, action: int, reward: float,
               candidates: np.ndarray | None = None) -> None:
        """Learn from the reward of `action` only (bandit feedback)."""

    # ------------------------------------------------------------------ #
    # Shared behaviour
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Return to an untrained state; same seed -> identical rerun."""
        self.rng = np.random.default_rng(self.seed)
        self.last_probs = None
        self._init_state()

    def diagnostics(self) -> dict:
        """Extra arrays saved with each run (e.g. EXP3 model probabilities); none by default."""
        return {}

    @property
    def name(self) -> str:
        """Label used in logs, CSV columns and plot legends."""
        return type(self).__name__

    def n_choices(self, candidates: np.ndarray | None = None) -> int:
        """Number of actions available this round."""
        if candidates is not None:
            return len(candidates)
        if self.n_actions is None:
            raise ValueError("No candidates given and n_actions is not set.")
        return self.n_actions
