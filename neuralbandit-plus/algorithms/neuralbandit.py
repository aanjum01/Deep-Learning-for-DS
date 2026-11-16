"""NeuralBandit1/2/3 [allesiardo2014]."""

import numpy as np

from algorithms.base import Bandit
from exploration.egreedy import egreedy
from exploration.exp3 import EXP3
from models.mlp import MLPStack


class NeuralBandit1(Bandit):
    """NeuralBandit1 [allesiardo2014, §3, Alg. 1, pp. 3-4].

    One network per action (d -> hidden -> 1, sigmoid, no biases) estimates P(reward = 1 | x).
    Play epsilon-greedy over the network outputs with gamma. Update as Eq. 1 is printed
    [allesiardo2014, Eq. 1 + Theorem 1, p. 4]: indicator 1[k_hat = k], so only the greedy action's
    network is updated, and only when it was played (its reward is only known then), with one
    gradient step of size lam weighted by 1 / P(k_hat) ~ 1.004 (plan D2-D6, D4 revised Nov 15).
    Reading Alg. 1's unmarked k_t as the played action instead makes explored updates x1,400 and
    kills the sigmoid networks (paper/RESEARCH_LOG.md, section 7).
    """

    def __init__(self, n_actions, d=94, hidden=25, lam=0.1, gamma=0.005, seed=0):
        super().__init__(n_actions, seed)
        self.d = d
        self.hidden = hidden
        self.lam = lam
        self.gamma = gamma
        self.reset()

    def _init_state(self):
        self.net = MLPStack(1, self.n_actions, self.d, self.hidden, [self.lam], self.rng)

    def select(self, context, candidates=None, avail=None):
        self._h, self._o = self.net.forward(context)  # cached for update (same context)
        a, self._greedy, self.last_probs = egreedy(self._o[0], self.gamma, self.rng, avail)
        return a

    def update(self, context, action, reward, candidates=None):
        if action == self._greedy:  # Eq. 1: 1[k_hat = k]
            self.net.update(context, self._h, self._o, action, reward, 1.0 / self.last_probs[action])


class _Committee(Bandit):
    """Shared part of NB2/NB3: the 15 NB1 models of the grid C x lam [allesiardo2014, §4-5, pp. 4-5].

    Stored as one MLPStack per C holding len(lam) models; model index m = c_index * len(lam) + lam_index.
    Every model is trained in parallel (plan D5): on a greedy step (played action == committee k_hat)
    network k of every model gets one step weighted by 1 / P(k_hat) (Eq. 1 as printed, plan D1/D2).
    """

    def __init__(self, n_actions, d=94, hidden=(1, 5, 25, 50, 100), lam=(0.01, 0.1, 1),
                 gamma=0.005, gamma_model=0.1, seed=0):
        super().__init__(n_actions, seed)
        self.d, self.hidden, self.lam = d, list(hidden), list(lam)
        self.gamma, self.gamma_model = gamma, gamma_model
        self.M = len(self.hidden) * len(self.lam)
        self.reset()

    def _init_state(self):
        self.stacks = [MLPStack(len(self.lam), self.n_actions, self.d, C, self.lam, self.rng)
                       for C in self.hidden]

    def _forward(self, x):
        """All models' outputs O (M, K); each stack's (h, o) is cached for the update."""
        self._ho = [s.forward(x) for s in self.stacks]
        return np.concatenate([o for _, o in self._ho])

    def _update_models(self, x, a, r):
        """Eq. 1, 1[k_hat = k]: update network a of every model only if a was the greedy action."""
        if a == self._greedy:
            for s, (h, o) in zip(self.stacks, self._ho):
                s.update(x, h, o, a, r, 1.0 / self._P[a])


class NeuralBandit2(_Committee):
    """NeuralBandit2 [allesiardo2014, Alg. 2, p. 6]: one EXP3 (Eq. 2-3) picks the model that plays.

    The chosen model m_t plays epsilon-greedy with gamma (plan D6). Its k_hat and P gate and weight the
    update of network a of every model (plan D1 revised, D5); EXP3 is rewarded with y at m_t.
    last_probs is the true action probability sum_m P_model(m) P_m(k) (for logging only).
    """

    def _init_state(self):
        super()._init_state()
        self.exp3 = EXP3(self.M, self.gamma_model, self.rng)

    def select(self, context, candidates=None, avail=None):
        O = self._forward(context)
        self._m = self.exp3.draw()
        a, self._greedy, self._P = egreedy(O[self._m], self.gamma, self.rng, avail)
        if avail is not None:  # plan Q11: only available arms can be greedy or explored
            O = np.where(avail, O, -np.inf)
        greedy = O.argmax(1)
        greedy[self._m] = self._greedy
        K = self.n_actions
        explore = self.gamma / K if avail is None else self.gamma / avail.sum() * avail
        self.last_probs = explore + (1 - self.gamma) * np.bincount(
            greedy, weights=self.exp3.probs(), minlength=K)
        return a

    def update(self, context, action, reward, candidates=None):
        self._update_models(context, action, reward)
        self.exp3.update(self._m, reward)

    def diagnostics(self):
        """EXP3's final model probabilities, shape (M,)."""
        return {"model_probs": self.exp3.probs()}


class NeuralBandit3(_Committee):
    """NeuralBandit3 [allesiardo2014, Alg. 3, p. 6]: one EXP3 per action picks that action's model.

    Score s_k = N^{m_k, k}(x) with m_k ~ EXP3_k; play epsilon-greedy on s with gamma. Alg. 3's k_hat and
    P gate and weight the update of network a of every model (plan D2). Only the played action's EXP3
    receives y (plan D3): the other actions' rewards are never observed.
    """

    def _init_state(self):
        super()._init_state()
        self.exp3s = [EXP3(self.M, self.gamma_model, self.rng) for _ in range(self.n_actions)]

    def select(self, context, candidates=None, avail=None):
        O = self._forward(context)
        self._ms = [e.draw() for e in self.exp3s]  # all K EXP3s draw (same rng use as without avail)
        a, self._greedy, self._P = egreedy(O[self._ms, np.arange(self.n_actions)], self.gamma, self.rng,
                                           avail)
        self.last_probs = self._P
        return a

    def update(self, context, action, reward, candidates=None):
        self._update_models(context, action, reward)
        self.exp3s[action].update(self._ms[action], reward)

    def diagnostics(self):
        """Each action's EXP3 final model probabilities, shape (K, M)."""
        return {"model_probs": np.array([e.probs() for e in self.exp3s])}
