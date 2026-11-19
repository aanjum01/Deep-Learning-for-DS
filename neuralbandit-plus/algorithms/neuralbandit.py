"""NeuralBandit1/2/3 [allesiardo2014]."""

import numpy as np
from scipy.special import expit

from algorithms.base import Bandit
from exploration.egreedy import egreedy
from exploration.exp3 import EXP3
from models.mlp import MLPStack
from nonstationary.controller import CONTROLLERS
from nonstationary.drift import make_detector
from nonstationary.forgetting import fade_toward_init
from nonstationary.window import Notebook


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


class NeuralBandit1SW(NeuralBandit1):
    """NeuralBandit1 + sliding window: reset + retrain on the last W steps (plan phase-6b, Q1-Q3; [gama2014, §3.1.1,
    §3.3.2] blind periodic retraining).

    Every Eq. 1 update NB1 makes is also logged (x, action, reward, weight 1/P). At the end of update, when
    (t - offset) % R == 0 and t > W (R = W // 5), the log is trimmed to the last W steps, a fresh network is drawn (paper init,
    from self.rng) and trained by one in-order pass of the logged updates with NB1's own step. In between, NB1 learns
    online as usual. Not literally "an NB1 that started W steps ago": the logged rows are gated by the old network's
    greedy action (plan R5). W may be changed and retrain() called from outside (6c/6d controllers).
    """

    def __init__(self, n_actions, W=50_000, offset=0, **kw):
        self.W, self.offset = W, offset  # offset shifts the reset schedule (timing-sensitivity check, RESEARCH_LOG §6)
        super().__init__(n_actions, **kw)

    def _init_state(self):
        super()._init_state()
        self.t = 0
        self.notebook = Notebook(self.d)
        self.retrain_steps = []

    def update(self, context, action, reward, candidates=None):
        super().update(context, action, reward)
        self.t += 1
        if action == self._greedy:  # log exactly the updates NB1 made (Eq. 1 gate)
            self.notebook.add(self.t, context, action, reward, 1.0 / self.last_probs[action])
        if (self.t - self.offset) % max(self.W // 5, 1) == 0 and self.t > self.W:  # max: a bad W reaches retrain's check
            self.retrain()

    def retrain(self):
        """Forget everything older than W steps: fresh network, one in-order pass over the logged updates."""
        if self.W < 5:
            raise ValueError(f"window W = {self.W} < 5 steps")
        self.notebook.trim(self.t, self.W)
        self.net = MLPStack(1, self.n_actions, self.d, self.hidden, [self.lam], self.rng)
        for x, a, r, w in self.notebook:
            h, o = self.net.forward(x)
            self.net.update(x, h, o, a, r, w)
        self.retrain_steps.append(self.t)

    def diagnostics(self):
        return {"retrain_steps": np.array(self.retrain_steps, dtype=np.int64)}


class NeuralBandit1Fade(NeuralBandit1):
    """NeuralBandit1 + gradual forgetting (plan phase-6b2, Q2/Q3/Q10/Q11): after NB1's own Eq. 1 update, at every step,
    every arm's network (both layers) is pulled toward its own initial draw, theta <- theta - rho (theta - theta_0)
    [kumar2024, L2 Init, §3; App. A.3, up to an O(alpha rho) term]. rho = 0 is NB1 exactly; rho may be changed during a
    run (6c/6d). Traces per 1,000 steps (Q9/R7/R13): mean |W1| and |w2| at the block's end, and the share of played
    actions whose estimate is within 0.02 of that arm's own starting estimate for the same context.
    """

    def __init__(self, n_actions, rho=1e-5, **kw):
        self.rho = rho
        super().__init__(n_actions, **kw)

    def _init_state(self):
        super()._init_state()
        self.W1_0, self.w2_0 = self.net.W1.copy(), self.net.w2.copy()  # exact copy, no rng draw
        self.t, self._near, self._trace = 0, 0, []

    def update(self, context, action, reward, candidates=None):
        super().update(context, action, reward)
        idx = np.flatnonzero(context)  # the played arm's starting estimate for this context (trace only)
        o0 = expit(expit(context[idx] @ self.W1_0[0, action, idx]) @ self.w2_0[0, action])
        self._near += abs(self._o[0, action] - o0) <= 0.02
        fade_toward_init(self.net, self.W1_0, self.w2_0, self.rho)
        self.t += 1
        if self.t % 1000 == 0:
            self._trace.append((np.abs(self.net.W1).mean(), np.abs(self.net.w2).mean(), self._near / 1000))
            self._near = 0

    def diagnostics(self):
        W1, w2, near = (np.array(c) for c in zip(*self._trace)) if self._trace else (np.zeros(0),) * 3
        return {"mean_abs_W1": W1, "mean_abs_w2": w2, "near_start_rate": near}


class NeuralBandit1Ctrl(NeuralBandit1Fade):
    """NB1 + slow fade + Page-Hinkley alarm + controller (plan phase-6c, §2/§5).

    PH watches NB1's surprise, reward minus the played arm's pre-update estimate, on every step (Q2). On an alarm,
    CONTROLLERS[controller](self) returns a decision dict {"rho", "length", ...} or None (no change). 6d's LLM adds fields
    (label, reason), which are logged as given. A decision with rho outside [0, 1) or length < 1 is invalid and counts as no
    change (plan §2). A boost replaces the base fade rate for `length` steps. During a boost PH is not fed and alarms are
    ignored. At its end the base rate returns and PH restarts (Q5). After a no-change alarm, river restarts PH by itself.
    theta_0 never changes. The controller is named by a string and the boost is a param, so both are in the saved-run key.
    """

    def __init__(self, n_actions, rho=1e-5, threshold=50.0, controller="rule", boost_rho=1 / 20_000, boost_length=20_000,
                 **kw):
        self.base_rho, self.threshold, self.controller = rho, threshold, controller
        self.boost_rho, self.boost_length = boost_rho, boost_length
        super().__init__(n_actions, rho=rho, **kw)

    def _init_state(self):
        super()._init_state()
        self.alarms = []  # one dict per alarm: {"step", **decision} ({"step"} alone = no change)
        self._calm()

    def _calm(self):
        """Base fade rate, no boost, a fresh PH."""
        self.rho, self.boost_end, self.ph = self.base_rho, 0, make_detector(self.threshold)

    def update(self, context, action, reward, candidates=None):
        surprise = float(reward - self._o[0, action])  # cached by select: the estimate before this update
        super().update(context, action, reward)  # NB1 step, then the fade at the current rho; t += 1
        if self.boost_end:
            if self.t >= self.boost_end:
                self._calm()
            return
        self.ph.update(surprise)
        if self.ph.drift_detected:
            d = CONTROLLERS[self.controller](self)
            if not (d and 0 <= d.get("rho", -1) < 1 and int(d.get("length", 0)) >= 1):
                d = None
            self.alarms.append({"step": self.t, **(d or {})})
            if d:
                self.rho, self.boost_end = d["rho"], self.t + int(d["length"])

    def diagnostics(self):
        get = lambda k, fill, dt: np.array([a.get(k, fill) for a in self.alarms], dtype=dt)  # noqa: E731
        return {**super().diagnostics(), "alarm_steps": get("step", 0, np.int64),
                "alarm_rho": get("rho", np.nan, np.float64),  # NaN = no change
                "alarm_length": get("length", 0, np.int64)}


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
