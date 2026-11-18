"""Metrics: regret and accuracy [allesiardo2014, §5]; recovery after planted shifts (plan phase-5d)."""

import numpy as np


def cumulative_regret(rewards, ref=0.93):
    """Cumulative regret vs. a reference reward per step [§5, p. 5].

    ref: the paper's constant 0.93 (Forest Cover), or an array of each step's best available reward
    (MIND dynamic regret, plan Q17).
    """
    return np.cumsum(ref - np.asarray(rewards))


def last_k_accuracy(rewards, k=100_000):
    """Mean reward over the last k steps [§5, p. 6]; NaN if the run is shorter than k."""
    return float(np.mean(rewards[-k:])) if len(rewards) >= k else float("nan")


# ---- Recovery after a planted shift (plan phase-5d) [shaker2015recovery, §3.1, §3.3, Eq. 8] ----

def trailing_mean(rewards, w=10_000):
    """Mean of the last w rewards at every step t (steps t-w+1..t); NaN for t < w-1 (plan Q6, R7)."""
    c = np.cumsum(np.insert(np.asarray(rewards, dtype=float), 0, 0.0))
    out = np.full(len(rewards), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def recovery(ratio, shift, end, w=10_000, thr=0.95):
    """Stopwatch on a ratio curve (E4 mean / E3 mean, plan R1) for the stretch [shift, end).

    t1 = first step with ratio < thr; t2 = first step from which ratio >= thr for w consecutive steps, the
    stretch ending before `end` (R10's duration; "stays recovered" is our addition, plan Q7/R7). Returns a dict:
    duration = t2 - t1 and since_shift = t2 - shift, or lower bounds (end - t1, end - shift) with
    recovered=False; never below thr -> both 0. max_loss = max(1 - ratio) on the stretch (R10 Eq. 8 with the
    no-shift run as reference); entered_at = ratio just before the shift (plan S4).
    """
    seg = ratio[shift:end]
    out = {"entered_at": float(ratio[shift - 1]), "max_loss": float(np.nanmax(1 - seg)), "t1": None, "t2": None,
           "duration": 0, "since_shift": 0, "recovered": True}
    below = np.flatnonzero(seg < thr)
    if len(below):
        k1 = below[0]
        c = np.concatenate(([0], np.cumsum(seg[k1:] >= thr)))
        full = np.flatnonzero(c[w:] - c[:-w] == w)  # starts of w consecutive steps all >= thr
        out["t1"] = shift + k1
        stop = out["t1"] + int(full[0]) if len(full) else end
        out.update(t2=stop if len(full) else None, recovered=bool(len(full)),
                   duration=stop - out["t1"], since_shift=stop - shift)
    return out


def clicks_lost(e4, e3, shift, length):
    """Signed extra regret of the shifted run vs its no-shift twin over steps [shift, shift+length) (plan Q8, R4)."""
    return int(np.sum(e3[shift:shift + length], dtype=np.int64) - np.sum(e4[shift:shift + length], dtype=np.int64))


def adapts_better(lost_a, lost_b, total_a, total_b):
    """Pre-registered verdict for A vs B at one shift (plan R2, S2, S3), from per-seed arrays.

    "better" only if A loses fewer clicks by more than 2x the combined standard error AND collects at least as
    many total clicks in the window; fewer clicks lost but fewer total clicks = trades normal performance for
    robustness; otherwise no clear difference.
    """
    lost_a, lost_b = np.asarray(lost_a, float), np.asarray(lost_b, float)
    se = np.sqrt(lost_a.var(ddof=1) / len(lost_a) + lost_b.var(ddof=1) / len(lost_b))
    if lost_b.mean() - lost_a.mean() > 2 * se:
        return "better" if np.mean(total_a) >= np.mean(total_b) else "trades performance for robustness"
    if lost_a.mean() - lost_b.mean() > 2 * se:
        return "worse"
    return "no clear difference"
