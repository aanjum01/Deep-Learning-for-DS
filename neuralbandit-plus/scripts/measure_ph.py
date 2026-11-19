"""Phase 6c step 4-5 (plan phase-6c §4): MIND-small, NB1 + fade 1/100k, no shifts, seeds 100-102.

Saves each seed's surprise stream once, then simulates PH offline over a lambda grid with the controller's logic (a 20k-step
boost after each alarm, PH not fed during it, then a fresh PH) -> boosted share per lambda. lambda_A / lambda_S = the smallest grid
value whose boosted share is <= 10% / 30% on every seed (pre-registered rule). MIND-large is never touched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.neuralbandit import NeuralBandit1Fade  # noqa: E402
from evaluation.environments import load_mind, mind_rows, mind_stream  # noqa: E402
from nonstationary.drift import make_detector  # noqa: E402

SEEDS, LENGTH, BUDGETS = (100, 101, 102), 20_000, {"A": 0.10, "S": 0.30}
GRID = np.round(np.geomspace(10, 2000, 25), 1)


def surprises(seed, data):
    path = ROOT / f"results/raw/phase6c_measure/surprise_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    p, out = NeuralBandit1Fade(16, rho=1e-5, d=19, hidden=1, lam=1, gamma=0.005, seed=seed), []
    for x, R, av in mind_stream(data, mind_rows(data, seed), None):
        a = p.select(x, avail=av)
        out.append(R[a] - p._o[0, a])
        p.update(x, a, int(R[a]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, s := np.array(out))
    return s


def simulate(s, lam):
    """Alarm steps (1-based) with the controller's boost / restart logic, on a fixed surprise stream."""
    ph, alarms, t = make_detector(lam), [], 0
    while t < len(s):
        ph.update(float(s[t]))
        t += 1
        if ph.drift_detected:
            alarms.append(t)
            t += LENGTH  # boosted: PH not fed
            ph = make_detector(lam)
    return alarms


data = load_mind("small")
rows = []
for seed in SEEDS:
    s = surprises(seed, data)
    for lam in GRID:
        al = simulate(s, lam)
        boosted = sum(min(LENGTH, len(s) - a) for a in al) / len(s)
        rows.append((seed, lam, len(al), boosted, al[:12]))
df = pd.DataFrame(rows, columns=["seed", "lam", "alarms", "boosted_share", "first_alarm_steps"])
df.to_csv(ROOT / "results/phase6c_measure.csv", index=False)
worst = df.groupby("lam").boosted_share.max()
print(df.pivot(index="lam", columns="seed", values="alarms").assign(max_boosted=worst.round(3)).to_string())
for k, b in BUDGETS.items():
    ok = worst[worst <= b]
    print(f"lambda_{k} (boosted <= {b:.0%} on every seed): {ok.index.min() if len(ok) else 'none in grid'}")
