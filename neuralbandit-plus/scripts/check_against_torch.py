"""One-time cross-check of models/mlp.py against torch autograd (plan D9).

Run from the project root (torch goes only into a temporary uv environment, never into .venv):
    uv run --no-project --with torch --with numpy --with scipy python scripts/check_against_torch.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mlp import MLPStack  # noqa: E402

K, d, C, lam = 7, 94, 25, 0.1
rng = np.random.default_rng(0)
net = MLPStack(1, K, d, C, [lam], rng)
W1 = torch.tensor(net.W1[0], requires_grad=True)  # float64, same initial weights
w2 = torch.tensor(net.w2[0], requires_grad=True)

for _ in range(1000):
    x = np.zeros(d)
    x[rng.choice(d, 12, replace=False)] = 1.0
    a, r = int(rng.integers(K)), float(rng.integers(2))
    w = rng.choice([1 / (1 - 0.005 + 0.005 / K), K / 0.005])  # greedy or explored 1/P
    net.update(x, *net.forward(x), a, r, w)
    o = torch.sigmoid(torch.sigmoid(torch.tensor(x) @ W1[a]) @ w2[a])  # no biases
    (0.5 * (o - r) ** 2).backward()
    with torch.no_grad():  # plain SGD step lam * w * grad
        W1 -= lam * w * W1.grad
        w2 -= lam * w * w2.grad
    W1.grad, w2.grad = None, None

e1 = np.abs(net.W1[0] - W1.detach().numpy()).max()
e2 = np.abs(net.w2[0] - w2.detach().numpy()).max()
print(f"max |W1 diff| = {e1:.3e}, max |w2 diff| = {e2:.3e}")
assert e1 < 1e-8 and e2 < 1e-8
print("OK: numpy backprop matches torch autograd")
