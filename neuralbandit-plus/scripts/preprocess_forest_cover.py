"""Forest Cover -> the paper's 94 binary features [allesiardo2014, §5]."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype

root = Path(__file__).resolve().parent.parent
data = fetch_covtype(data_home=root / "data/raw", as_frame=True)
df = data.data

# 10 numeric columns -> 5 equal-frequency bins each, one-hot (50 columns)
bins = [pd.get_dummies(pd.qcut(df.iloc[:, c], 5)) for c in range(10)]
X = pd.concat(bins + [df.iloc[:, 10:]], axis=1).to_numpy(dtype=np.uint8)  # + 44 binary = 94
y = (data.target.to_numpy() - 1).astype(np.int8)  # classes 1..7 -> actions 0..6

perm = np.random.default_rng(0).permutation(len(X))  # shuffled once [allesiardo2014, §5]
out = root / "data/processed"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "forest_cover.npz", X=X[perm], y=y[perm])
print(X.shape, X.dtype, y.shape, y.dtype)
