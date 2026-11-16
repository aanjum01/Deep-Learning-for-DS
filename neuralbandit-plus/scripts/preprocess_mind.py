"""MIND -> data/processed/mind_{size}.npz for the category-arm contextual bandit (plan phase-5 §3, §9).

Per impression: avail/R over arms (categories of shown articles), context X = category shares of the
user's history + earlier clicks, rows in time order. Parses line by line with compact numpy storage.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "dev"]
ROOT = Path(__file__).resolve().parent.parent / "data"


def build(root_dir, small_users=None):
    root_dir = Path(root_dir)
    news = pd.concat([pd.read_csv(root_dir / s / "news.tsv", sep="\t", header=None, usecols=[0, 1], quoting=3)
                      for s in SPLITS]).drop_duplicates(0)
    ctx_cats = sorted(news[1].unique())
    cat_idx = dict(zip(news[0], news[1].map({c: i for i, c in enumerate(ctx_cats)})))
    C = len(ctx_cats)

    n = sum(sum(1 for _ in open(root_dir / s / "behaviors.tsv")) for s in SPLITS)
    split, imp_id, key = np.zeros(n, np.uint8), np.zeros(n, np.int64), np.zeros(n, np.int64)
    shown = np.zeros((n, C), bool)                    # category shown in the impression
    hist, nclick = np.zeros((n, C), np.uint16), np.zeros((n, C), np.uint8)  # history / clicked counts
    times, users = [], []
    i = 0
    for s, name in enumerate(SPLITS):
        with open(root_dir / name / "behaviors.tsv") as f:
            for line in f:
                iid, user, time, h, imps = line.rstrip("\n").split("\t")
                split[i], imp_id[i], key[i] = s, int(iid), hash((user, time, imps))
                times.append(time)
                users.append(user)
                for it in imps.split():
                    c = cat_idx.get(it[:-2])
                    if c is not None:
                        shown[i, c] = True
                        nclick[i, c] += it[-1] == "1"
                for c, k in Counter(cat_idx[x] for x in h.split() if x in cat_idx).items():
                    hist[i, c] = k
                i += 1

    t = pd.to_datetime(pd.Series(times), format="%m/%d/%Y %I:%M:%S %p").to_numpy("datetime64[s]")
    users = np.array(users)
    keep = ~pd.Series(key).duplicated().to_numpy()  # same user+time+impressions string: keep first
    n_dup = int((~keep).sum())
    if small_users is not None:
        keep &= ~np.isin(users, list(small_users))
    n_filtered = int(n - n_dup - keep.sum())
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(t[idx], kind="stable")]  # ties keep file order (train before dev)
    split, imp_id, shown, hist, nclick, t, users = (a[idx] for a in (split, imp_id, shown, hist, nclick, t, users))

    user = np.unique(users, return_inverse=True)[1]
    arms = shown.any(0)                                # categories ever shown = arms (sorted, plan Q9)
    avail, R = shown[:, arms], (nclick[:, arms] > 0).astype(np.uint8)
    assert avail.any(1).all(), "an impression with no available arm would break masking"

    # History + the user's clicks from EARLIER impressions only (exclusive per-user cumsum, plan Q14)
    earlier = pd.DataFrame(nclick, dtype=np.int64).groupby(user).cumsum().to_numpy() - nclick
    cnt = hist + earlier
    X = np.ones((len(idx), C + 1), np.float32)          # last column = constant 1 (plan Q16)
    X[:, :C] = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)  # all zeros when no history yet

    return dict(X=X, R=R, avail=avail, time=t, user=user.astype(np.int32), impression_id=imp_id, split=split,
                arms=np.array(ctx_cats)[arms], ctx_cats=np.array(ctx_cats), n_dup=n_dup, n_filtered=n_filtered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["small", "large"], required=True)
    size = parser.parse_args().size
    small_users = None
    if size == "large":  # Q22: small = tuning set, large minus small's users = evaluation set
        small_users = set(pd.concat([pd.read_csv(ROOT / "raw/mind/small" / s / "behaviors.tsv", sep="\t",
                                                 header=None, usecols=[1]) for s in SPLITS])[1])
    out = build(ROOT / "raw/mind" / size, small_users)
    n_dup, n_filtered = out.pop("n_dup"), out.pop("n_filtered")
    (ROOT / "processed").mkdir(exist_ok=True)
    np.savez(ROOT / f"processed/mind_{size}.npz", **out)
    print(f"mind_{size}: {len(out['X']):,} rows, {out['user'].max() + 1:,} users, {len(out['arms'])} arms, "
          f"X {out['X'].shape}, duplicates dropped {n_dup:,}, small-user filter dropped {n_filtered:,}")
    print(f"time {out['time'].min()} -> {out['time'].max()}; mean R.max(1) {out['R'].max(1).mean():.4f}, "
          f"mean avail.sum(1) {out['avail'].sum(1).mean():.3f}")
