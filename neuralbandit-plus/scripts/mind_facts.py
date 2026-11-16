"""Data facts for the Phase 5 data review (plan §6, §6b). Aggregates only — no raw rows printed.

Reads behaviors.tsv line by line and keeps a few numbers per impression (categories as bit masks),
so MIND-large fits in memory.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--size", choices=["small", "large"], default="small")
root = Path(__file__).resolve().parent.parent / "data/raw/mind" / parser.parse_args().size
SPLITS = ["train", "dev"]

news = pd.concat([pd.read_csv(root / s / "news.tsv", sep="\t", header=None, usecols=[0, 1], quoting=3)
                  for s in SPLITS]).drop_duplicates(0)
cats = sorted(news[1].unique())
bit = {c: 1 << i for i, c in enumerate(cats)}
cat_bit = dict(zip(news[0], news[1].map(bit)))

cols = {k: [] for k in ["split", "user", "time", "n_shown", "n_click", "shown", "click", "hist_len", "hist", "key"]}
missing_shown, hist_ids = set(), set()
for split in SPLITS:
    with open(root / split / "behaviors.tsv") as f:
        for line in f:
            _, user, time, hist, imps = line.rstrip("\n").split("\t")
            shown = click = n_click = 0
            items = imps.split()
            for it in items:
                nid, label = it[:-2], it[-1]
                b = cat_bit.get(nid)
                if b is None:
                    missing_shown.add(nid)
                    continue
                shown |= b
                if label == "1":
                    click |= b
                    n_click += 1
            h = hist.split()
            hist_ids.update(h)
            for k, v in zip(cols, [split, user, time, len(items), n_click, shown, click, len(h), hash(hist),
                                   hash((user, time, imps))]):
                cols[k].append(v)

d = pd.DataFrame(cols)
d["time"] = pd.to_datetime(d["time"], format="%m/%d/%Y %I:%M:%S %p")
shown, click = d["shown"].to_numpy(np.int64), d["click"].to_numpy(np.int64)
k_shown, k_click = np.bitwise_count(shown), np.bitwise_count(click)
Q = [0, 10, 50, 90, 99, 100]


def pct(s):
    return " ".join(f"p{p}={np.percentile(s, p):g}" for p in Q)


print(f"== Files ({root.name}) ==")
for s, g in d.groupby("split", sort=False):
    print(f"{s}: {len(g):,} impressions, {g.user.nunique():,} users, {g.time.min()} → {g.time.max()}")
print(f"identical impressions in both files (user+time+list): {d.key.duplicated(keep=False).sum():,}")
tr, dv = (set(d.user[d.split == s]) for s in SPLITS)
print(f"users in both files: {len(tr & dv):,}")
print(f"total: {len(d):,} impressions, {d.user.nunique():,} users; news: {len(news):,}")
print(f"steps per run after the 90% user cut ≈ {0.9 * len(d):,.0f}")

print("\n== Impressions ==")
print("articles shown per impression:", pct(d.n_shown))
print("clicks per impression:", pct(d.n_click))
print(f"zero-click impressions: {(d.n_click == 0).sum():,}")
print(f"CTR (clicks / shown articles): {d.n_click.sum() / d.n_shown.sum():.4f}")
print(f"shown news IDs missing from news.tsv: {len(missing_shown)}")

print("\n== Categories (arms) ==")
arms = [c for c in cats if (shown & bit[c]).any()]
print(f"categories in news.tsv: {len(cats)}; shown in impressions: {len(arms)}")
print("categories shown per impression:", pct(k_shown), f"mean={k_shown.mean():.2f}")
print(f"mean share of arms shown per impression: {k_shown.mean() / len(arms):.1%}")
t = pd.DataFrame([(c, ((shown & bit[c]) > 0).mean(), ((click & bit[c]) > 0).mean(),
                   ((click & bit[c]) > 0).sum() / max(((shown & bit[c]) > 0).sum(), 1)) for c in arms],
                 columns=["category", "shown_in", "reward1_share", "reward_if_shown"])
print(t.sort_values("shown_in", ascending=False).to_string(index=False, float_format="{:.4f}".format))
print(f"Random's expected reward (mean clicked-shown ÷ shown categories): {(k_click / k_shown).mean():.4f}")
print(f"best-available oracle reward (= share of impressions with a click): {(k_click > 0).mean():.4f}")

print("\n== Users & history ==")
per_user = d.user.value_counts()
print("impressions per user:", pct(per_user), f"mean={per_user.mean():.2f}")
print("history length:", pct(d.hist_len))
print(f"empty histories: {(d.hist_len == 0).sum():,}")
print(f"history news IDs missing from news.tsv: {len(hist_ids - set(cat_bit)):,} of {len(hist_ids):,}")
multi = d[d.user.map(per_user) > 1]
n_hist = multi.groupby("user")["hist"].nunique()
print(f"users with >1 impression: {len(n_hist):,}; of them with >1 distinct History: {(n_hist > 1).sum():,}")

print("\n== Natural drift: click share per category per day ==")
day = d.time.dt.date.to_numpy()
daily = pd.DataFrame({c: (click & bit[c]) > 0 for c in arms}).groupby(day).mean()
print(daily[t.sort_values("reward1_share", ascending=False).category.head(8)].to_string(float_format="{:.3f}".format))
print("impressions per day:", {str(k): v for k, v in pd.Series(day).value_counts().sort_index().items()})

if root.name == "large":  # Q22: tuning (small) and evaluation (large minus small's users) must not share users
    s_users = set(pd.concat([pd.read_csv(root.parent / "small" / s / "behaviors.tsv", sep="\t", header=None,
                                         usecols=[1]) for s in SPLITS])[1])
    print("\n== Small vs large (Q22) ==")
    print(f"small users: {len(s_users):,}; of them in large: {len(s_users & set(d.user)):,}")
    left = ~d.user.isin(s_users)
    print(f"large minus small's users: {left.sum():,} impressions, {d.user[left].nunique():,} users "
          f"(≈ {0.9 * left.sum():,.0f} steps per run after the 90% cut)")
