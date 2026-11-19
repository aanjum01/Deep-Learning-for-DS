import runpy
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

import evaluation.runner as runner

ROOT = Path(__file__).resolve().parent.parent
Diverged = pytest.importorskip("algorithms.neuralucb").Diverged


def test_diverged_run_saved_and_summarised(tmp_path, monkeypatch):
    name = "test_diverged_tmp"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"name": name, "steps": 2000, "seeds": [0, 1], "drift_every": None, "algorithms": [
        {"label": "Random", "class": "RandomPolicy", "params": {}}, {"label": "Bad", "class": "RandomPolicy", "params": {"x": 1}}]}))
    real = runner.run_one
    parallel = joblib.Parallel

    def fake(cls, params, *a):
        if "x" in params:
            raise Diverged(1234)
        return real(cls, params, *a)

    monkeypatch.setattr(runner, "run_one", fake)
    monkeypatch.setattr(joblib, "Parallel", lambda **kw: parallel(n_jobs=1))  # in-process: sees the patched run_one
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(cfg)])
    try:
        for _ in range(2):  # second pass reuses the saved runs, diverged ones included
            runpy.run_path(str(ROOT / "scripts/run_experiment.py"), run_name="__main__")
        d = np.load(ROOT / f"results/raw/{name}/Bad_seed0.npz")
        assert d["diverged"] and int(d["diverged_step"]) == 1234 and len(d["rewards"]) == 0
        s = pd.read_csv(ROOT / f"results/{name}_summary.csv")
        assert list(s.diverged) == [False, False, True, True]
        assert s[s.label == "Bad"].mean_reward.isna().all() and s[s.label == "Random"].mean_reward.notna().all()
        assert set(pd.read_csv(ROOT / f"results/{name}_regret.csv").label) == {"Random"}
    finally:
        shutil.rmtree(ROOT / f"results/raw/{name}", ignore_errors=True)
        for f in ROOT.glob(f"results/{name}_*.csv"):
            f.unlink()
