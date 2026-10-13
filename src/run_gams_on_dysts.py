"""
Step 4 (dysts) — experiment 6: GAM (ec) on ONE dysts system, autoregressive, no teacher forcing.

    data (for_dysts/dysts_data.py) -> model (for_dysts/dysts_gam_model.py)
    -> training (for_dysts/dysts_training_loop.py) -> test (for_dysts/dysts_evaluation.py) -> save

Train / test as run_nbeats_on_dysts_data_for_paper.py (train file: fit on values[:1000],
forecast values[1000:1200] free-running, sMAPE 0-200 + valid horizon); validation
windows from the test-file trajectory.

Settings: --gam_config (src/configs/gams/gams_autoreg_dysts.yaml): `data`, `model`,
`training`, `evaluation` sections. Command-line arguments override it.

Output: <root_dir>/<system>/
    result.json     {system: {"GAMAutoReg": {"prediction": [...], "smape": ..., "valid_horizon_*": ...,
                    "horizon_smape_0_100": [...], ...}}}  -- the same entry format as the
                    N-BEATS results file (final model: ridge heads if fitted)
    summary.json    configs, training history, val metrics (both heads), test metrics
                    (both heads), ridge alphas, active features with input names, timings
    gam_state.pkl   model.state_dict(): config, dt, scaler lo/hi, keras weights, ridge

Run from the repo root; the repo root (for src.metrics), src/ and ptsbenchmark/ on PYTHONPATH:
    PYTHONPATH=.:src:ptsbenchmark python src/run_gams_on_dysts.py --system Lorenz
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
import pickle
import time
import warnings
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from our_models.gam_autoreg.for_dysts.dysts_data import data_path, load_dysts_system, load_gz_json
from our_models.gam_autoreg.for_dysts.dysts_gam_model import DystsGAM, DystsGAMConfig
from our_models.gam_autoreg.for_dysts.dysts_training_loop import DystsTrainConfig, train_dysts_gam
from our_models.gam_autoreg.for_dysts.dysts_evaluation import evaluate_dysts

MODEL_NAME = "GAMAutoReg"
REPO_DIR = Path(__file__).resolve().parent.parent
DYSTS_DATA = REPO_DIR / "dysts_data/dysts_data/data"


def parse_args():
    p = argparse.ArgumentParser(description="GAM (ec), autoregressive, on one dysts system.")
    p.add_argument("--system", required=True, help='dysts system name, e.g. "Lorenz"')
    p.add_argument("--gam_config", default=str(REPO_DIR / "src/configs/gams/gams_autoreg_dysts.yaml"))
    p.add_argument("--data_dir", default=str(DYSTS_DATA))
    p.add_argument("--lyapunov_file", default=str(DYSTS_DATA / "lyapunov_exponents_qr.txt"))
    p.add_argument("--root_dir", default=str(REPO_DIR / "dysts_data/results/gams"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lags", type=int, nargs="+", default=None)
    p.add_argument("--history_length", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--predict_increment", action="store_true")
    p.add_argument("--no_ridge", action="store_true", help="Keep the Dense(1) heads")
    return p.parse_args()


def build_configs(args):
    """yaml + command line -> data kwargs, DystsGAMConfig, DystsTrainConfig, threshold."""
    with open(args.gam_config) as f:
        cfg = yaml.safe_load(f) or {}
    data_kw = dict(cfg.get("data", {}))
    if args.history_length is not None:
        data_kw["history_length"] = args.history_length
    if "val_starts" in data_kw:
        data_kw["val_starts"] = tuple(data_kw["val_starts"])

    model_kw = dict(cfg.get("model", {}))
    if args.lags is not None:
        model_kw["lags"] = args.lags
    if args.predict_increment:
        model_kw["predict_increment"] = True
    gam_cfg = DystsGAMConfig(**model_kw, seed=args.seed)

    train_kw = dict(cfg.get("training", {}))
    if args.max_epochs is not None:
        train_kw["max_epochs"] = args.max_epochs
    if args.learning_rate is not None:
        train_kw["learning_rate"] = args.learning_rate
    if args.no_ridge:
        train_kw["fit_ridge"] = False
    train_cfg = DystsTrainConfig(**train_kw, seed=args.seed)

    threshold = float(cfg.get("evaluation", {}).get("threshold", 50.0))
    return data_kw, gam_cfg, train_cfg, threshold


def to_jsonable(x):
    """numpy scalars / arrays -> python types (for json.dump)."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    data_kw, gam_cfg, train_cfg, threshold = build_configs(args)
    np.random.seed(args.seed)

    # 1. data
    t0 = time.time()
    train_file, val_file = data_kw.get("train_file", "train"), data_kw.get("val_file", "test")
    d = load_dysts_system(args.system, args.data_dir, lyapunov_file=args.lyapunov_file,
                          equation_data=load_gz_json(data_path(args.data_dir, train_file)),
                          val_equation_data=load_gz_json(data_path(args.data_dir, val_file)),
                          **data_kw)
    gam_cfg.resolve_lags(d.history_length)                    # lags: null -> 1..history_length
    if max(gam_cfg.lags) > d.history_length:
        raise ValueError(f"max lag {max(gam_cfg.lags)} > history_length {d.history_length}")
    save_dir = Path(args.root_dir) / args.system
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{args.system}] D={d.dim} dt={d.dt:.5g} lambda_max={d.lyapunov} L={d.history_length} "
          f"train_windows={len(d.train)} val_windows={len(d.val)} lags={gam_cfg.lags} "
          f"inputs/GAM={d.dim * len(gam_cfg.lags)} load_time={time.time() - t0:.1f}s", flush=True)

    # 2. model
    model = DystsGAM(gam_cfg, dim=d.dim, dt=d.dt)
    print(f"[{args.system}] {d.dim} GAMs x {model.n_features} features", flush=True)

    # 3. training (+ ridge)
    evaluate = partial(evaluate_dysts, dt=d.dt, lam=d.lyapunov, threshold=threshold)
    result = train_dysts_gam(model, d.train, d.val, train_cfg, evaluate)

    # 4. test (one window: values[1000:1200]); final model = ridge heads if fitted
    t_test = time.time()
    use_ridge = model.ridge is not None
    test = evaluate(model, d.test, stage="test", use_ridge=use_ridge, return_details=True)
    test_time = time.time() - t_test
    test_dense = (evaluate(model, d.test, stage="test", use_ridge=False, return_details=True)
                  if use_ridge else test)
    print(f"[{args.system}] test sMAPE={test['test_smape']} "
          f"valid_horizon_lyap={test['test_valid_horizon_lyap']} "
          f"(dense heads: sMAPE={test_dense['test_smape']})", flush=True)

    # 5. save
    entry = test["test_windows"][0]                           # prediction + metrics, N-BEATS format
    with open(save_dir / "result.json", "w") as f:
        json.dump(to_jsonable({args.system: {MODEL_NAME: entry}}), f, indent=2)
    with open(save_dir / "gam_state.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)
    try:
        active = model.active_features()
    except Exception as e:                                    # informative only
        active = {"error": repr(e)}
    summary = {
        "system": args.system, "model": MODEL_NAME, "dim": d.dim, "dt": d.dt,
        "lyapunov": d.lyapunov, "data": {**data_kw, "split_point": d.split_point,
                                         "n_train_windows": len(d.train)},
        "gam_config": asdict(gam_cfg), "train_config": asdict(train_cfg),
        "threshold": threshold, "n_features_per_gam": model.n_features,
        "n_trainable_params": int(sum(np.prod(np.shape(v)) for v in model.tv)),
        "train": asdict(result),
        "test": {k: v for k, v in test.items() if k != "test_windows"},
        "test_dense_heads": {k: v for k, v in test_dense.items() if k != "test_windows"},
        "test_head": "ridge" if use_ridge else "dense", "test_time_s": test_time,
        "active_features": active,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(to_jsonable(summary), f, indent=2, default=str)
    print(f"[{args.system}] saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()