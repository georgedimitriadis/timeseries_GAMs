
"""
Step 4 — experiment 3: GAM (ec) on ProbTS data, autoregressive, no teacher forcing.

    data (data.py) -> model (gam_model.py) -> training (training_loop.py) -> test -> save

Settings
--------
--probts_config  ProbTS yaml of the dataset, e.g. src/configs/ltsf/etth1/dlinear_autoreg.yaml.
                 Used for: dataset, scaler, split_val, context/prediction length,
                 batch_size, test_batch_size, max_epochs, limit_train_batches,
                 accumulate_grad_batches, seed_everything.
--gam_config     src/configs/gams/gams_autoreg.yaml: `model` -> GAMConfig,
                 `training` -> learning rate, weight decay, EMA, fit_ridge.
Command-line arguments override both.

Output: <root_dir>/<dataset>_GAMAutoReg_CTX<c>_PRED<p>_seed<s>/
    horizons_results.csv   test metrics of the final model (ridge head if fitted),
                           same columns as ProbTS (test_ND, test_CRPS, ...), so
                           src/analysis/ptsbenchmark_results/extract_results.py reads it
    summary.json           configs, dataset info, training history, val/test metrics
                           with both heads, ridge alpha, active features, timings
    gam_state.pkl          model.state_dict(): config, scaler lo/hi, keras weights, ridge

Run from the repo root with src/ and ptsbenchmark/ on PYTHONPATH:
    PYTHONPATH=src:ptsbenchmark python src/run_gams_on_probts.py \
        --probts_config src/configs/ltsf/etth1/dlinear_autoreg.yaml \
        --dataset etth1 --context_length 96 --prediction_length 96 \
        --data_path ptsbenchmark/datasets --root_dir ptsbenchmark/log_dir_gams
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from our_models.gam_autoreg.data import load_probts_data
from our_models.gam_autoreg.gam_model import GAMAutoReg, GAMConfig
from our_models.gam_autoreg.training_loop import TrainConfig, evaluate_probts, train_gam

MODEL_NAME = "GAMAutoReg"   # no underscore: extract_results.py splits folder names on "_"
REPO_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--probts_config", required=True)
    p.add_argument("--gam_config", default=str(REPO_DIR / "src/configs/gams/gams_autoreg.yaml"))
    p.add_argument("--dataset", default=None)
    p.add_argument("--context_length", type=int, default=None)
    p.add_argument("--prediction_length", type=int, default=None)
    p.add_argument("--data_path", default=str(REPO_DIR / "ptsbenchmark/datasets"))
    p.add_argument("--root_dir", default=str(REPO_DIR / "ptsbenchmark/log_dir_gams"))
    p.add_argument("--split_val", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lags", type=int, nargs="+", default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--no_ridge", action="store_true", help="Keep the Dense(1) head")
    return p.parse_args()


def build_configs(args):
    """ProbTS yaml + GAM yaml + command line -> data args, GAMConfig, TrainConfig, seed."""
    with open(args.probts_config) as f:
        pcfg = yaml.safe_load(f)
    with open(args.gam_config) as f:
        gcfg = yaml.safe_load(f) or {}

    dm = dict(pcfg["data"]["data_manager"]["init_args"])
    for key in ("dataset", "context_length", "prediction_length", "split_val"):
        if getattr(args, key) is not None:
            dm[key] = getattr(args, key)
    data_args = {"dataset": dm["dataset"], "path": args.data_path,
                 "context_length": dm.get("context_length"),
                 "prediction_length": dm.get("prediction_length"),
                 "split_val": dm.get("split_val", True), "scaler": dm.get("scaler", "standard")}

    seed = args.seed if args.seed is not None else pcfg.get("seed_everything", 0)

    model_kw = dict(gcfg.get("model", {}))
    if args.lags is not None:
        model_kw["lags"] = args.lags
    gam_cfg = GAMConfig(**model_kw, seed=seed)

    trainer, data = pcfg.get("trainer", {}), pcfg.get("data", {})
    train_kw = dict(gcfg.get("training", {}))
    if args.learning_rate is not None:
        train_kw["learning_rate"] = args.learning_rate
    if args.no_ridge:
        train_kw["fit_ridge"] = False
    train_cfg = TrainConfig(
        max_epochs=args.max_epochs or trainer.get("max_epochs", 50),
        batches_per_epoch=trainer.get("limit_train_batches", 100),
        batch_size=data.get("batch_size", 32),
        accumulate_grad_batches=trainer.get("accumulate_grad_batches", 1),
        eval_batch_size=data.get("test_batch_size", 32),
        quantiles_num=pcfg.get("model", {}).get("quantiles_num", 20),
        seed=seed, **train_kw)
    return data_args, gam_cfg, train_cfg, seed


def result_tag(dataset, context_length, prediction_length, seed):
    """Folder name in the ProbTS format: <dataset>_<model>_CTX<c>_PRED<p>_seed<s>."""
    return "_".join([dataset, MODEL_NAME, f"CTX{context_length}", f"PRED{prediction_length}",
                     f"seed{seed}"])


def check_lags(lags, meta):
    if max(lags) > meta.history_length:
        raise ValueError(f"max lag {max(lags)} > history_length {meta.history_length}")
    if max(lags) > meta.context_length:
        print(f"WARNING: max lag {max(lags)} > context_length {meta.context_length}: the GAM "
              f"uses values older than the window DLinear sees.", flush=True)


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    data_args, gam_cfg, train_cfg, seed = build_configs(args)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. data
    t0 = time.time()
    data = load_probts_data(**data_args)
    meta = data.meta
    check_lags(gam_cfg.lags, meta)
    tag = result_tag(meta.dataset, meta.context_length, meta.prediction_length, seed)
    save_dir = Path(args.root_dir) / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] C={meta.target_dim} history={meta.history_length} H={meta.prediction_length} "
          f"train_windows={len(data.train)} val_windows={len(data.val) if data.val else 0} "
          f"test_windows={len(data.test)} load_time={time.time() - t0:.1f}s", flush=True)

    # 2. model
    model = GAMAutoReg(gam_cfg, n_channels=meta.target_dim)

    # 3. training (+ ridge)
    result = train_gam(model, data, train_cfg)

    # 4. test
    t_test = time.time()
    use_ridge = model.ridge is not None
    test = evaluate_probts(model, data.test, data.scaler, meta.freq, train_cfg.eval_batch_size,
                           "test", train_cfg.quantiles_num, use_ridge=use_ridge)
    test_time = time.time() - t_test
    test_dense = (evaluate_probts(model, data.test, data.scaler, meta.freq,
                                  train_cfg.eval_batch_size, "test", train_cfg.quantiles_num,
                                  use_ridge=False) if use_ridge else test)
    print(f"[{tag}] test_ND={test['test_ND']:.4f} test_CRPS={test['test_CRPS']:.4f} "
          f"(dense head: {test_dense['test_CRPS']:.4f})", flush=True)

    # 5. save
    pd.DataFrame([{**test, "horizon": str(meta.prediction_length)}]).to_csv(
        save_dir / "horizons_results.csv", index="idx")
    with open(save_dir / "gam_state.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)
    try:
        active = model.active_features()
    except Exception as e:                         # informative only
        active = {"error": repr(e)}
    summary = {
        "tag": tag, "model": MODEL_NAME, "meta": asdict(meta), "data_args": data_args,
        "gam_config": asdict(gam_cfg), "train_config": asdict(train_cfg),
        "n_features": model.n_features,
        "n_trainable_params": int(sum(np.prod(np.shape(v)) for v in model.tv)),
        "train": asdict(result), "test": test, "test_dense_head": test_dense,
        "test_head": "ridge" if use_ridge else "dense", "test_time_s": test_time,
        "active_features": active,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{tag}] saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
