"""
Run Darts N-BEATS on the dysts multivariate benchmark, reproducing
`compute_benchmarks_multivariate.py` from dysts_data so the output is directly
comparable to the stored results file

    dysts_data/dysts_data/benchmarks/results/results_test_multivariate__pts_per_period_100__periods_12.json.gz

i.e. the SHORT-horizon (2-period) multivariate benchmark:
  * multivariate, 100 timepoints per period, 12 periods (1200 points/trajectory)
  * noise-free
  * train and test trajectories from DIFFERENT initial conditions (separate files)
  * split: split_point = int(5/6 * len) = 1000 train / 200 forecast (LONG=False)
  * N-BEATS hyperparameters loaded per-system from the tuned hyperparameter JSON
    (input_chunk_length, output_chunk_length) -- NOT re-tuned here
  * scoring via dysts.metrics.compute_metrics (same sMAPE convention as stored)

This is the configuration you CAN compare against with the data in the repo.
It is NOT the Figure 2 configuration (that used periods_60, 1/6 split, a
5000-point horizon; those files are not in the dataset).

The official script fits ONE model on the multivariate training series and
predicts the whole validation block in one call:
    model.fit(y_train_ts); model.predict(len(y_val))
We reproduce exactly that (no per-IC loop, no CV, no Radau re-integration:
the trajectories are read from the precomputed JSON, matching the pipeline).

Paths
-----
Point --data-dir at dysts_data's `dysts/data` (holding the train/test .json.gz)
and --hyperparameter-file at the tuned NBEATS hyperparameters JSON. Defaults are
relative to a typical dysts_data checkout; override as needed.

Usage
-----
    python run_nbeats_dysts.py --systems Lorenz
    python run_nbeats_dysts.py --systems all
    python run_nbeats_dysts.py --systems all --compare   # print stored vs mine
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import warnings

import numpy as np
from tqdm import tqdm

import torch
from darts import TimeSeries
from darts.models import NBEATSModel

import dysts.metrics


# ----------------------------------------------------------------------
# Config matching compute_benchmarks_multivariate.py (LONG = False)
# ----------------------------------------------------------------------
DATANAME = "multivariate__pts_per_period_100__periods_12"
SPLIT_NUM, SPLIT_DEN = 5, 6           # split_point = int(1/6 * len)
MODEL_NAME = "NBEATSModel"


def silence_logs():
    warnings.filterwarnings("ignore")
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning", "darts",
                 "pytorch_lightning.utilities.rank_zero",
                 "pytorch_lightning.accelerators.cuda"):
        logging.getLogger(name).setLevel(logging.ERROR)


def load_gz_json(path):
    with gzip.open(path, mode="r") as f:
        return json.loads(f.read())


def get_trainer_kwargs():
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.set_float32_matmul_precision("high")
        gpu = {"accelerator": "gpu", "devices": [0]}
    else:
        gpu = {"accelerator": "cpu"}
    gpu.update({"enable_progress_bar": False, "logger": False,
                "enable_model_summary": False})
    return gpu, has_gpu

def smape_0_200(y_true, y_pred, eps=1e-10):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(200 * np.mean(np.abs(y_true - y_pred) /
                               (np.abs(y_true) + np.abs(y_pred) + eps)))

def build_model(hp_entry, trainer_kwargs):
    """Instantiate NBEATSModel from the stored hyperparameters, but override the
    stored pl_trainer_kwargs (which pin someone else's GPU/device count) with
    our own quiet trainer kwargs."""
    hp = dict(hp_entry)
    hp.pop("pl_trainer_kwargs", None)
    hp["pl_trainer_kwargs"] = dict(trainer_kwargs)
    return NBEATSModel(**hp)


def run_system(name, train_data, hp_entry, trainer_kwargs):
    """Reproduce the per-system body of compute_benchmarks_multivariate.py."""
    train_data = np.copy(np.asarray(train_data))
    split_point = int(SPLIT_NUM / SPLIT_DEN * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, _ = TimeSeries.from_values(train_data).split_before(split_point)

    model = build_model(hp_entry, trainer_kwargs)
    model.fit(y_train_ts)

    try:
        y_val_pred = model.predict(len(y_val)).values().squeeze()
    except Exception:
        y_val_pred = np.array([None] * len(y_val))

    # scoring: identical to the pipeline (compute_metrics; ValueError -> None)
    try:
        metrics = dysts.metrics.compute_metrics(y_val, y_val_pred)
        metrics["smape"] = smape_0_200(y_val, y_val_pred)
    except ValueError:
        metrics = dysts.metrics.compute_metrics(y_val, y_val)
        for k in metrics:
            metrics[k] = None

    return {"prediction": np.asarray(y_val_pred).tolist(), **metrics}


def main():
    p = argparse.ArgumentParser(
        description="N-BEATS on dysts, matching compute_benchmarks_multivariate.py (periods_12).")
    p.add_argument("--systems", nargs="+", default=["Lorenz"], help='System name(s) or "all".')
    p.add_argument("--data-dir", type=str,
                   default="dysts_data/dysts_data/data",
                   help="Directory holding train_/test_ *.json.gz")
    p.add_argument("--hyperparameter-file", type=str,
                   default=("dysts_data/dysts_data/benchmarks/hyperparameters/"
                            f"hyperparameters_multivariate_train_{DATANAME}.json"))
    p.add_argument("--stored-results", type=str,
                   default=("dysts_data/dysts_data/benchmarks/results/"
                            f"results_test_{DATANAME}.json.gz"),
                   help="Stored official results, for --compare.")
    p.add_argument("--outfile", type=str, default=f"dysts_data/results/results_{DATANAME}_mine.json")
    p.add_argument("--compare", action="store_true",
                   help="Print stored vs reproduced sMAPE per system.")
    p.add_argument("--verbose", action="verbosity",
                   help="Print individual system information.")
    args = p.parse_args()

    silence_logs()
    trainer_kwargs, has_gpu = get_trainer_kwargs()
    print(f"has gpu: {has_gpu}")

    train_path = os.path.join(args.data_dir, f"train_{DATANAME}.json.gz")
    equation_data = load_gz_json(train_path)          # {system: {"values": [...], ...}}
    all_hp = json.load(open(args.hyperparameter_file))

    if len(args.systems) == 1 and args.systems[0].lower() == "all":
        names = [s for s in equation_data.keys()
                 if s in all_hp and MODEL_NAME in all_hp[s]]
    else:
        names = args.systems

    stored = load_gz_json(args.stored_results) if args.compare else None

    results = {}
    for name in tqdm(names, desc="systems"):
        if name not in equation_data:
            print(f"[{name}] not in data file; skipping")
            continue
        if name not in all_hp or MODEL_NAME not in all_hp[name]:
            print(f"[{name}] no NBEATS hyperparameters; skipping")
            continue
        try:
            res = run_system(name, equation_data[name]["values"],
                             all_hp[name][MODEL_NAME], trainer_kwargs)
        except Exception as e:
            print(f"[{name}] ERROR: {e!r}")
            continue
        results[name] = {MODEL_NAME: res}

        mine = res.get("smape")
        if args.compare and stored is not None and name in stored and MODEL_NAME in stored[name]:
            ref = stored[name][MODEL_NAME].get("smape")
            dmsg = ""
            if mine is not None and ref is not None:
                dmsg = f"  (diff {mine - ref:+.3f})"
            tqdm.write(f"[{name}] stored sMAPE={ref}  mine={mine}{dmsg}")
        else:
            tqdm.write(f"[{name}] sMAPE={mine}")

    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} systems to {args.outfile}")

    mine_scores = [v[MODEL_NAME]["smape"] for v in results.values()
                   if v[MODEL_NAME].get("smape") is not None]
    if mine_scores:
        print(f"median sMAPE (mine): {np.median(mine_scores):.3f}")
    if args.compare and stored is not None:
        ref_scores = [stored[n][MODEL_NAME]["smape"] for n in results
                      if n in stored and MODEL_NAME in stored[n]
                      and stored[n][MODEL_NAME].get("smape") is not None]
        if ref_scores:
            print(f"median sMAPE (stored): {np.median(ref_scores):.3f}")


if __name__ == "__main__":
    main()