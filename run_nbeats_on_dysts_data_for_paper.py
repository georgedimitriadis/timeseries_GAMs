"""
Run Darts N-BEATS on the dysts multivariate benchmark, reproducing
`compute_benchmarks_multivariate.py` from dysts_data so the output is directly
comparable to the stored results file

    dysts_data/dysts_data/benchmarks/results/results_test_multivariate__pts_per_period_100__periods_12.json.gz

Configuration:
  * multivariate, 100 timepoints per period, 12 periods (1200 points/trajectory)
  * noise-free
  * split: split_point = int(5/6 * len) = 1000 train / 200 forecast
  * N-BEATS hyperparameters loaded per-system from the tuned hyperparameter JSON
  * scoring via dysts.metrics.compute_metrics; smape overridden to 0-200 scale


Added metrics: valid forecast horizon
       * valid_horizon_*      per-step sMAPE (0-100 scale) < threshold (default 50);
                              latest step index; time = step * dt (this repo's convention)
       * valid_horizon_cum_*  Gilpin 2023, Appendix A: running-average sMAPE (0-200 scale)
                              over steps 1..t < 50; latest t; time = t * dt
       * *_lyap = time * lambda_max (None if lambda_max is 0 or missing)

Saved models: one per system, <model-dir>/<system>.pt (+ <system>.pt.ckpt, Darts format).
  * --model-dir default: <outfile without .json>_models/  (e.g. results_..._v3.json -> results_..._v3_models/)
  * load: NBEATSModel.load("<model-dir>/<system>.pt")
  * the result entry of each system has "model_path"
  * --no-save-models: metrics only

Usage
-----
    python run_nbeats_on_dysts_data_for_paper.py --systems Lorenz
    python run_nbeats_on_dysts_data_for_paper.py --systems all --compare
    python run_nbeats_on_dysts_data_for_paper.py --systems all --outfile out/results_v1.json \
        --model-dir out/results_v1_models
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from darts import TimeSeries
from darts.models import NBEATSModel

import dysts.metrics
from src.utils import silence_logs, load_gz_json, get_trainer_kwargs, load_lyapunov
from src.metrics import smape_0_200, valid_horizon, valid_horizon_cumulative


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATANAME = "multivariate__pts_per_period_100__periods_12"
SPLIT_NUM, SPLIT_DEN = 5, 6           # split_point = int(5/6 * len)
MODEL_NAME = "NBEATSModel"
DEFAULT_THRESHOLD = 50.0              # per-step sMAPE, 0-100 scale (valid_horizon_*)


def build_model(hp_entry, trainer_kwargs):
    hp = dict(hp_entry)
    hp.pop("pl_trainer_kwargs", None)
    hp["pl_trainer_kwargs"] = dict(trainer_kwargs)
    return NBEATSModel(**hp)


def run_system(name, sys_data, hp_entry, trainer_kwargs, lam, threshold, model_path=None):
    train_data = np.copy(np.asarray(sys_data["values"]))
    dt = float(sys_data["dt"])
    split_point = int(SPLIT_NUM / SPLIT_DEN * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, _ = TimeSeries.from_values(train_data).split_before(split_point)

    model = build_model(hp_entry, trainer_kwargs)
    model.fit(y_train_ts)
    if model_path is not None:
        model.save(model_path)                          # Darts: <path> and <path>.ckpt

    try:
        y_val_pred = model.predict(len(y_val)).values().squeeze()
    except Exception:
        y_val_pred = np.array([None] * len(y_val))

    try:
        metrics = dysts.metrics.compute_metrics(y_val, y_val_pred)
        metrics["smape"] = smape_0_200(y_val, y_val_pred)
        step, t, lyap, per_step = valid_horizon(y_val, y_val_pred, dt, lam, threshold)
        metrics["valid_horizon_step"] = step
        metrics["valid_horizon_time"] = t
        metrics["valid_horizon_lyap"] = lyap
        metrics["horizon_smape_0_100"] = per_step
        c_step, c_t, c_lyap, cum = valid_horizon_cumulative(y_val, y_val_pred, dt, lam, 50.0)
        metrics["valid_horizon_cum_step"] = c_step
        metrics["valid_horizon_cum_time"] = c_t
        metrics["valid_horizon_cum_lyap"] = c_lyap
        metrics["horizon_smape_cum_0_200"] = cum

    except ValueError:
        metrics = dysts.metrics.compute_metrics(y_val, y_val)
        for k in metrics:
            metrics[k] = None
        metrics["valid_horizon_step"] = None
        metrics["valid_horizon_time"] = None
        metrics["valid_horizon_lyap"] = None
        metrics["horizon_smape_0_100"] = None
        metrics["valid_horizon_cum_step"] = None
        metrics["valid_horizon_cum_time"] = None
        metrics["valid_horizon_cum_lyap"] = None
        metrics["horizon_smape_cum_0_200"] = None

    return {"prediction": np.asarray(y_val_pred).tolist(), **metrics, "model_path": model_path}


def main():
    p = argparse.ArgumentParser(
        description="N-BEATS on dysts, matching compute_benchmarks_multivariate.py (periods_12).")
    p.add_argument("--systems", nargs="+", default=["Lorenz"], help='System name(s) or "all".')
    p.add_argument("--data-dir", type=str,
                   default="dysts_data/dysts_data/data")
    p.add_argument("--hyperparameter-file", type=str,
                   default=("dysts_data/dysts_data/benchmarks/hyperparameters/"
                            f"hyperparameters_multivariate_train_{DATANAME}.json"))
    p.add_argument("--stored-results", type=str,
                   default=("dysts_data/dysts_data/benchmarks/results/"
                            f"results_test_{DATANAME}.json.gz"))
    p.add_argument("--lyapunov-file", type=str,
                   default="dysts_data/dysts_data/data/lyapunov_exponents_qr.txt")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="sMAPE threshold on 0-100 scale (default 50).")
    p.add_argument("--outfile", type=str,
                   default=f"dysts_data/results/results_{DATANAME}_mine.json")
    p.add_argument("--model-dir", type=str, default=None,
                   help="folder for the fitted models, one <system>.pt per system "
                        "(default: <outfile without .json>_models)")
    p.add_argument("--no-save-models", action="store_true", help="do not save the fitted models")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-system lines and summary. "
                        "If not set, only the tqdm bar over systems is shown.")
    args = p.parse_args()

    v = args.verbose

    def vprint(*a, **k):
        if v:
            print(*a, **k)

    silence_logs()
    trainer_kwargs, has_gpu = get_trainer_kwargs()
    vprint(f"has gpu: {has_gpu}")

    data_path  = os.path.join(args.data_dir, f"test_{DATANAME}.json.gz")
    equation_data = load_gz_json(data_path )
    all_hp = json.load(open(args.hyperparameter_file))
    lyap = load_lyapunov(args.lyapunov_file)

    if len(args.systems) == 1 and args.systems[0].lower() == "all":
        names = [s for s in equation_data.keys()
                 if s in all_hp and MODEL_NAME in all_hp[s]]
    else:
        names = args.systems

    stored = load_gz_json(args.stored_results) if args.compare else None

    model_dir = None
    if not args.no_save_models:
        model_dir = args.model_dir or os.path.splitext(args.outfile)[0] + "_models"
        os.makedirs(model_dir, exist_ok=True)
        vprint(f"models -> {model_dir}")

    results = {}
    for name in tqdm(names, desc="systems"):
        if name not in equation_data:
            vprint(f"[{name}] not in data file; skipping")
            continue
        if name not in all_hp or MODEL_NAME not in all_hp[name]:
            vprint(f"[{name}] no NBEATS hyperparameters; skipping")
            continue
        lam = lyap.get(name)
        if lam is None:
            vprint(f"[{name}] no lambda_max; valid_horizon_lyap will be None")
        try:
            model_path = os.path.join(model_dir, f"{name}.pt") if model_dir else None
            res = run_system(name, equation_data[name],
                             all_hp[name][MODEL_NAME], trainer_kwargs, lam, args.threshold, model_path)
        except Exception as e:
            vprint(f"[{name}] ERROR: {e!r}")
            continue
        results[name] = {MODEL_NAME: res}

        if v:
            mine = res.get("smape")
            vh = res.get("valid_horizon_lyap")
            if args.compare and stored is not None and name in stored and MODEL_NAME in stored[name]:
                ref = stored[name][MODEL_NAME].get("smape")
                dmsg = f"  (diff {mine - ref:+.3f})" if (mine is not None and ref is not None) else ""
                tqdm.write(f"[{name}] stored sMAPE={ref}  mine={mine}{dmsg}  vh_lyap={vh}")
            else:
                tqdm.write(f"[{name}] sMAPE={mine}  vh_lyap={vh}")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=2)
    vprint(f"\nWrote {len(results)} systems to {args.outfile}")

    mine_scores = [v[MODEL_NAME]["smape"] for v in results.values()
                   if v[MODEL_NAME].get("smape") is not None]
    if mine_scores:
        vprint(f"median sMAPE (mine): {np.median(mine_scores):.3f}")
    vh_scores = [v[MODEL_NAME]["valid_horizon_lyap"] for v in results.values()
                 if v[MODEL_NAME].get("valid_horizon_lyap") is not None]
    if vh_scores:
        vprint(f"median valid_horizon_lyap: {np.median(vh_scores):.3f}")
    cum_scores = [v[MODEL_NAME]["valid_horizon_cum_lyap"] for v in results.values()
                  if v[MODEL_NAME].get("valid_horizon_cum_lyap") is not None]
    if cum_scores:
        vprint(f"median valid_horizon_cum_lyap: {np.median(cum_scores):.3f}")
    if args.compare and stored is not None:
        ref_scores = [stored[n][MODEL_NAME]["smape"] for n in results
                      if n in stored and MODEL_NAME in stored[n]
                      and stored[n][MODEL_NAME].get("smape") is not None]
        if ref_scores:
            vprint(f"median sMAPE (stored): {np.median(ref_scores):.3f}")


if __name__ == "__main__":
    main()