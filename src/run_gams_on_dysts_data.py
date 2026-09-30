
import argparse
import json
import os

import dysts.metrics
import numpy as np
from tqdm import tqdm

from darts import TimeSeries

from src.utils import silence_logs, load_gz_json, get_trainer_kwargs, load_lyapunov
from src.metrics import smape_0_200, valid_horizon

DATANAME = "multivariate__pts_per_period_100__periods_12"
SPLIT_NUM, SPLIT_DEN = 5, 6           # split_point = int(5/6 * len)
DEFAULT_THRESHOLD = 50.0              # 0-100 scale (paper)

def build_model(hyperparams, kwargs):
    pass

def run_system(sys_data, hp_entry, trainer_kwargs, lam, threshold):
    train_data = np.copy(np.asarray(sys_data["values"]))
    dt = float(sys_data["dt"])
    split_point = int(SPLIT_NUM / SPLIT_DEN * len(train_data))
    y_train, y_val = train_data[:split_point], train_data[split_point:]
    y_train_ts, _ = TimeSeries.from_values(train_data).split_before(split_point)

    model = build_model(hp_entry, trainer_kwargs)
    model.fit(y_train_ts)

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
    except ValueError:
        metrics = dysts.metrics.compute_metrics(y_val, y_val)
        for k in metrics:
            metrics[k] = None
        metrics["valid_horizon_step"] = None
        metrics["valid_horizon_time"] = None
        metrics["valid_horizon_lyap"] = None
        metrics["horizon_smape_0_100"] = None

    return {"prediction": np.asarray(y_val_pred).tolist(), **metrics}

def main():
    p = argparse.ArgumentParser(
        description="GAMS on dysts, matching compute_benchmarks_multivariate.py (periods_12).")
    p.add_argument("--systems", nargs="+", default=["Lorenz"], help='System name(s) or "all".')
    p.add_argument("--data-dir", type=str,
                   default="dysts_data/dysts_data/data")
    p.add_argument("--stored-results", type=str,
                   default=("dysts_data/dysts_data/benchmarks/results/"
                            f"results_test_{DATANAME}.json.gz"))
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="sMAPE threshold on 0-100 scale (default 50).")
    p.add_argument("--outfile", type=str,
                   default=f"src/exps/gams/dysts/results_{DATANAME}_mine.json")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-system lines and summary. "
                        "If not set, only the tqdm bar over systems is shown.")
    args = p.parse_args()

    verbose = args.verbose

    def vprint(*a, **k):
        if verbose:
            print(*a, **k)

    silence_logs()
    trainer_kwargs, has_gpu = get_trainer_kwargs()
    vprint(f"has gpu: {has_gpu}")
    train_path = os.path.join(args.data_dir, f"train_{DATANAME}.json.gz")
    equation_data = load_gz_json(train_path)
    lyap = load_lyapunov(args.lyapunov_file)

    if len(args.systems) == 1 and args.systems[0].lower() == "all":
        names = [s for s in equation_data.keys()]
    else:
        names = args.systems

    stored = load_gz_json(args.stored_results) if args.compare else None

    results = {}
    for name in tqdm(names, desc="systems"):
        if name not in equation_data:
            vprint(f"[{name}] not in data file; skipping")
            continue
        lam = lyap.get(name)
        if lam is None:
            vprint(f"[{name}] no lambda_max; valid_horizon_lyap will be None")
        try:
            res = run_system(equation_data[name], trainer_kwargs, lam, args.threshold)
        except Exception as e:
            vprint(f"[{name}] ERROR: {e!r}")
            continue
        results[name] = res

        if verbose:
            mine = res.get("smape")
            vh = res.get("valid_horizon_lyap")
            if args.compare and stored is not None and name in stored:
                ref = stored[name]['NBEATSModel'].get("smape")
                dmsg = f"  (diff {mine - ref:+.3f})" if (mine is not None and ref is not None) else ""
                tqdm.write(f"[{name}] stored sMAPE={ref}  mine={mine}{dmsg}  vh_lyap={vh}")
            else:
                tqdm.write(f"[{name}] sMAPE={mine}  vh_lyap={vh}")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=2)
    vprint(f"\nWrote {len(results)} systems to {args.outfile}")

    mine_scores = [v["smape"] for v in results.values()
                   if v.get("smape") is not None]
    if mine_scores:
        vprint(f"median sMAPE (mine): {np.median(mine_scores):.3f}")
    vh_scores = [v["valid_horizon_lyap"] for v in results.values()
                 if v.get("valid_horizon_lyap") is not None]
    if vh_scores:
        vprint(f"median valid_horizon_lyap: {np.median(vh_scores):.3f}")
    if args.compare and stored is not None:
        ref_scores = [stored[n]["smape"] for n in results
                      if n in stored and 'NBEATSModel' in stored[n]
                      and stored[n]['NBEATSModel'].get("smape") is not None]
        if ref_scores:
            vprint(f"median sMAPE (stored): {np.median(ref_scores):.3f}")

if __name__ == "__main__":
    main()