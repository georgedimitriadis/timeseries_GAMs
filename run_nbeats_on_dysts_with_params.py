"""
Run Darts N-BEATS on the dysts benchmark, following the N-BEATS path of
dysts_data's `darts_benchmarks.py`, with a few changes to printouts, output
folders, and a cached cross-validation lookback.

Protocol (per system), inherited verbatim from the benchmark
------------------------------------------------------------
* Integrate 1000 pts at pts_per_period=10; np.random.seed(0); draw ICs without
  replacement from that pool. The benchmark draws 20 (train). Here we draw 40
  when needed: first 20 = train ICs (bit-identical to the benchmark, same seed
  and draw order), next 20 = forecast ICs (only used by split=separate_ic).
* Each IC: set eq.ic, integrate training_length + forecast_length (512 + 300)
  points, timescale="Lyapunov", method="Radau", pts_per_period (default 30),
  atol=rtol=1e-12.
* N-BEATS: output_chunk_length=1 fixed; cross-validate input_chunk_length in
  {5,25,50,75,100} on an 85/15 split inside each training trajectory, scored
  with the benchmark's own sMAPE; pick the argmin.
* Refit on the full 512-pt training trajectory with the best lookback and
  predict 300 steps; score sMAPE against the true 300-pt continuation.

The four dataset parameters (the paper's 16 = 2^4 sets per system)
-----------------------------------------------------------------
  granularity  : pts_per_period                     (default 30)
  noise        : 0 = noise-free (benchmark default), 1 = Brownian noise applied
  view         : multivariate (default) | univariate
  split        : temporal (default) | separate_ic

  * granularity, view, noise are the paper's axes directly.
  * split is the axis we implement DIFFERENTLY from the paper. In the paper's
    *dataset*, train and test are two separate trajectories from different ICs
    (a real binary). The benchmark *code*, by contrast, makes ONE trajectory
    per IC and splits it in time: train = first 512, forecast target = last 300
    (points right after training). We expose both:
      - temporal    : benchmark behaviour; the 300-pt target is the immediate
                      continuation of the training series (same IC).
      - separate_ic : the paper's semantics; the 300-pt target comes from a
                      DIFFERENT IC's trajectory. Model trained on train-IC i
                      forecasts forecast-IC i. To keep the target the same
                      "distance from IC" as training (removing a burn-in
                      confound), the forecast trajectory is also integrated for
                      512+300; the model is seeded with its points
                      [512-lookback:512] and predicts [512:812].  (option "b")

NOTE ON NOISE: the benchmark scripts you have (darts_benchmarks.py,
generate_trajectories.py) integrate NOISE-FREE -- there is no noise value in
that code to copy. So noise=1 applies dysts' make_trajectory noise with the
amplitude set by --noise-amplitude (default marked below). Drop in the real
2021 value if you have it.

Usage
-----
    python run_nbeats_dysts.py --systems Lorenz
    python run_nbeats_dysts.py --systems Lorenz --split separate_ic
    python run_nbeats_dysts.py --systems all --max-systems 5 --univariate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import numpy as np
from tqdm import tqdm

import dysts.flows
from dysts.systems import get_attractor_list

import torch
from darts import TimeSeries
from darts.models.forecasting.nbeats import NBEATSModel


# ----------------------------------------------------------------------
# Constants copied verbatim from darts_benchmarks.py
# ----------------------------------------------------------------------
NUM_IC = 20
TRAINING_LENGTH = 512
FORECAST_LENGTH = 300
IC_SAMPLE_LENGTH = 1000
IC_SAMPLE_PTS_PER_PERIOD = 10

NBEATS_FIXED_HP = {"output_chunk_length": 1}
NBEATS_LOOKBACK_GRID = [5, 25, 50, 75, 100]
CV_SPLIT_FRAC = 0.85

# Applied only when noise==1. The benchmark has no value to copy; set your own.
DEFAULT_NOISE_AMPLITUDE = 0.0  # <-- placeholder; --noise-amplitude overrides


# ----------------------------------------------------------------------
# The benchmark's exact sMAPE
# ----------------------------------------------------------------------
def smape(y, yhat):
    """Symmetric mean absolute percentage error, verbatim from the benchmark."""
    assert len(yhat) == len(y)
    n = len(y)
    err = np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)) * 100
    return (2 / n) * np.sum(err)


# ----------------------------------------------------------------------
# Quiet: suppress Lightning / Darts chatter
# ----------------------------------------------------------------------
def silence_logs():
    warnings.filterwarnings("ignore")
    for name in ("pytorch_lightning", "lightning.pytorch", "lightning",
                 "pytorch_lightning.utilities.rank_zero",
                 "pytorch_lightning.accelerators.cuda", "darts"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")


def get_gpu_params():
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        torch.set_float32_matmul_precision("high")
        acc = {"accelerator": "gpu", "devices": [0]}
    else:
        acc = {"accelerator": "cpu"}
    # quiet trainer
    acc.update({"enable_progress_bar": False, "logger": False,
                "enable_model_summary": False})
    return acc, has_gpu


def make_nbeats(hp, trainer_kwargs):
    return NBEATSModel(**hp, pl_trainer_kwargs=dict(trainer_kwargs))


# ----------------------------------------------------------------------
# Parameter bundle + naming
# ----------------------------------------------------------------------
class RunParams:
    def __init__(self, pts_per_period, noise, univariate, split, noise_amplitude):
        self.pts_per_period = pts_per_period
        self.noise = int(noise)                      # 0 / 1
        self.univariate = bool(univariate)
        self.split = split                           # "temporal" | "separate_ic"
        self.noise_amplitude = noise_amplitude

    @property
    def view(self):
        return "univariate" if self.univariate else "multivariate"

    def run_name(self, system):
        # underscores only, no dots; split rendered without underscore inside
        split_tag = "separateic" if self.split == "separate_ic" else "temporal"
        return (f"{system}_ppp{self.pts_per_period}_noise{self.noise}"
                f"_{self.view}_{split_tag}")

    def cv_key(self, system):
        # full parameter set per your spec (same as folder name)
        return self.run_name(system)


# ----------------------------------------------------------------------
# CV cache
# ----------------------------------------------------------------------
def cv_cache_path(base_dir):
    return os.path.join(base_dir, "cv_results", "lookbacks.json")


def load_cv_cache(base_dir):
    path = cv_cache_path(base_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cv_cache(base_dir, cache):
    path = cv_cache_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


# ----------------------------------------------------------------------
# Integration
# ----------------------------------------------------------------------
def integrate_ic(eq, ic, params, n_points):
    eq.ic = np.copy(ic)
    kwargs = {"pts_per_period": params.pts_per_period, "atol": 1e-12, "rtol": 1e-12}
    if params.noise == 1:
        kwargs["noise"] = params.noise_amplitude
    return eq.make_trajectory(n_points, timescale="Lyapunov", method="Radau", **kwargs)


def sample_ics(eq, n_needed):
    """np.random.seed(0); draw n_needed ICs w/o replacement from the 1000-pt pool.
    First NUM_IC are the benchmark's train ICs (identical draw order)."""
    ic_traj = eq.make_trajectory(IC_SAMPLE_LENGTH, pts_per_period=IC_SAMPLE_PTS_PER_PERIOD,
                                 atol=1e-12, rtol=1e-12)
    if ic_traj is None:
        raise RuntimeError("IC-sampling trajectory failed to integrate")
    np.random.seed(0)
    sel = np.random.choice(range(IC_SAMPLE_LENGTH), size=n_needed, replace=False).astype(int)
    return ic_traj[sel, :]


def build_train_trajectories(eq, train_ics, params):
    trajs = []
    for ic in train_ics:
        traj = integrate_ic(eq, ic, params, TRAINING_LENGTH + FORECAST_LENGTH)
        if traj is None:
            raise RuntimeError("training integration failed")
        trajs.append(traj)
    trajs = np.array(trajs)
    return trajs[:, :TRAINING_LENGTH], trajs[:, TRAINING_LENGTH:]


def build_forecast_trajectories(eq, forecast_ics, params):
    """For separate_ic: full 512+300 integrations of the forecast ICs (option b)."""
    trajs = []
    for ic in forecast_ics:
        traj = integrate_ic(eq, ic, params, TRAINING_LENGTH + FORECAST_LENGTH)
        if traj is None:
            raise RuntimeError("forecast-IC integration failed")
        trajs.append(traj)
    return np.array(trajs)


# ----------------------------------------------------------------------
# Cross-validation with a single tqdm bar over 20 IC x 5 lookbacks
# ----------------------------------------------------------------------
def _fit_predict(train_data, horizon, hp, trainer_kwargs, univariate):
    if univariate:
        preds = []
        for j in range(train_data.shape[1]):
            m = make_nbeats(hp, trainer_kwargs)
            m.fit(TimeSeries.from_values(train_data[:, j][:, None]))
            preds.append(m.predict(horizon).values().squeeze().copy())
        return np.array(preds).T
    m = make_nbeats(hp, trainer_kwargs)
    m.fit(TimeSeries.from_values(train_data.copy()))
    return m.predict(horizon).values().squeeze().copy()


def cross_validate_lookback(traj_train, trainer_kwargs, params, system):
    split_index = int(CV_SPLIT_FRAC * traj_train.shape[1])
    n_ic = traj_train.shape[0]
    scores = {lb: [] for lb in NBEATS_LOOKBACK_GRID}

    total = n_ic * len(NBEATS_LOOKBACK_GRID)
    bar = tqdm(total=total, desc=f"CV {system}", leave=True)
    for lb in NBEATS_LOOKBACK_GRID:
        hp = {"input_chunk_length": lb, **NBEATS_FIXED_HP}
        for i in range(n_ic):
            train_data = traj_train[i, :split_index].copy()
            test_data = traj_train[i, split_index:].copy()
            try:
                y_pred = _fit_predict(train_data, len(test_data), hp, trainer_kwargs,
                                      params.univariate)
                y_true = TimeSeries.from_values(test_data).values().squeeze().copy()
                scores[lb].append(smape(y_pred, y_true))
            except Exception:
                scores[lb].append(np.nan)
            bar.update(1)
    bar.close()

    mean_scores = {lb: np.nanmean(s) for lb, s in scores.items()}
    best_lb = min(mean_scores, key=mean_scores.get)
    return best_lb, mean_scores


# ----------------------------------------------------------------------
# Final forecast + scoring
# ----------------------------------------------------------------------
def forecast_temporal(traj_train, traj_true, best_lb, trainer_kwargs, params):
    hp = {"input_chunk_length": best_lb, **NBEATS_FIXED_HP}
    per_ic, preds = [], []
    for i in range(len(traj_train)):
        try:
            yp = _fit_predict(traj_train[i], FORECAST_LENGTH, hp, trainer_kwargs,
                              params.univariate)
        except Exception:
            yp = np.nan * np.ones((FORECAST_LENGTH, traj_train.shape[-1]))
        preds.append(yp)
        yt = traj_true[i]
        per_ic.append(np.nan if np.any(~np.isfinite(yp))
                      else smape(yt.reshape(-1), yp.reshape(-1)))
    return np.array(preds), per_ic


def forecast_separate_ic(traj_train, forecast_trajs, best_lb, trainer_kwargs, params):
    """Train on train-IC i, forecast forecast-IC i (option b):
    seed with forecast traj [512-lb:512], target [512:812]."""
    hp = {"input_chunk_length": best_lb, **NBEATS_FIXED_HP}
    per_ic, preds = [], []
    seed_start = TRAINING_LENGTH - best_lb
    for i in range(len(traj_train)):
        target = forecast_trajs[i, TRAINING_LENGTH:TRAINING_LENGTH + FORECAST_LENGTH]
        try:
            if params.univariate:
                cols = []
                for j in range(traj_train.shape[-1]):
                    m = make_nbeats(hp, trainer_kwargs)
                    m.fit(TimeSeries.from_values(traj_train[i][:, j][:, None]))
                    seed = forecast_trajs[i, seed_start:TRAINING_LENGTH, j][:, None]
                    yp = m.predict(FORECAST_LENGTH,
                                   series=TimeSeries.from_values(seed)).values().squeeze().copy()
                    cols.append(yp)
                yp = np.array(cols).T
            else:
                m = make_nbeats(hp, trainer_kwargs)
                m.fit(TimeSeries.from_values(traj_train[i].copy()))
                seed = forecast_trajs[i, seed_start:TRAINING_LENGTH]
                yp = m.predict(FORECAST_LENGTH,
                               series=TimeSeries.from_values(seed)).values().squeeze().copy()
        except Exception:
            yp = np.nan * np.ones((FORECAST_LENGTH, traj_train.shape[-1]))
        preds.append(yp)
        per_ic.append(np.nan if np.any(~np.isfinite(yp))
                      else smape(target.reshape(-1), yp.reshape(-1)))
    return np.array(preds), per_ic


# ----------------------------------------------------------------------
# Per-system driver
# ----------------------------------------------------------------------
def run_system(system, params, base_dir, trainer_kwargs, cv_cache):
    run_name = params.run_name(system)
    out_dir = os.path.join(base_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    eq = getattr(dysts.flows, system)()

    # --- ICs ---
    n_needed = 2 * NUM_IC if params.split == "separate_ic" else NUM_IC
    all_ics = sample_ics(eq, n_needed)
    train_ics = all_ics[:NUM_IC]
    forecast_ics = all_ics[NUM_IC:2 * NUM_IC] if params.split == "separate_ic" else None

    # --- train trajectories ---
    traj_train, traj_true_temporal = build_train_trajectories(eq, train_ics, params)
    np.save(os.path.join(out_dir, f"forecast_{eq.name}_true_dysts"),
            traj_true_temporal, allow_pickle=True)

    # --- CV lookback (cached) ---
    key = params.cv_key(system)
    if key in cv_cache:
        best_lb = cv_cache[key]
        print(f"[{system}] using cached lookback = {best_lb}")
    else:
        best_lb, mean_scores = cross_validate_lookback(traj_train, trainer_kwargs, params, system)
        cv_cache[key] = int(best_lb)
        save_cv_cache(base_dir, cv_cache)
        print(f"[{system}] CV done. mean sMAPE per lookback: "
              f"{ {lb: round(float(v), 3) for lb, v in mean_scores.items()} }")
        print(f"[{system}] chosen lookback = {best_lb}")

    # --- final forecast ---
    if params.split == "separate_ic":
        forecast_trajs = build_forecast_trajectories(eq, forecast_ics, params)
        preds, per_ic = forecast_separate_ic(traj_train, forecast_trajs, best_lb,
                                             trainer_kwargs, params)
        # save the separate-IC targets too, for later analysis
        np.save(os.path.join(out_dir, f"forecast_{eq.name}_true_separateic"),
                forecast_trajs[:, TRAINING_LENGTH:TRAINING_LENGTH + FORECAST_LENGTH],
                allow_pickle=True)
    else:
        preds, per_ic = forecast_temporal(traj_train, traj_true_temporal, best_lb,
                                          trainer_kwargs, params)

    np.save(os.path.join(out_dir, f"forecast_{eq.name}_NBEATS_pred"), preds, allow_pickle=True)

    med = float(np.nanmedian(per_ic))
    print(f"[{system}] TEST  split={params.split}  lookback={best_lb}  "
          f"median sMAPE over {FORECAST_LENGTH} steps = {med:.3f}")
    print(f"[{system}] per-IC sMAPE: {[round(float(s), 2) for s in per_ic]}")

    return {"system": system, "lookback": int(best_lb), "smape_median": med,
            "smape_per_ic": [float(s) for s in per_ic], "out_dir": out_dir}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def resolve_systems(systems, max_systems):
    if len(systems) == 1 and systems[0].lower() == "all":
        names = get_attractor_list("continuous_no_delay")
        return names[:max_systems] if max_systems else names
    return systems


def main():
    p = argparse.ArgumentParser(description="Darts N-BEATS on dysts (paper protocol).")
    p.add_argument("--systems", nargs="+", default=["Lorenz"], help='System name(s) or "all".')
    p.add_argument("--max-systems", type=int, default=None)
    p.add_argument("--base-dir", type=str, default="darts/results")
    # the four dataset parameters
    p.add_argument("--pts-per-period", type=int, default=30)
    p.add_argument("--noise", type=int, choices=[0, 1], default=0)
    p.add_argument("--noise-amplitude", type=float, default=DEFAULT_NOISE_AMPLITUDE,
                   help="Amplitude used when --noise 1 (benchmark had no value; set your own).")
    p.add_argument("--univariate", action="store_true")
    p.add_argument("--split", choices=["temporal", "separate_ic"], default="temporal")
    args = p.parse_args()

    silence_logs()
    trainer_kwargs, has_gpu = get_gpu_params()
    print(f"has gpu: {has_gpu}")

    params = RunParams(args.pts_per_period, args.noise, args.univariate,
                       args.split, args.noise_amplitude)
    os.makedirs(args.base_dir, exist_ok=True)
    cv_cache = load_cv_cache(args.base_dir)

    names = resolve_systems(args.systems, args.max_systems)
    print(f"Running N-BEATS on {len(names)} system(s). base_dir={args.base_dir}")
    print(f"params: ppp={params.pts_per_period} noise={params.noise} "
          f"view={params.view} split={params.split}\n")

    results = {}
    for name in names:
        try:
            results[name] = run_system(name, params, args.base_dir, trainer_kwargs, cv_cache)
        except Exception as e:
            print(f"[{name}] ERROR: {e!r}")
            results[name] = {"system": name, "error": repr(e)}

    scored = [r["smape_median"] for r in results.values() if "smape_median" in r]
    if scored:
        print(f"\nMedian sMAPE across {len(scored)} scored systems: {np.median(scored):.3f}")


if __name__ == "__main__":
    main()