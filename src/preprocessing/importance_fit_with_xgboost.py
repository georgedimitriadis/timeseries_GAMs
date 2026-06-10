"""
Pairwise predictive causality via XGBoost — parallelised over target channels.

For each target channel, fits:
  1. A baseline model  : target's own lags only
  2. N-1 augmented models: baseline features + one candidate channel's lags

Metric: R²_gain = R²(augmented, val) - R²(baseline, val)
        measured on a held-out chronological validation slice.

The candidate with the largest R²_gain is returned as best_predictor.

Parallelism strategy
--------------------
- Outer loop (targets) is distributed across processes via joblib.
- XGBoost n_jobs=1 inside each worker to avoid CPU over-subscription.
- Data is passed as a shared read-only numpy array (zero-copy on Linux/macOS
  via fork; a single copy on Windows via spawn).

Lags used: [24, 96, 192, 336, 720]
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import Optional

from probts.data.data_manager import DataManager, MULTI_VARIATE_DATASETS


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

LAGS = [24, 96, 192, 336, 720]

# XGBoost params used inside every worker.
# n_jobs=1 is intentional — parallelism comes from the process pool, not
# from threading inside each model.
_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",   # "gpu_hist" if you have a GPU
    n_jobs=1,
    random_state=42,
    verbosity=0,
)

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _lag_matrix(series: np.ndarray, lags: list[int], max_lag: int) -> np.ndarray:
    """Return a 2-D array of shape (T - max_lag, len(lags))."""
    T = len(series)
    return np.column_stack([series[max_lag - lag : T - lag] for lag in lags])


# ---------------------------------------------------------------------------
# Per-target worker  (runs in a subprocess)
# ---------------------------------------------------------------------------

def _process_target(
    target: int,
    data: np.ndarray,        # (T, N)  read-only
    lags: list[int],
    n_tr: int,
    n_rows: int,
) -> dict:
    """
    Fit baseline + N-1 augmented models for one target channel.
    Returns a dict with channel, best_predictor, r2_gain.
    """
    max_lag = max(lags)
    T, N = data.shape
    tr_sl = slice(0, n_tr)
    va_sl = slice(n_tr, n_rows)

    y = data[max_lag:, target]

    # --- own-lag features (shared across all augmented models) ---
    X_own = _lag_matrix(data[:, target], lags, max_lag)   # (n_rows, L)

    # --- baseline ---
    def _score(X):
        m = xgb.XGBRegressor(**_XGB_PARAMS)
        m.fit(X[tr_sl], y[tr_sl], eval_set=[(X[va_sl], y[va_sl])], verbose=False)
        return float(r2_score(y[va_sl], m.predict(X[va_sl])))

    base_r2 = _score(X_own)

    # --- candidates ---
    best_gain = -np.inf
    best_chan  = -1

    for cand in range(N):
        if cand == target:
            continue
        X_cand = _lag_matrix(data[:, cand], lags, max_lag)   # (n_rows, L)
        X_aug  = np.concatenate([X_own, X_cand], axis=1)      # (n_rows, 2L)
        gain   = _score(X_aug) - base_r2
        if gain > best_gain:
            best_gain = gain
            best_chan  = cand

    return {
        "channel":        target,
        "best_predictor": best_chan,
        "r2_gain":        round(float(best_gain), 6),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_best_predictors(
    data: np.ndarray | pd.DataFrame,
    lags: list[int] = LAGS,
    val_fraction: float = 0.2,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For each channel, find the single other channel whose past is most
    helpful in predicting it (beyond its own past).

    Parameters
    ----------
    data : array-like, shape (T, N)
        T time steps, N channels.  Stationarise before passing in
        (difference / detrend) if your series have trends or unit roots.
    lags : list[int]
        Lag indices to use as features for both own-history and candidates.
        Default: [24, 96, 192, 336, 720].
    val_fraction : float
        Chronological hold-out fraction for R² evaluation.  Default 0.2.
    n_jobs : int
        Number of parallel worker processes.
        -1 = use all available CPUs (default).
        Set to 1 to disable parallelism (useful for debugging).
    verbose : bool
        Show a tqdm progress bar over target channels.

    Returns
    -------
    pd.DataFrame with columns:
        channel          int   target channel index (0-based)
        best_predictor   int   channel whose lags gave the largest R² gain
        r2_gain          float R²(augmented) - R²(baseline) on val set.
                               ≈ 0   → no detectable predictive coupling
                               > 0.05 → meaningful coupling (rule of thumb)
                               > 0.1  → strong coupling

    Notes
    -----
    * On Linux/macOS, joblib uses fork — the data array is shared read-only
      with near-zero copy overhead.
    * On Windows, joblib uses spawn — the array is serialised once per
      worker process (not per task), so memory overhead is manageable.
    * Total models fitted = N * (1 + (N-1)) = N².
      For N=360 that is ~130 000 XGBoost fits; expect ~20–40 min on a
      modern 8-core machine.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    data = np.asarray(data, dtype=np.float32)

    T, N = data.shape
    max_lag = max(lags)

    if T <= max_lag + 50:
        raise ValueError(
            f"Too few samples: T={T}, max_lag={max_lag}. "
            "Need T > max_lag + 50."
        )

    n_rows = T - max_lag
    n_val  = max(1, int(n_rows * val_fraction))
    n_tr   = n_rows - n_val

    # Determine actual worker count for the progress bar label
    cpu_count = os.cpu_count() or 1
    n_workers = cpu_count if n_jobs == -1 else min(abs(n_jobs), cpu_count)

    if verbose:
        print(f"Dataset: T={T}, N={N} | "
              f"lags={lags} | workers={n_workers} | "
              f"models to fit: {N * N:,}")

    targets = range(N)

    # tqdm wrapper for progress; disable=not verbose keeps the API clean
    target_iter = tqdm(targets, desc="targets", disable=not verbose,
                       total=N, unit="ch")

    results = Parallel(n_jobs=n_jobs, backend="loky", prefer="processes")(
        delayed(_process_target)(t, data, lags, n_tr, n_rows)
        for t in target_iter
    )

    df = pd.DataFrame(results, columns=["channel", "best_predictor", "r2_gain"])
    df.sort_values("channel", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser(description="Pairwise XGBoost causality scan")
    parser.add_argument("--demo",    action="store_true", help="Run synthetic demo")
    parser.add_argument("--T",       type=int, default=3000, help="Time steps (demo)")
    parser.add_argument("--N",       type=int, default=8,    help="Channels (demo)")
    parser.add_argument("--n_jobs",  type=int, default=-1,   help="Worker processes")
    args = parser.parse_args()

    if args.demo:
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((args.T, args.N)).astype(np.float32)
        # Plant two known couplings
        raw[:, 0] += 0.9 * np.roll(raw[:, 1], 24)    # ch1 → ch0 at lag 24
        raw[:, 2] += 0.9 * np.roll(raw[:, 3], 96)    # ch3 → ch2 at lag 96

        t0 = time.perf_counter()
        result = find_best_predictors(raw, n_jobs=args.n_jobs, verbose=True)
        print(f"\nDone in {time.perf_counter()-t0:.1f}s\n")
        print(result.to_string(index=False))
        print("\nExpected: ch0→ch1, ch2→ch3")