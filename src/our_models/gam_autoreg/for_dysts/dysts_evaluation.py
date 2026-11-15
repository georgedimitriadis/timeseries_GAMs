"""
dysts metrics for the autoregressive GAM, computed exactly as
run_nbeats_on_dysts_data_for_paper.py does for N-BEATS:

    metrics = dysts.metrics.compute_metrics(y_true, y_pred)
    metrics["smape"] = smape_0_200(y_true, y_pred)                     # src/metrics.py
    valid_horizon(y_true, y_pred, dt, lambda_max, threshold=50)        # src/metrics.py
        -> valid_horizon_step / _time / _lyap, horizon_smape_0_100
    valid_horizon_cumulative(..., threshold=50)                        # src/metrics.py
        -> valid_horizon_cum_step / _time / _lyap, horizon_smape_cum_0_200
           (running-average sMAPE, 0-200 scale, as Gilpin 2023, Appendix A)
    a ValueError (e.g. non-finite forecast) -> every metric None

on forecasts in the original scale, y [H, D] per window.

evaluate_dysts() is passed to the dysts training loop (validation, with
functools.partial to fix dt / lambda_max) and used by the run script for the test
window. With several windows (e.g. several val_starts) the numeric metrics are
averaged over windows; if any window failed, the average is None.

Imports: `src.metrics` (the same module the N-BEATS script uses), so the repo root
must be on PYTHONPATH together with src/.
"""

from __future__ import annotations

import warnings
from typing import Optional

import dysts.metrics
import numpy as np

from src.metrics import smape_0_200, valid_horizon, valid_horizon_cumulative
from our_models.gam_autoreg.windows import EvalWindows

DEFAULT_THRESHOLD = 50.0      # sMAPE threshold of the valid horizon, 0-100 scale (N-BEATS script)
CUM_THRESHOLD = 50.0          # running-average sMAPE threshold, 0-200 scale (Gilpin 2023, Appendix A)
_PER_STEP_KEYS = ("horizon_smape_0_100", "horizon_smape_cum_0_200")


def score_window(y_true: np.ndarray, y_pred: np.ndarray, dt: float, lam: Optional[float],
                 threshold: float = DEFAULT_THRESHOLD) -> dict:
    """_score_window without two known dysts.metrics warnings: scipy coherence with
    nperseg 256 > 200 forecast steps (it uses 200), and sqrt of a value just below 0 in a
    distance metric when scoring y_true against itself (failed forecasts)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="nperseg = .* is greater than input length")
        warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
        return _score_window(y_true, y_pred, dt, lam, threshold)


def _score_window(y_true: np.ndarray, y_pred: np.ndarray, dt: float, lam: Optional[float],
                 threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    One forecast, y [H, D] original scale. Same code path as run_system() in
    run_nbeats_on_dysts_data_for_paper.py (including the ValueError fallback).
    """
    try:
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("non-finite forecast")
        metrics = dysts.metrics.compute_metrics(y_true, y_pred)
        metrics["smape"] = smape_0_200(y_true, y_pred)
        step, t, lyap, per_step = valid_horizon(y_true, y_pred, dt, lam, threshold)
        metrics["valid_horizon_step"] = step
        metrics["valid_horizon_time"] = t
        metrics["valid_horizon_lyap"] = lyap
        metrics["horizon_smape_0_100"] = per_step
        c_step, c_t, c_lyap, cum = valid_horizon_cumulative(y_true, y_pred, dt, lam, CUM_THRESHOLD)
        metrics["valid_horizon_cum_step"] = c_step
        metrics["valid_horizon_cum_time"] = c_t
        metrics["valid_horizon_cum_lyap"] = c_lyap
        metrics["horizon_smape_cum_0_200"] = cum
    except ValueError:
        metrics = dysts.metrics.compute_metrics(y_true, y_true)
        for k in metrics:
            metrics[k] = None
        metrics["smape"] = None
        metrics["valid_horizon_step"] = None
        metrics["valid_horizon_time"] = None
        metrics["valid_horizon_lyap"] = None
        metrics["horizon_smape_0_100"] = None
        metrics["valid_horizon_cum_step"] = None
        metrics["valid_horizon_cum_time"] = None
        metrics["valid_horizon_cum_lyap"] = None
        metrics["horizon_smape_cum_0_200"] = None
    return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and k not in _PER_STEP_KEYS
                else v) for k, v in metrics.items()}


def _mean_or_none(values):
    if any(v is None for v in values):
        return None
    return float(np.mean(values))


def evaluate_dysts(model, windows: EvalWindows, dt: float, lam: Optional[float], stage: str,
                   use_ridge: Optional[bool] = None, threshold: float = DEFAULT_THRESHOLD,
                   return_details: bool = False) -> dict:
    """
    Forecast every window (free-running, H = windows.future.shape[1]) and score it.

    Returns {f"{stage}_{metric}": mean over windows} for the scalar metrics
    (e.g. "val_smape", "val_valid_horizon_lyap"), plus, with return_details,
    f"{stage}_windows": [per-window dict with "prediction" and all metrics].
    model: any model with forecast(past, H, use_ridge) -> [N, H, D] in the original scale
    (DystsGAM).
    """
    H = windows.future.shape[1]
    pred = model.forecast(windows.past, H, use_ridge=use_ridge)              # [N, H, D]
    per_window = [score_window(windows.future[i], pred[i], dt, lam, threshold)
                  for i in range(len(windows))]

    scalar_keys = [k for k in per_window[0] if k not in _PER_STEP_KEYS]
    out = {f"{stage}_{k}": _mean_or_none([m[k] for m in per_window]) for k in scalar_keys}
    if return_details:
        out[f"{stage}_windows"] = [{"prediction": np.asarray(pred[i]).tolist(), **m}
                                   for i, m in enumerate(per_window)]
    return out