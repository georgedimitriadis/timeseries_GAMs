"""
Step 1 (dysts) — data for the autoregressive GAM on the dysts benchmark.

Same data and split as run_nbeats_on_dysts_data_for_paper.py:
  * file      dysts_data/dysts_data/data/<split_file>_multivariate__pts_per_period_100__periods_12.json.gz
              (split_file="train" is what the N-BEATS script reads)
  * system    one trajectory, values [1200, D] (D = 3..10), noise-free, step dt
  * split     split_point = int(5/6 * 1200) = 1000
              -> the first 1000 points are for fitting, the last 200 are forecast
                 (free-running, 200 steps) and scored

The GAM additionally needs a validation window (epoch selection, RidgeCV's
PredefinedSplit). It is carved from the end of the fitting part:
              val_start = split_point - val_size        (default 1000 - 200 = 800)
    train     values[:val_start]                         -> TrainWindows
    val       context values[val_start - L : val_start], target values[val_start : split_point]
    test      context values[split_point - L : split_point], target values[split_point:]
  L = history_length (values before a forecast start given to the model).
  val_size=0 -> no validation window, train on values[:split_point].

    d = load_dysts_system("Lorenz", data_dir, history_length=100, train_horizon=200)
    d.train  TrainWindows  (series [T_train, D], windows of history_length + train_horizon)
    d.val    EvalWindows   1 window: past [1, L, D], future [1, val_size, D]
    d.test   EvalWindows   1 window: past [1, L, D], future [1, 200, D]

Values are kept in the original scale (no external scaler): past_s == past and
future_s == future, so the same EvalWindows / GAM code as for ProbTS applies;
the GAM's own per-channel MinMax scaler maps them to [-1, 1].
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from our_models.gam_autoreg.windows import EvalWindows, TrainWindows, valid_starts

DATANAME = "multivariate__pts_per_period_100__periods_12"
SPLIT_NUM, SPLIT_DEN = 5, 6                 # split_point = int(5/6 * len), as the N-BEATS script


# ----------------------------------------------------------------------
# Container
# ----------------------------------------------------------------------
@dataclass
class DystsSystemData:
    name: str
    dt: float
    lyapunov: Optional[float]               # lambda_max (QR file), None if missing
    dim: int                                # D
    history_length: int                     # L
    train_horizon: int                      # rollout length of the training windows
    split_point: int                        # 1000: start of the scored forecast
    val_start: int                          # split_point - val_size
    values: np.ndarray = field(repr=False)  # [T, D] full trajectory
    train: TrainWindows = field(repr=False)
    val: Optional[EvalWindows] = field(repr=False)
    test: EvalWindows = field(repr=False)

    @property
    def test_horizon(self) -> int:
        return self.test.future.shape[1]


# ----------------------------------------------------------------------
# File helpers
# ----------------------------------------------------------------------
def data_path(data_dir, split_file: str = "train") -> Path:
    return Path(data_dir) / f"{split_file}_{DATANAME}.json.gz"


def load_gz_json(path):
    with gzip.open(path, mode="r") as f:
        return json.loads(f.read())


def load_lyapunov(path) -> dict:
    """'Name: value' lines; first occurrence wins (same rule as src/utils.load_lyapunov)."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            if k in out:
                continue
            try:
                out[k] = float(v.strip())
            except ValueError:
                continue
    return out


def list_systems(data_dir, split_file: str = "train") -> list:
    return list(load_gz_json(data_path(data_dir, split_file)).keys())


def nbeats_lookback(hp_file, name: str) -> Optional[int]:
    """Tuned N-BEATS input_chunk_length of a system (None if not in the file)."""
    with open(hp_file) as f:
        hp = json.load(f)
    return hp.get(name, {}).get("NBEATSModel", {}).get("input_chunk_length")


# ----------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------
def _eval_window(values: np.ndarray, start: int, horizon: int, history_length: int) -> EvalWindows:
    """One window: past values[start-L:start], future values[start:start+horizon]."""
    past = values[start - history_length:start][None].astype(np.float32)      # [1, L, D]
    future = values[start:start + horizon][None].astype(np.float32)           # [1, H, D]
    return EvalWindows(past=past, future=future, future_observed=np.ones_like(future),
                       past_s=past, future_s=future)


def split_indices(T: int, val_size: int):
    """split_point (start of the scored forecast) and val_start."""
    split_point = int(SPLIT_NUM / SPLIT_DEN * T)
    return split_point, split_point - val_size


def load_dysts_system(name: str, data_dir, history_length: int = 100, train_horizon: int = 200,
                      val_size: int = 200, split_file: str = "train",
                      lyapunov_file=None, equation_data: Optional[dict] = None) -> DystsSystemData:
    """
    name            dysts system, e.g. "Lorenz"
    history_length  L: values before each forecast start given to the model (>= max lag)
    train_horizon   rollout length of the training windows (free-running, no teacher forcing)
    val_size        length of the validation target carved from the end of the fitting part
                    (0 = no validation window)
    equation_data   the loaded json (pass it when looping over systems to read the file once)
    """
    if equation_data is None:
        equation_data = load_gz_json(data_path(data_dir, split_file))
    sys_data = equation_data[name]
    values = np.asarray(sys_data["values"], dtype=np.float64)                 # [T, D]
    T, D = values.shape
    split_point, val_start = split_indices(T, val_size)

    train_end = val_start if val_size > 0 else split_point
    if history_length > train_end - train_horizon:
        raise ValueError(f"{name}: history_length {history_length} + train_horizon {train_horizon} "
                         f"> train length {train_end}")
    series = values[:train_end].astype(np.float32)
    train = TrainWindows(series=series, observed=np.ones_like(series),
                         starts=valid_starts(train_end, history_length, train_horizon),
                         history_length=history_length, prediction_length=train_horizon)

    val = _eval_window(values, val_start, val_size, history_length) if val_size > 0 else None
    test = _eval_window(values, split_point, T - split_point, history_length)

    lam = load_lyapunov(lyapunov_file).get(name) if lyapunov_file else None
    return DystsSystemData(name=name, dt=float(sys_data["dt"]), lyapunov=lam, dim=D,
                           history_length=history_length, train_horizon=train_horizon,
                           split_point=split_point, val_start=val_start, values=values,
                           train=train, val=val, test=test)
