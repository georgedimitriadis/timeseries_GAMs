"""
Step 1 (dysts) — data for the autoregressive GAM on the dysts benchmark.

Train and test exactly as run_nbeats_on_dysts_data_for_paper.py:
  * file      dysts_data/dysts_data/data/train_multivariate__pts_per_period_100__periods_12.json.gz
              (train_file="train" is what the N-BEATS script reads)
  * system    one trajectory, values [1200, D] (D = 3..10), noise-free, step dt
  * split     split_point = int(5/6 * 1200) = 1000
    train     values[:1000]                              -> TrainWindows (same 1000 points as N-BEATS)
    test      context values[1000 - L : 1000], target values[1000:1200]  (200 free-running steps)

Validation (epoch selection, RidgeCV's PredefinedSplit) comes from the OTHER file,
test_multivariate_...json.gz (val_file="test"): same systems and dt, a trajectory
from a different initial condition, so it overlaps neither the training points
nor the scored target.
    val       for each s in val_starts: context val_values[s - L : s],
              target val_values[s : s + val_horizon]
              default val_starts=(1000,), val_horizon=200: the same position as the test target
  L = history_length (values before a forecast start given to the model).

    d = load_dysts_system("Lorenz", data_dir, history_length=100, train_horizon=200)
    d.train  TrainWindows  (series [1000, D], windows of history_length + train_horizon)
    d.val    EvalWindows   past [len(val_starts), L, D], future [len(val_starts), val_horizon, D]
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
    split_point: int                        # 1000: start of the scored forecast; train = values[:1000]
    val_starts: tuple                       # target starts of the val windows (val trajectory)
    val_file: str                           # file of the val trajectory ("test")
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
def eval_windows(values: np.ndarray, starts, horizon: int, history_length: int) -> EvalWindows:
    """Windows at target starts s: past values[s-L:s], future values[s:s+horizon]."""
    T = values.shape[0]
    for s in starts:
        if s - history_length < 0 or s + horizon > T:
            raise ValueError(f"window start {s}: needs [{s - history_length}, {s + horizon}) "
                             f"inside [0, {T})")
    past = np.stack([values[s - history_length:s] for s in starts]).astype(np.float32)   # [N, L, D]
    future = np.stack([values[s:s + horizon] for s in starts]).astype(np.float32)        # [N, H, D]
    return EvalWindows(past=past, future=future, future_observed=np.ones_like(future),
                       past_s=past, future_s=future)


def split_point_of(T: int) -> int:
    """Start of the scored forecast, as the N-BEATS script: int(5/6 * T)."""
    return int(SPLIT_NUM / SPLIT_DEN * T)


def load_dysts_system(name: str, data_dir, history_length: int = 100, train_horizon: int = 200,
                      val_starts=(200, 400, 600, 800, 1000), val_horizon: int = 200,
                      train_file: str = "train", val_file: str = "test", lyapunov_file=None,
                      equation_data: Optional[dict] = None,
                      val_equation_data: Optional[dict] = None) -> DystsSystemData:
    """
    name               dysts system, e.g. "Lorenz"
    history_length     L: values before each forecast start given to the model (>= max lag)
    train_horizon      rollout length of the training windows (free-running, no teacher forcing)
    val_starts         target starts of the val windows in the val_file trajectory
    val_horizon        length of each val target
    train_file         file with the train/test trajectory ("train", as the N-BEATS script)
    val_file           file with the val trajectory (a different initial condition)
    equation_data, val_equation_data
                       the loaded json files (pass them when looping over systems to read
                       each file once)
    """
    if equation_data is None:
        equation_data = load_gz_json(data_path(data_dir, train_file))
    if val_equation_data is None:
        val_equation_data = load_gz_json(data_path(data_dir, val_file))
    sys_data, val_data = equation_data[name], val_equation_data[name]
    values = np.asarray(sys_data["values"], dtype=np.float64)                 # [T, D]
    val_values = np.asarray(val_data["values"], dtype=np.float64)
    T, D = values.shape
    if val_values.shape[1] != D or not np.isclose(float(val_data["dt"]), float(sys_data["dt"])):
        raise ValueError(f"{name}: val trajectory has a different dimension or dt")

    split_point = split_point_of(T)
    if history_length > split_point - train_horizon:
        raise ValueError(f"{name}: history_length {history_length} + train_horizon {train_horizon} "
                         f"> train length {split_point}")
    series = values[:split_point].astype(np.float32)
    train = TrainWindows(series=series, observed=np.ones_like(series),
                         starts=valid_starts(split_point, history_length, train_horizon),
                         history_length=history_length, prediction_length=train_horizon)

    val_starts = tuple(int(v) for v in val_starts)
    val = eval_windows(val_values, val_starts, val_horizon, history_length)
    test = eval_windows(values, (split_point,), T - split_point, history_length)

    lam = load_lyapunov(lyapunov_file).get(name) if lyapunov_file else None
    return DystsSystemData(name=name, dt=float(sys_data["dt"]), lyapunov=lam, dim=D,
                           history_length=history_length, train_horizon=train_horizon,
                           split_point=split_point, val_starts=val_starts, val_file=val_file,
                           values=values, train=train, val=val, test=test)