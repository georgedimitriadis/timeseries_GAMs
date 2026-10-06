"""
Step 1 — data for the autoregressive GAM on ProbTS datasets.

What this module gives you
--------------------------
    data = load_probts_data(dataset, path, context_length, prediction_length, ...)

    data.train      TrainWindows   standard-scaled train series [T, C] + the
                                   window start indices ProbTS samples from
    data.val        EvalWindows    the exact ProbTS validation windows
    data.test       EvalWindows    the exact ProbTS test windows
    data.scaler     the ProbTS StandardScaler (fitted on the train split)
    data.meta       dataset, freq, target_dim, context/history/prediction length

Every window has the ProbTS layout:
    past   [N, history_length, C]    history_length = context_length + max(freq lags)
    future [N, prediction_length, C]

Helpers:
    to_series / from_series   [N, L, C] <-> [N*C, L]  (one row per channel;
                              the GAM is one model shared by all channels)
    one_step_rows             tabular rows (values at the lags -> next value),
                              the format the original tabular pipeline used
    predefined_split          sklearn PredefinedSplit: train rows = -1,
                              val rows = 0 (for ECRegressor.ps and RidgeCV(cv=...))

Imports
-------
`probts` is imported as a top-level package, so `ptsbenchmark/` must be on
PYTHONPATH (as in run_probts_all_datasets.sh / the Makefile), together with
`src/` for `our_models`.

How the splits match ProbTS (ptsbenchmark/probts/data/data_manager.py)
-----------------------------------------------------------------------
* Scaler: data_manager.scaler, fitted by ProbTS on the train split.
* Train:
    long-term  : dataset_raw[:border_end[0]]
    short-term : grouped train target; with split_val the last
                 num_test_dates * prediction_length steps are cut off, as in
                 data_utils.split_train_val
  Valid window ends t (= forecast start) are
    history_length <= t <= T - prediction_length,
  the range allowed by ProbTS's ExpectedNumInstanceSampler (min_past =
  history_length, min_future = prediction_length). ProbTS draws them at
  random; here batches are drawn uniformly from the same range.
* Val / test: read directly from data_manager.val_iter_dataset /
  test_iter_dataset, so the windows (rolling for long-term, last-date
  windows for short-term) are identical to the ones DLinear is scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from sklearn.model_selection import PredefinedSplit

from probts.data.data_manager import DataManager, MULTI_VARIATE_DATASETS


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------
@dataclass
class Meta:
    dataset: str
    freq: str
    target_dim: int
    context_length: int
    history_length: int
    prediction_length: int


@dataclass
class TrainWindows:
    """Standard-scaled train series and the valid window end positions."""
    series: np.ndarray            # [T, C] float32, standard-scaled
    observed: np.ndarray          # [T, C] float32, 1 = observed, 0 = missing (NaN in raw data)
    starts: np.ndarray            # [n_windows] int: forecast start t of each valid window
    history_length: int           # context_length + max(freq lags)
    prediction_length: int

    def __len__(self):
        return len(self.starts)

    def windows(self, t: np.ndarray):
        """Cut n windows ending at forecast starts `t`.
        Returns past [n, history_length, C], future [n, H, C], future_observed [n, H, C]."""
        t = np.asarray(t)
        past_idx = t[:, None] + np.arange(-self.history_length, 0)[None, :]
        fut_idx = t[:, None] + np.arange(self.prediction_length)[None, :]
        return self.series[past_idx], self.series[fut_idx], self.observed[fut_idx]

    def channel_rows(self, t: np.ndarray, c: np.ndarray):
        """One channel per window: window ending at t[i] for channel c[i].
        Returns past [n, history_length], future [n, H], future_observed [n, H]."""
        t, c = np.asarray(t), np.asarray(c)
        past_idx = t[:, None] + np.arange(-self.history_length, 0)[None, :]
        fut_idx = t[:, None] + np.arange(self.prediction_length)[None, :]
        cc = c[:, None]
        return self.series[past_idx, cc], self.series[fut_idx, cc], self.observed[fut_idx, cc]

    def sample_batch(self, rng: np.random.Generator, batch_size: int):
        """Random batch of windows (uniform over the valid starts, with replacement)."""
        return self.windows(rng.choice(self.starts, size=batch_size, replace=True))

    def iter_all(self, batch_size: int, stride: int = 1):
        """All windows in time order (every `stride`-th start), in batches."""
        starts = self.starts[::stride]
        for i in range(0, len(starts), batch_size):
            yield self.windows(starts[i:i + batch_size])


@dataclass
class EvalWindows:
    """Fixed ProbTS val/test windows. Raw values are kept for the metrics
    (ProbTS scores in the original scale); *_s are standard-scaled."""
    past: np.ndarray              # [N, history_length, C] raw
    future: np.ndarray            # [N, H, C] raw
    future_observed: np.ndarray   # [N, H, C]
    past_s: np.ndarray            # [N, history_length, C] scaled
    future_s: np.ndarray          # [N, H, C] scaled

    def __len__(self):
        return self.past.shape[0]

    def batches(self, batch_size: int):
        """Index slices in the same order/size as ProbTS's test DataLoader
        (needed to reproduce its batch-weighted metric average)."""
        for i in range(0, len(self), batch_size):
            yield slice(i, min(i + batch_size, len(self)))


@dataclass
class ProbTSData:
    meta: Meta
    train: TrainWindows
    val: Optional[EvalWindows]
    test: EvalWindows
    scaler: object = field(repr=False)


# ----------------------------------------------------------------------
# Pure helpers (NumPy only)
# ----------------------------------------------------------------------
def valid_starts(T: int, history_length: int, prediction_length: int) -> np.ndarray:
    """Forecast starts t with a full history before and a full horizon after."""
    return np.arange(history_length, T - prediction_length + 1)


def stsf_train_length(T: int, num_test_dates: int, prediction_length: int, split_val: bool) -> int:
    """Length of the short-term train series after data_utils.split_train_val."""
    return T - num_test_dates * prediction_length if split_val else T


def to_series(x: np.ndarray) -> np.ndarray:
    """[N, L, C] -> [N*C, L]; row n*C + c is channel c of window n."""
    N, L, C = x.shape
    return np.ascontiguousarray(x.transpose(0, 2, 1).reshape(N * C, L))


def from_series(x: np.ndarray, C: int) -> np.ndarray:
    """[N*C, L] -> [N, L, C]."""
    NC, L = x.shape
    return x.reshape(NC // C, C, L).transpose(0, 2, 1)


def series_weights(future_observed: np.ndarray) -> np.ndarray:
    """Loss weights per series row [N*C, H]: minimum over channels of the
    observed mask, as Forecaster.get_weighted_loss, repeated for each channel."""
    N, H, C = future_observed.shape
    return np.repeat(future_observed.min(axis=-1), C, axis=0)


def one_step_rows(past: np.ndarray, future: np.ndarray, lags):
    """
    Tabular rows for one-step-ahead regression on each window and channel.
    Used for a teacher forced training since it provides the TRUE historical data for all windows.
    For every forecast step h (0..H-1) of every series, the inputs are the TRUE
    values at t+h-lag and the target is the true value at t+h.
    past [N, Lp, C], future [N, H, C] -> X [N*C*H, n_lags], y [N*C*H]
    (Rows use true history, i.e. teacher-forced inputs.)
    """
    lags = np.asarray(lags)
    full = np.concatenate([to_series(past), to_series(future)], axis=1)   # [N*C, Lp+H]
    Lp, H = past.shape[1], future.shape[1]
    steps = Lp + np.arange(H)                                             # positions of targets
    X = full[:, steps[:, None] - lags[None, :]]                           # [N*C, H, n_lags]
    y = full[:, steps]                                                    # [N*C, H]
    return X.reshape(-1, len(lags)), y.reshape(-1)


def predefined_split(n_train: int, n_val: int):
    """PredefinedSplit for rows ordered [train rows, val rows]:
    train = -1 (never in a test fold), val = fold 0. Deterministic."""
    return PredefinedSplit(np.concatenate([np.full(n_train, -1), np.zeros(n_val, dtype=int)]))


# ----------------------------------------------------------------------
# Loading via ProbTS
# ----------------------------------------------------------------------
def _scale(scaler, x: np.ndarray) -> np.ndarray:
    return scaler.transform(torch.from_numpy(np.asarray(x, dtype=np.float32))).numpy()


def _collect(iter_dataset, scaler) -> EvalWindows:
    """Materialise a ProbTS val/test iterable dataset into arrays."""
    past, future, obs = [], [], []
    for item in iter_dataset:
        p, f, o = (np.asarray(item[k], dtype=np.float32) for k in
                   ("past_target_cdf", "future_target_cdf", "future_observed_values"))
        if p.ndim == 1:                              # univariate datasets
            p, f, o = p[:, None], f[:, None], o[:, None]
        past.append(p), future.append(f), obs.append(o)
    past, future, obs = np.stack(past), np.stack(future), np.stack(obs)
    return EvalWindows(past, future, obs, _scale(scaler, past), _scale(scaler, future))


def _train_series(dm) -> np.ndarray:
    """Raw train series [T, C] exactly as ProbTS trains on it."""
    if hasattr(dm, "border_end"):                                   # long-term CSV datasets
        return dm.dataset_raw.values[: dm.border_end[0]].astype(np.float32)
    grouped = MultivariateGrouper(max_target_dim=int(dm.target_dim))(dm.dataset_raw.train)
    arr = np.asarray(grouped[0]["target"], dtype=np.float32).T      # [T, C]
    T = stsf_train_length(arr.shape[0], dm.num_test_dates, int(dm.prediction_length), dm.split_val)
    return arr[:T]


def load_probts_data(dataset: str, path: str, context_length: Optional[int] = None,
                     prediction_length: Optional[int] = None, split_val: bool = True,
                     scaler: str = "standard", **dm_kwargs) -> ProbTSData:
    """
    Build the ProbTS DataManager (same arguments as the DLinear runs) and
    return train / val / test windows. Short-term datasets take context and
    prediction length from GluonTS metadata when they are None.
    """

    dm = DataManager(dataset=dataset, path=path, context_length=context_length,
                     prediction_length=prediction_length, split_val=split_val,
                     scaler=scaler, **dm_kwargs)
    if dm.multi_hor:
        raise ValueError("Pass a single prediction_length.")
    if not hasattr(dm, "border_end") and dataset not in MULTI_VARIATE_DATASETS:
        raise NotImplementedError("Only multivariate short-term and long-term datasets are supported.")

    meta = Meta(dataset=dm.dataset, freq=dm.freq, target_dim=int(dm.target_dim),
                context_length=int(dm.context_length), history_length=int(dm.history_length),
                prediction_length=int(dm.prediction_length))

    raw = _train_series(dm)
    observed = (~np.isnan(raw)).astype(np.float32)
    series = _scale(dm.scaler, np.nan_to_num(raw, nan=0.0))   # NaN -> 0 before scaling, as GluonTS imputes
    train = TrainWindows(series=series, observed=observed,
                         starts=valid_starts(len(series), meta.history_length, meta.prediction_length),
                         history_length=meta.history_length,
                         prediction_length=meta.prediction_length)

    val = _collect(dm.val_iter_dataset, dm.scaler) if dm.val_iter_dataset is not None else None
    test = _collect(dm.test_iter_dataset, dm.scaler)
    return ProbTSData(meta=meta, train=train, val=val, test=test, scaler=dm.scaler)