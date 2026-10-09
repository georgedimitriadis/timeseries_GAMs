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

from probts.data.data_manager import DataManager, MULTI_VARIATE_DATASETS
from our_models.gam_autoreg.windows import (  # noqa: F401  (re-exported)
    EvalWindows, TrainWindows, from_series, one_step_rows, predefined_split,
    series_weights, to_series, valid_starts)


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
class ProbTSData:
    meta: Meta
    train: TrainWindows
    val: Optional[EvalWindows]
    test: EvalWindows
    scaler: object = field(repr=False)


# ----------------------------------------------------------------------
# Pure helpers (NumPy only)
# ----------------------------------------------------------------------
def stsf_train_length(T: int, num_test_dates: int, prediction_length: int, split_val: bool) -> int:
    """Length of the short-term train series after data_utils.split_train_val."""
    return T - num_test_dates * prediction_length if split_val else T


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
