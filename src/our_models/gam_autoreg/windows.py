"""
Generic window containers and helpers shared by the ProbTS (data.py) and the
dysts (dysts_data.py) loaders. NumPy / sklearn only.

    TrainWindows   one continuous train series [T, C] + valid forecast starts
    EvalWindows    fixed evaluation windows (past / future, raw and scaled)
    to_series / from_series, series_weights, one_step_rows, predefined_split
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import PredefinedSplit


# ----------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------
@dataclass
class TrainWindows:
    """Standard-scaled train series and the valid window end positions."""
    series: np.ndarray            # [T, C] float32, standard-scaled
    observed: np.ndarray          # [T, C] float32, 1 = observed, 0 = missing (NaN in raw data)
    starts: np.ndarray            # [n_windows] int: forecast start t of each valid window
    history_length: int
    prediction_length: int

    def __len__(self):
        return len(self.starts)

    def windows(self, t: np.ndarray):
        """Cut windows ending at forecast starts `t`.
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



# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def valid_starts(T: int, history_length: int, prediction_length: int) -> np.ndarray:
    """Forecast starts t with a full history before and a full horizon after."""
    return np.arange(history_length, T - prediction_length + 1)



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

