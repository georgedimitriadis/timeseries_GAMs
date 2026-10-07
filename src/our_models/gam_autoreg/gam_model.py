"""
Step 2 — the autoregressive GAM model (gam_model.py).

Pipeline (same three stages as the tabular ECMACMethod pipeline, without the
supplementary linear model):

    1. ChannelMinMaxScaler   per-channel min/max of the standard-scaled train
                             series, mapping each channel to [-1, 1]
                             (= sklearn MinMaxScaler((-1, 1)) fitted per channel)
    2. ec network            ECRegressor.get_features() -> Dense(1)
                             (the network ECBase._fit builds for regression)
    3. RidgeCV               replaces the Dense(1) head after the network is
                             trained; fitted on ec features collected along the
                             network's own free-running rollout

Everything runs in the scaled space z:
    z = 2 * (x - lo[c]) / (hi[c] - lo[c]) - 1        x: standard-scaled value of channel c
The network maps the values at the lags (in z) to the next value (in z), so a
prediction can be fed back as an input. Forecasts are mapped back with the
inverse before the loss / metrics.

One model is shared by all channels; a row of data is one channel of one
window (data.to_series layout: row n*C + c).

What this module does NOT do: the optimisation loop (training.py) and the
experiment script (step 4).

Keras runs on the JAX backend; KERAS_BACKEND is set before keras is imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import keras
import numpy as np
from keras import layers
from sklearn.linear_model import RidgeCV

from our_models.gam_autoreg.data import EvalWindows, TrainWindows, predefined_split, to_series
from our_models.gam_autoreg.ec.elco import ECRegressor


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class GAMConfig:
    lags: Optional[Sequence[int]] = (1, 3, 9, 27)   # inputs: values at t-1, t-3, ...;
                                                 # None = whole context (1..context_length),
                                                 # set with resolve_lags(context_length)
    arity: int = 2                               # 2 = pairwise interactions
    use_linear: bool = True
    use_cubic: bool = True
    use_raw_linear: bool = True
    mixing_layer_on: bool = False
    final_linear_layer_regularizer: Optional[str] = None
    ridge_alphas: Sequence[float] = (0.1, 1.0, 10.0)   # RidgeCV default grid
    ridge_max_rows: int = 1_000_000              # cap per split (train / val) for the ridge fit
    ridge_batch: int = 4096                      # series per rollout chunk when collecting features
    feedback_clip: Optional[float] = 3.0         # clip each prediction to [-c, c] (z space) before
                                                 # it is fed back; None = no clip
    seed: int = 0

    def __post_init__(self):
        if self.lags is None:                    # resolved later from the context length
            return
        lags = [int(l) for l in self.lags]
        if any(l < 1 for l in lags) or len(set(lags)) != len(lags):
            raise ValueError(f"lags must be unique integers >= 1, got {self.lags}")
        self.lags = tuple(sorted(lags))

    def resolve_lags(self, context_length: int):
        """lags=None -> (1, 2, ..., context_length). No effect if lags are given."""
        if self.lags is None:
            self.lags = tuple(range(1, int(context_length) + 1))
        return self


# ----------------------------------------------------------------------
# Stage 1: scaler
# ----------------------------------------------------------------------
class ChannelMinMaxScaler:
    """Per-channel min/max -> [-1, 1]. Fitted on the (standard-scaled) train series."""

    def __init__(self):
        self.lo = None      # [C]
        self.hi = None      # [C]

    def fit(self, series: np.ndarray, observed: Optional[np.ndarray] = None):
        """series [T, C]; missing values (observed == 0) are ignored."""
        x = series.astype(np.float64)
        if observed is not None:
            x = np.where(observed > 0, x, np.nan)
        self.lo = np.nanmin(x, axis=0).astype(np.float32)
        self.hi = np.nanmax(x, axis=0).astype(np.float32)
        same = self.hi - self.lo < 1e-8                  # constant channel
        self.hi = np.where(same, self.lo + 1.0, self.hi)
        return self

    def row_params(self, n_rows: int):
        """lo, hi for rows in data.to_series layout (channel = row % C)."""
        C = len(self.lo)
        ch = np.arange(n_rows) % C
        return self.lo[ch], self.hi[ch]

    @staticmethod
    def transform_rows(x, lo, hi):
        """x [n, L] -> z; lo, hi [n]. Works for numpy and jax arrays."""
        return 2.0 * (x - lo[:, None]) / (hi[:, None] - lo[:, None]) - 1.0

    @staticmethod
    def inverse_rows(z, lo, hi):
        return (z + 1.0) / 2.0 * (hi[:, None] - lo[:, None]) + lo[:, None]


# ----------------------------------------------------------------------
# Stage 2 + 3: network, rollout, ridge head
# ----------------------------------------------------------------------
class GAMAutoReg:
    """
    Holds the three stages and the pure-JAX rollout used for training and
    forecasting.

    Parameters of the network are kept outside keras as two lists of arrays,
    (tv, ntv) = (trainable, non-trainable) variables, so the training loop can
    differentiate through the rollout with jax (keras "stateless" API).
    """

    def __init__(self, cfg: GAMConfig, n_channels: int):
        if cfg.lags is None:
            raise ValueError("cfg.lags is None: call cfg.resolve_lags(context_length) first")
        self.cfg = cfg
        self.n_channels = int(n_channels)
        self.lags = np.asarray(cfg.lags, dtype=np.int32)
        self.max_lag = int(self.lags.max())
        self.lag_pos = jnp.asarray(self.max_lag - self.lags)   # column of each lag in the buffer

        self.scaler = ChannelMinMaxScaler()

        # ec network: inputs -> (optional mixing) -> EquationLayer -> Dropout -> Dense(1)
        ec = ECRegressor(arity=cfg.arity, mixing_layer_on=cfg.mixing_layer_on,
                         use_linear=cfg.use_linear, use_cubic=cfg.use_cubic,
                         use_raw_linear=cfg.use_raw_linear,
                         final_linear_layer_regularizer=cfg.final_linear_layer_regularizer)
        _, inputs, self.equation_layer, feats = ec.get_features(n_inputs=len(self.lags))
        out = layers.Dense(1, kernel_regularizer=ec.linear_layer_regularizer)(feats)
        self.net = keras.Model(inputs, [feats, out])            # returns features and prediction
        self.n_features = int(feats.shape[-1])

        self.tv = [v.value for v in self.net.trainable_variables]
        self.ntv = [v.value for v in self.net.non_trainable_variables]

        # ridge head (set by fit_ridge); None -> use the Dense(1) head
        self.ridge = None   # dict(coef [F], intercept, alpha)

    # ---------------- scaling ----------------
    def fit_scaler(self, train: TrainWindows):
        self.scaler.fit(train.series, train.observed)
        return self

    def to_z(self, x_rows: np.ndarray):
        lo, hi = self.scaler.row_params(x_rows.shape[0])
        return self.scaler.transform_rows(x_rows, lo, hi)

    # ---------------- pure-JAX forward ----------------
    def rollout(self, tv, ntv, buf, H: int, training: bool,
                ridge_coef=None, ridge_intercept=None, return_features: bool = False):
        """
        Free-running rollout in z space.

        buf   [n, max_lag]  last max_lag values (z) before the forecast start
        H     number of steps
        ridge_coef / ridge_intercept: if given, the ridge head is used instead of Dense(1)

        Returns z_pred [n, H], ntv (updated dropout RNG state), reg (EquationLayer
        penalty of the first step; it does not depend on the inputs), and
        features [n, H, F] if return_features.
        Each step is wrapped in jax.checkpoint: during backprop activations are
        recomputed instead of stored, so memory grows with H only through the
        small carry (buffer + RNG state).
        """
        use_ridge = ridge_coef is not None

        def step(carry, _):
            b, ntv_ = carry
            x = jnp.take(b, self.lag_pos, axis=1)                         # [n, n_lags]
            if training:
                (f, y), ntv_, losses = self.net.stateless_call(tv, ntv_, x, training=True,
                                                               return_losses=True)
                # EquationLayer adds a loss of shape (1,) (smoothing_weight has shape (1,));
                # sum to a scalar so the training loss is a scalar.
                reg = (jnp.sum(jnp.stack([jnp.sum(l) for l in losses])).astype(jnp.float32)
                       if losses else jnp.asarray(0.0, jnp.float32))
            else:
                (f, y), ntv_ = self.net.stateless_call(tv, ntv_, x, training=False)
                reg = jnp.asarray(0.0, jnp.float32)
            y = (f @ ridge_coef + ridge_intercept) if use_ridge else y[:, 0]
            y = y.astype(jnp.float32)
            if self.cfg.feedback_clip is not None:
                y = jnp.clip(y, -self.cfg.feedback_clip, self.cfg.feedback_clip)
            b = jnp.concatenate([b[:, 1:], y[:, None]], axis=1)          # feed prediction back
            out = (y, reg, f) if return_features else (y, reg)
            return (b, ntv_), out

        (_, ntv_out), outs = jax.lax.scan(jax.checkpoint(step), (buf, ntv), None, length=H)
        z = outs[0].T                                                     # [n, H]
        reg = outs[1][0]
        if return_features:
            return z, ntv_out, reg, jnp.swapaxes(outs[2], 0, 1)           # [n, H, F]
        return z, ntv_out, reg

    def initial_buffer(self, past_z):
        """past_z [n, L] -> last max_lag values [n, max_lag]."""
        return past_z[:, -self.max_lag:]

    # ---------------- forecasting (numpy in / out) ----------------
    @partial(jax.jit, static_argnums=(0, 4, 5))
    def _forecast_z(self, tv, ntv, buf, H, use_ridge, coef, intercept):
        if use_ridge:
            z, _, _ = self.rollout(tv, ntv, buf, H, training=False,
                                   ridge_coef=coef, ridge_intercept=intercept)
        else:
            z, _, _ = self.rollout(tv, ntv, buf, H, training=False)
        return z

    def forecast(self, past_s: np.ndarray, H: int, use_ridge: Optional[bool] = None,
                 batch: int = 65536) -> np.ndarray:
        """
        past_s [N, L, C] standard-scaled windows -> forecasts [N, H, C] (standard-scaled).
        use_ridge: None -> ridge head if fitted, else Dense(1).
        """
        use_ridge = (self.ridge is not None) if use_ridge is None else use_ridge
        coef, icpt = self._ridge_arrays() if use_ridge else (None, None)
        N, _, C = past_s.shape
        rows = to_series(past_s)                                          # [N*C, L]
        lo, hi = self.scaler.row_params(rows.shape[0])
        out = np.empty((rows.shape[0], H), dtype=np.float32)
        for i in range(0, rows.shape[0], batch):
            sl = slice(i, i + batch)
            z_past = self.scaler.transform_rows(rows[sl], lo[sl], hi[sl])
            z = self._forecast_z(self.tv, self.ntv, jnp.asarray(self.initial_buffer(z_past)),
                                 int(H), bool(use_ridge), coef, icpt)
            out[sl] = self.scaler.inverse_rows(np.asarray(z), lo[sl], hi[sl])
        return out.reshape(N, C, H).transpose(0, 2, 1)

    # ---------------- stage 3: ridge on rollout features ----------------
    @partial(jax.jit, static_argnums=(0, 4))
    def _rollout_features(self, tv, ntv, buf, H):
        _, _, _, f = self.rollout(tv, ntv, buf, H, training=False, return_features=True)
        return f

    def _collect(self, past, future, obs, lo, hi):
        """Rows of one split -> (X [rows, F], y [rows]) along the Dense(1)-head rollout,
        targets in z space; rows with an unobserved target are dropped."""
        H = future.shape[1]
        Xs, ys = [], []
        for i in range(0, past.shape[0], self.cfg.ridge_batch):
            sl = slice(i, i + self.cfg.ridge_batch)
            z_past = self.scaler.transform_rows(past[sl], lo[sl], hi[sl])
            f = np.asarray(self._rollout_features(self.tv, self.ntv,
                                                  jnp.asarray(self.initial_buffer(z_past)), H))
            z_fut = self.scaler.transform_rows(future[sl], lo[sl], hi[sl])
            keep = obs[sl].reshape(-1) > 0
            Xs.append(f.reshape(-1, f.shape[-1])[keep])
            ys.append(z_fut.reshape(-1)[keep])
        return np.concatenate(Xs), np.concatenate(ys)

    def _pick_pairs(self, n_windows: int, H: int, rng: np.random.Generator):
        """(window, channel) pairs: all of them if within ridge_max_rows, else a
        seeded random subset of ridge_max_rows // H pairs."""
        C = self.n_channels
        n_all = n_windows * C
        n_keep = min(n_all, max(1, self.cfg.ridge_max_rows // H))
        idx = np.arange(n_all) if n_keep == n_all else np.sort(rng.choice(n_all, n_keep, replace=False))
        return idx // C, idx % C

    def fit_ridge(self, train: TrainWindows, val: EvalWindows):
        """
        Fit RidgeCV on features from the network's own free-running rollout.
          rows   : (window, channel, step); targets = true future values (z)
          split  : train rows -> -1, val rows -> fold 0 (PredefinedSplit),
                   alpha chosen on val, then refit on train + val (RidgeCV behaviour)
        """
        rng = np.random.default_rng(self.cfg.seed)
        H = train.prediction_length

        w, c = self._pick_pairs(len(train), H, rng)
        p, f, o = train.channel_rows(train.starts[w], c)
        lo, hi = self.scaler.lo[c], self.scaler.hi[c]
        X_tr, y_tr = self._collect(p, f, o, lo, hi)

        w, c = self._pick_pairs(len(val), H, rng)
        p, f, o = val.past_s[w, :, c], val.future_s[w, :, c], val.future_observed[w, :, c]
        lo, hi = self.scaler.lo[c], self.scaler.hi[c]
        X_va, y_va = self._collect(p, f, o, lo, hi)

        ridge = RidgeCV(alphas=tuple(self.cfg.ridge_alphas),
                        cv=predefined_split(len(y_tr), len(y_va)))
        ridge.fit(np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va]))
        self.ridge = {"coef": np.asarray(ridge.coef_, dtype=np.float32),
                      "intercept": np.float32(ridge.intercept_),
                      "alpha": float(ridge.alpha_),
                      "n_train_rows": int(len(y_tr)), "n_val_rows": int(len(y_va))}
        return self.ridge

    def _ridge_arrays(self):
        return jnp.asarray(self.ridge["coef"]), jnp.asarray(self.ridge["intercept"])

    # ---------------- state ----------------
    def set_params(self, tv, ntv):
        self.tv, self.ntv = list(tv), list(ntv)

    def sync_to_keras(self):
        """Copy (tv, ntv) into the keras variables (for saving / get_active_features)."""
        for v, x in zip(self.net.trainable_variables, self.tv):
            v.assign(x)
        for v, x in zip(self.net.non_trainable_variables, self.ntv):
            v.assign(x)

    def active_features(self):
        self.sync_to_keras()
        return self.equation_layer.get_active_features()

    def state_dict(self):
        self.sync_to_keras()
        return {
            "config": asdict(self.cfg),
            "n_channels": self.n_channels,
            "scaler_lo": self.scaler.lo, "scaler_hi": self.scaler.hi,
            "weights": {v.path: np.asarray(v.value) for v in self.net.weights},
            "ridge": self.ridge,
        }