"""
Step 2 (ProbTS) — the autoregressive GAM model for the ProbTS data.

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

One network is shared by all channels (channel-independent, as DLinear with
individual=false); a row of data is one channel of one window
(windows.to_series layout: row n*C + c).

Shared parts (config, scaler, network builder, ridge fit, parameter handling):
basis_gam_model.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from dataclasses import asdict
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from our_models.gam_autoreg.basis_gam_model import (  # noqa: F401  (GAMConfig, ChannelMinMaxScaler re-exported)
    BaseGAM, ChannelMinMaxScaler, GAMConfig, call_network, fit_ridge_cv)
from our_models.gam_autoreg.windows import EvalWindows, TrainWindows, to_series


class GAMAutoReg(BaseGAM):
    """
    Holds the three stages and the pure-JAX rollout used for training and
    forecasting. One network (self.net); weights in (self.tv, self.ntv).
    """

    def __init__(self, cfg: GAMConfig, n_channels: int):
        if cfg.lags is None:
            raise ValueError("cfg.lags is None: call cfg.resolve_lags(context_length) first")
        self.cfg = cfg
        self.n_channels = int(n_channels)
        self.lags = np.asarray(cfg.lags, dtype=np.int32)
        self.max_lag = int(self.lags.max())
        self.lag_pos = jnp.asarray(self.max_lag - self.lags)   # column of each lag in the buffer

        self._init_networks(cfg, n_inputs=len(self.lags), names=["gam"])
        self.net = self.nets[0]
        self.equation_layer = self.equation_layers[0]

        # ridge head (set by fit_ridge); None -> use the Dense(1) head
        self.ridge = None   # dict(coef [F], intercept, alpha)

    # ---------------- scaling ----------------
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
            f, y, ntv_, reg = call_network(self.net, tv, ntv_, x, training)
            if use_ridge:
                y = (f @ ridge_coef + ridge_intercept).astype(jnp.float32)
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

        self.ridge = fit_ridge_cv(X_tr, y_tr, X_va, y_va, self.cfg.ridge_alphas)
        return self.ridge

    def _ridge_arrays(self):
        return jnp.asarray(self.ridge["coef"]), jnp.asarray(self.ridge["intercept"])

    # ---------------- state ----------------
    def active_features(self):
        self.sync_to_keras()
        return self.equation_layer.get_active_features()

    def state_dict(self):
        self.sync_to_keras()
        return {
            "config": asdict(self.cfg),
            "n_channels": self.n_channels,
            "scaler_lo": self.scaler.lo, "scaler_hi": self.scaler.hi,
            "weights": self.network_weights()[0],
            "ridge": self.ridge,
        }
