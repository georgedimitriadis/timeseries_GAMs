"""
Step 2 (dysts) — autoregressive GAM for the dysts systems: cross-channel inputs,
one separate GAM per coordinate.

For a system with D coordinates and lags (l_1, ..., l_K):

    inputs (same for every coordinate's GAM), channel-major order:
        x0[t-l_1], ..., x0[t-l_K], x1[t-l_1], ..., x1[t-l_K], ..., x{D-1}[t-l_K]
        -> D * K values (input_names() gives the names)
    GAM d  (d = 0..D-1): ECRegressor.get_features() -> Dense(1) -> output g_d
        own EquationLayer and head per coordinate, so each coordinate gets its
        own equation; with arity 2 the pairwise products include cross-coordinate
        terms (e.g. x0[t-1] * x1[t-1])
    output (cfg.predict_increment):
        False: g_d = x_d[t] in z space
        True : g_d = (x_d[t] - x_d[t-1]) / dt in the original units (an estimate of
               dx_d/dt, the right-hand side of the ODE); x_d[t] = x_d[t-1] + dt * g_d
    all D predictions are appended to the history together and fed back
    (free-running rollout, no teacher forcing)

Inputs to the splines are always in z space ([-1, 1], the EquationLayer domain);
the loss is always MSE on z-scaled values (each coordinate weighted equally).
Metrics are computed on forecasts mapped back to the original scale.

Pipeline per GAM (as the tabular ECMACMethod, without the supplementary linear model):
    1. ChannelMinMaxScaler   per-coordinate min/max of the train series -> [-1, 1] (z space)
    2. ec network            trained by the dysts training loop
    3. RidgeCV               one per coordinate, replaces Dense(1) after training; fitted on
                             the GAM's features along the network's own free-running
                             rollout; alpha chosen on the val window (PredefinedSplit);
                             targets: x_d[t] (z), or with predict_increment
                             (x_d[t] - previous rollout value) / dt

Weights of all D networks are kept as one flat list of arrays (tv, ntv) so one
optimiser updates them together; self._tv_slices / self._ntv_slices give each
network's part (BaseGAM in basis_gam_model.py, shared with the ProbTS model).

prepare_batch() and loss() live here (not in the training loop) because they
depend on this layout.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from our_models.gam_autoreg.basis_gam_model import BaseGAM, GAMConfig, call_network, fit_ridge_cv
from our_models.gam_autoreg.windows import EvalWindows, TrainWindows


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class DystsGAMConfig(GAMConfig):
    """GAMConfig (lags, arity, ec options, ridge, feedback_clip, seed) plus:"""
    predict_increment: bool = False   # True: each GAM outputs (x[t] - x[t-1]) / dt in the
                                      # original units instead of x[t] in z space


# ----------------------------------------------------------------------
# Pure helpers (work on numpy and jax arrays)
# ----------------------------------------------------------------------
def gather_inputs(buf, lag_pos):
    """buf [N, max_lag, D] -> GAM inputs [N, D*K] in channel-major order
    (all lags of x0, then all lags of x1, ...)."""
    x = buf[:, lag_pos, :]                        # [N, K, D]
    N, K, D = x.shape
    return x.transpose(0, 2, 1).reshape(N, D * K)


def to_z(x, lo, hi):
    """x [..., D] -> [-1, 1] per coordinate."""
    return 2.0 * (x - lo) / (hi - lo) - 1.0


def from_z(z, lo, hi):
    return (z + 1.0) / 2.0 * (hi - lo) + lo


def input_names(dim: int, lags) -> list:
    """Names of the GAM inputs in gather_inputs order, e.g. 'x1[t-3]'."""
    return [f"x{d}[t-{l}]" for d in range(dim) for l in lags]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class DystsGAM(BaseGAM):
    def __init__(self, cfg: DystsGAMConfig, dim: int, dt: float):
        if cfg.lags is None:
            raise ValueError("cfg.lags is None: call cfg.resolve_lags(history_length) first")
        self.cfg = cfg
        self.dim = int(dim)
        self.dt = float(dt)                                           # system time step
        self.lags = np.asarray(cfg.lags, dtype=np.int32)
        self.max_lag = int(self.lags.max())
        self.lag_pos = np.asarray(self.max_lag - self.lags)          # column of lag k in the buffer
        self.n_inputs = self.dim * len(self.lags)

        # D separate networks (same structure, separate weights), scaler, flat parameters
        self._init_networks(cfg, n_inputs=self.n_inputs,
                            names=[f"gam_x{d}" for d in range(self.dim)])

        self.ridge = None   # list of D dicts (coef [F], intercept, alpha) after fit_ridge

    def input_names(self):
        return input_names(self.dim, self.cfg.lags)

    # ---------------- pure-JAX forward ----------------
    def rollout(self, tv, ntv, buf, H: int, training: bool,
                ridge_coef=None, ridge_intercept=None, return_features: bool = False):
        """
        Free-running rollout in z space.

        buf  [N, max_lag, D]  last max_lag values (z) before the forecast start
        ridge_coef [D, F], ridge_intercept [D]: use the ridge heads instead of Dense(1)

        Returns z_pred [N, H, D] (the predicted values in z, also with predict_increment),
        ntv, reg (sum of the D EquationLayer penalties, first step), and
        features [N, H, D, F] if return_features.
        """
        use_ridge = ridge_coef is not None
        lag_pos = jnp.asarray(self.lag_pos)
        lo, hi = jnp.asarray(self.scaler.lo), jnp.asarray(self.scaler.hi)

        def step(carry, _):
            b, ntv_ = carry
            x = gather_inputs(b, lag_pos)                              # [N, D*K]
            ys, fs, reg, new_ntv = [], [], jnp.asarray(0.0, jnp.float32), list(ntv_)
            for d, net in enumerate(self.nets):
                f, y, ntvd, reg_d = call_network(net, tv[self._tv_slices[d]],
                                                 ntv_[self._ntv_slices[d]], x, training)
                new_ntv[self._ntv_slices[d]] = ntvd
                reg = reg + reg_d
                if use_ridge:
                    y = (f @ ridge_coef[d] + ridge_intercept[d]).astype(jnp.float32)
                ys.append(y)
                fs.append(f)
            y = jnp.stack(ys, axis=1)                                  # [N, D] network outputs
            if self.cfg.predict_increment:                             # g = dx/dt (original units)
                x_prev = from_z(b[:, -1, :], lo, hi)
                y = to_z(x_prev + self.dt * y, lo, hi)                 # next value, back to z
            if self.cfg.feedback_clip is not None:
                y = jnp.clip(y, -self.cfg.feedback_clip, self.cfg.feedback_clip)
            b = jnp.concatenate([b[:, 1:, :], y[:, None, :]], axis=1)  # feed predictions back
            out = (y, reg, jnp.stack(fs, axis=1)) if return_features else (y, reg)
            return (b, new_ntv), out

        (_, ntv_out), outs = jax.lax.scan(jax.checkpoint(step), (buf, list(ntv)), None, length=H)
        z = jnp.swapaxes(outs[0], 0, 1)                                # [N, H, D]
        reg = outs[1][0]
        if return_features:
            return z, ntv_out, reg, jnp.swapaxes(outs[2], 0, 1)        # [N, H, D, F]
        return z, ntv_out, reg

    # ---------------- training helpers (used by the dysts training loop) ----------------
    def prepare_batch(self, past, future, observed):
        """Windows [B, L, D], [B, H, D], [B, H, D] -> jnp (buf_z, future_z, w).
        w [B, H]: minimum over coordinates of the observed mask."""
        lo, hi = self.scaler.lo, self.scaler.hi
        buf = to_z(past[:, -self.max_lag:, :], lo, hi)
        return (jnp.asarray(buf, jnp.float32), jnp.asarray(to_z(future, lo, hi), jnp.float32),
                jnp.asarray(observed.min(axis=-1), jnp.float32))

    def loss(self, tv, ntv, buf, future_z, w):
        """Free-running rollout over H = future_z.shape[1] steps; weighted MSE on z values
        (mean over coordinates) + EquationLayer penalties.
        Returns loss, (ntv, mse, frac_clipped)."""
        H = future_z.shape[1]
        z, ntv, reg = self.rollout(tv, ntv, buf, H, training=True)
        se = ((z - future_z) ** 2).mean(axis=-1)                       # [B, H]
        se_w = jnp.where(w != 0, se * w, 0.0)
        mse = (se_w.sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)).mean()
        clip = self.cfg.feedback_clip
        frac_clipped = jnp.mean(jnp.abs(z) >= clip) if clip is not None else jnp.asarray(0.0)
        return mse + reg, (ntv, mse, frac_clipped)

    # ---------------- forecasting ----------------
    @partial(jax.jit, static_argnums=(0, 4, 5))
    def _forecast_z(self, tv, ntv, buf, H, use_ridge, coef, intercept):
        if use_ridge:
            z, _, _ = self.rollout(tv, ntv, buf, H, training=False,
                                   ridge_coef=coef, ridge_intercept=intercept)
        else:
            z, _, _ = self.rollout(tv, ntv, buf, H, training=False)
        return z

    def forecast(self, past: np.ndarray, H: int, use_ridge: Optional[bool] = None) -> np.ndarray:
        """past [N, L, D] original scale -> forecasts [N, H, D] original scale.
        use_ridge: None -> ridge heads if fitted, else Dense(1)."""
        use_ridge = (self.ridge is not None) if use_ridge is None else use_ridge
        coef, icpt = self._ridge_arrays() if use_ridge else (None, None)
        lo, hi = self.scaler.lo, self.scaler.hi
        buf = jnp.asarray(to_z(past[:, -self.max_lag:, :], lo, hi), jnp.float32)
        z = self._forecast_z(self.tv, self.ntv, buf, int(H), bool(use_ridge), coef, icpt)
        return from_z(np.asarray(z), lo, hi)

    # ---------------- stage 3: ridge per coordinate on rollout features ----------------
    @partial(jax.jit, static_argnums=(0, 4))
    def _rollout_features(self, tv, ntv, buf, H):
        z, _, _, f = self.rollout(tv, ntv, buf, H, training=False, return_features=True)
        return z, f

    def _collect(self, past, future):
        """
        past [N, L, D], future [N, H, D] -> features [N*H, D, F], targets [N*H, D]
        along the Dense(1) rollout. Targets:
          predict_increment=False: true x_d[t] in z
          predict_increment=True : (true x_d[t] - x_d[t-1] of the rollout) / dt, original
                                   units; x_d[t-1] is the previous rollout value (the last
                                   context value at the first step), so the ridge head
                                   corrects the step the rollout actually takes
        """
        lo, hi = self.scaler.lo, self.scaler.hi
        X, Y = [], []
        for i in range(0, past.shape[0], self.cfg.ridge_batch):
            p, fut = past[i:i + self.cfg.ridge_batch], future[i:i + self.cfg.ridge_batch]
            buf_np = to_z(p[:, -self.max_lag:, :], lo, hi)
            z, f = self._rollout_features(self.tv, self.ntv, jnp.asarray(buf_np, jnp.float32),
                                          fut.shape[1])
            z, f = np.asarray(z), np.asarray(f)
            if self.cfg.predict_increment:
                prev_z = np.concatenate([buf_np[:, -1:, :], z[:, :-1, :]], axis=1)
                target = (fut - from_z(prev_z, lo, hi)) / self.dt
            else:
                target = to_z(fut, lo, hi)
            X.append(f.reshape(-1, self.dim, f.shape[-1]))
            Y.append(target.reshape(-1, self.dim))
        return np.concatenate(X), np.concatenate(Y)

    def fit_ridge(self, train: TrainWindows, val: EvalWindows):
        """
        One RidgeCV per coordinate on features from the network's own rollout.
          rows   : (window, step); targets: see _collect (x[t] in z, or the increment)
          train  : train windows (seeded subset if more than ridge_max_rows rows)
          split  : train rows -> -1, val rows -> fold 0 (PredefinedSplit); alpha chosen on
                   val, then refit on train + val (RidgeCV behaviour)
        """
        rng = np.random.default_rng(self.cfg.seed)
        H = train.prediction_length
        n_keep = min(len(train), max(1, self.cfg.ridge_max_rows // H))
        starts = train.starts if n_keep == len(train) else \
            np.sort(rng.choice(train.starts, n_keep, replace=False))
        p, f, _ = train.windows(starts)
        X_tr, y_tr = self._collect(p, f)
        X_va, y_va = self._collect(val.past, val.future)

        self.ridge = [fit_ridge_cv(X_tr[:, d], y_tr[:, d], X_va[:, d], y_va[:, d],
                                   self.cfg.ridge_alphas) for d in range(self.dim)]
        return self.ridge

    def _ridge_arrays(self):
        return (jnp.asarray(np.stack([r["coef"] for r in self.ridge])),
                jnp.asarray(np.array([r["intercept"] for r in self.ridge], np.float32)))

    # ---------------- state ----------------
    def active_features(self):
        """Per coordinate: EquationLayer.get_active_features() plus input names."""
        self.sync_to_keras()
        names = self.input_names()
        out = {}
        for d, eq in enumerate(self.equation_layers):
            a = eq.get_active_features()
            a["input_names"] = names
            out[f"x{d}"] = a
        return out

    def state_dict(self):
        self.sync_to_keras()
        return {"config": asdict(self.cfg), "dim": self.dim, "dt": self.dt,
                "input_names": self.input_names(),
                "scaler_lo": self.scaler.lo, "scaler_hi": self.scaler.hi,
                "weights": self.network_weights(),
                "ridge": self.ridge}
