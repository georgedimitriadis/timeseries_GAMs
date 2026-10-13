"""
Parts of the autoregressive GAM shared by the ProbTS model
(for_probts/probts_gam_model.py) and the dysts model (for_dysts/dysts_gam_model.py).

    GAMConfig             lags, ec options, ridge settings, feedback_clip, seed
    ChannelMinMaxScaler   per-channel min/max -> [-1, 1] (the EquationLayer domain)
    build_gam_network     ECRegressor.get_features() -> Dense(1), returning features and output
    call_network          one stateless call of a network (training / inference) with its penalty
    fit_ridge_cv          RidgeCV on (train rows, val rows) with a PredefinedSplit
    BaseGAM               holds one or more networks, their weights as one flat list
                          (tv, ntv) and the per-network slices; set_params / sync_to_keras

Keras runs on the JAX backend; KERAS_BACKEND is set before keras is imported.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

from dataclasses import dataclass
from typing import Optional, Sequence

import jax.numpy as jnp
import keras
import numpy as np
from keras import layers
from sklearn.linear_model import RidgeCV

from our_models.gam_autoreg.ec.elco import ECRegressor
from our_models.gam_autoreg.windows import TrainWindows, predefined_split


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
    ridge_alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)   # RidgeCV default grid
    ridge_max_rows: int = 1_000_000              # cap per split (train / val) for the ridge fit
    ridge_batch: int = 4096                      # series per rollout chunk when collecting features
    feedback_clip: Optional[float] = 2.0         # clip each prediction to [-c, c] (z space) before
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
    """Per-channel min/max -> [-1, 1]. Fitted on the train series [T, C]."""

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

    # rows layout (windows.to_series: row n*C + c, used by the ProbTS model)
    def row_params(self, n_rows: int):
        """lo, hi for rows in windows.to_series layout (channel = row % C)."""
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
# Stage 2: network
# ----------------------------------------------------------------------
def build_gam_network(cfg: GAMConfig, n_inputs: int, name: Optional[str] = None):
    """
    inputs [n, n_inputs] -> (optional mixing) -> EquationLayer -> Dropout(0.1) -> features
    -> Dense(1); the network ECBase._fit builds for regression.
    Returns (keras.Model returning [features, prediction], equation_layer, n_features).
    """
    ec = ECRegressor(arity=cfg.arity, mixing_layer_on=cfg.mixing_layer_on,
                     use_linear=cfg.use_linear, use_cubic=cfg.use_cubic,
                     use_raw_linear=cfg.use_raw_linear,
                     final_linear_layer_regularizer=cfg.final_linear_layer_regularizer)
    _, inputs, equation_layer, feats = ec.get_features(n_inputs=n_inputs)
    out = layers.Dense(1, kernel_regularizer=ec.linear_layer_regularizer)(feats)
    net = keras.Model(inputs, [feats, out], name=name)
    return net, equation_layer, int(feats.shape[-1])


def call_network(net, tv, ntv, x, training: bool):
    """
    One stateless call. Returns features [n, F], prediction [n], updated ntv (dropout RNG
    state) and the EquationLayer penalty as a scalar (0 at inference).
    The penalty is added by the layer with shape (1,) (smoothing_weight has shape (1,));
    it is summed to a scalar so the training loss is a scalar.
    """
    if training:
        (f, y), ntv, losses = net.stateless_call(tv, ntv, x, training=True, return_losses=True)
        reg = (jnp.sum(jnp.stack([jnp.sum(l) for l in losses])).astype(jnp.float32)
               if losses else jnp.asarray(0.0, jnp.float32))
    else:
        (f, y), ntv = net.stateless_call(tv, ntv, x, training=False)
        reg = jnp.asarray(0.0, jnp.float32)
    return f, y[:, 0].astype(jnp.float32), ntv, reg


# ----------------------------------------------------------------------
# Stage 3: ridge
# ----------------------------------------------------------------------
def fit_ridge_cv(X_tr, y_tr, X_va, y_va, alphas) -> dict:
    """
    RidgeCV on [train rows, val rows] with PredefinedSplit (train -> -1, val -> fold 0):
    alpha chosen on the val rows, then refit on train + val (RidgeCV behaviour).
    """
    ridge = RidgeCV(alphas=tuple(alphas), cv=predefined_split(len(y_tr), len(y_va)))
    ridge.fit(np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    return {"coef": np.asarray(ridge.coef_, dtype=np.float32),
            "intercept": np.float32(ridge.intercept_), "alpha": float(ridge.alpha_),
            "n_train_rows": int(len(y_tr)), "n_val_rows": int(len(y_va))}


# ----------------------------------------------------------------------
# Base class: networks + flat parameters
# ----------------------------------------------------------------------
class BaseGAM:
    """
    Holds one or more networks of the same structure. Their weights are kept outside
    keras as two flat lists of arrays, (tv, ntv) = (trainable, non-trainable), so the
    training loop can differentiate through the rollout with jax (keras "stateless" API)
    and one optimiser updates all networks. _tv_slices[i] / _ntv_slices[i] give
    network i's part.
    """

    def _init_networks(self, cfg: GAMConfig, n_inputs: int, names):
        self.scaler = ChannelMinMaxScaler()
        self.nets, self.equation_layers = [], []
        for name in names:
            net, eq, n_features = build_gam_network(cfg, n_inputs, name=name)
            self.nets.append(net)
            self.equation_layers.append(eq)
        self.n_features = n_features
        self.tv, self.ntv, self._tv_slices, self._ntv_slices = [], [], [], []
        for net in self.nets:
            a = len(self.tv)
            self.tv += [v.value for v in net.trainable_variables]
            self._tv_slices.append(slice(a, len(self.tv)))
            a = len(self.ntv)
            self.ntv += [v.value for v in net.non_trainable_variables]
            self._ntv_slices.append(slice(a, len(self.ntv)))

    @property
    def trainable_variables(self):
        """keras variables in the same order as self.tv (for optimizer.build)."""
        return [v for net in self.nets for v in net.trainable_variables]

    def fit_scaler(self, train: TrainWindows):
        self.scaler.fit(train.series, train.observed)
        return self

    def set_params(self, tv, ntv):
        self.tv, self.ntv = list(tv), list(ntv)

    def sync_to_keras(self):
        """Copy (tv, ntv) into the keras variables (for saving / get_active_features)."""
        for i, net in enumerate(self.nets):
            for v, x in zip(net.trainable_variables, self.tv[self._tv_slices[i]]):
                v.assign(x)
            for v, x in zip(net.non_trainable_variables, self.ntv[self._ntv_slices[i]]):
                v.assign(x)

    def network_weights(self):
        """Per network: {variable path: array} (after sync_to_keras)."""
        return [{v.path: np.asarray(v.value) for v in net.weights} for net in self.nets]
