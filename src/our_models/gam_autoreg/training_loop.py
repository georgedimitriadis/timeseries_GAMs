"""
Step 3 — training loop for the autoregressive GAM (no teacher forcing).

    result = train_gam(model, data, TrainConfig(...))

What it does
------------
1. Fits the per-channel MinMax scaler on the train split (model.fit_scaler).
2. Trains the ec network + Dense(1) head on free-running rollouts:
     - batch      : TrainWindows.sample_batch -> rows (one per window x channel)
     - forward    : model.rollout(training=True) over the full prediction length,
                    each prediction fed back as the newest input
     - loss       : MSE in the standard-scaled space, weighted by the observed
                    mask (as ProbTS Forecaster.get_weighted_loss), plus the
                    EquationLayer penalty
     - optimiser  : keras AdamW (stateless, on the JAX backend) + EMA of the
                    weights; EMA weights are used for every evaluation, as
                    AdamW(use_ema=True) + SwapEMAWeights in ECBase._fit
     - accumulate_grad_batches as in the ProbTS configs
3. After every epoch: val metrics with the ProbTS Evaluator (denormalised);
   the epoch with the lowest val_CRPS is kept (run.py's checkpoint rule).
4. Puts the best weights in the model and fits the RidgeCV head on rollout
   features (model.fit_ridge); reports val metrics with both heads.

evaluate_probts() is also used by the experiment script for the test set.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import jax
import jax.numpy as jnp
import keras
import numpy as np
import torch

from probts.utils.evaluator import Evaluator
from our_models.gam_autoreg.data import EvalWindows, ProbTSData, series_weights, to_series
from our_models.gam_autoreg.gam_model import ChannelMinMaxScaler, GAMAutoReg


# ----------------------------------------------------------------------
# Config / result
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 50                  # ProbTS trainer.max_epochs
    batches_per_epoch: int = 100          # ProbTS trainer.limit_train_batches
    batch_size: int = 32                  # ProbTS data.batch_size (windows; rows = batch_size * C)
    accumulate_grad_batches: int = 1      # ProbTS trainer.accumulate_grad_batches
    eval_batch_size: int = 32             # ProbTS data.test_batch_size (metric averaging per batch)
    learning_rate: float = 0.01           # ECBase default
    weight_decay: float = 0.004           # keras AdamW default
    ema_momentum: float = 0.99            # keras optimizer EMA default
    quantiles_num: int = 20               # ProbTS configs
    fit_ridge: bool = True
    seed: int = 0
    verbose: bool = True


@dataclass
class TrainResult:
    best_epoch: int
    best_score: float
    selection_metric: str
    history: list = field(default_factory=list)
    val_dense: Optional[dict] = None      # val metrics, Dense(1) head
    val_ridge: Optional[dict] = None      # val metrics, ridge head
    ridge: Optional[dict] = None
    train_time_s: float = 0.0


# ----------------------------------------------------------------------
# Evaluation (ProbTS metrics)
# ----------------------------------------------------------------------
def evaluate_probts(model: GAMAutoReg, windows: EvalWindows, probts_scaler, freq: str,
                    batch_size: int, stage: str, quantiles_num: int = 20,
                    use_ridge: Optional[bool] = None) -> dict:
    """
    Same computation as ProbTSForecastModule.evaluate + calculate_weighted_average:
    per batch of `batch_size` windows, ProbTS Evaluator on denormalised forecasts
    [B, 1, H, C] against the raw future; batch-size weighted mean over batches.
    Keys: f"{stage}_ND", f"{stage}_CRPS", ...
    """
    evaluator = Evaluator(quantiles_num=quantiles_num)
    H = windows.future.shape[1]
    pred_s = model.forecast(windows.past_s, H, use_ridge=use_ridge)            # [N, H, C] standard-scaled
    values, sizes = {}, []
    for sl in windows.batches(batch_size):
        denorm = probts_scaler.inverse_transform(torch.from_numpy(pred_s[sl]).float())
        metrics = evaluator(torch.from_numpy(windows.future[sl]), denorm.unsqueeze(1),
                            past_data=torch.from_numpy(windows.past[sl]), freq=freq)
        sizes.append(windows.future[sl].shape[0])
        for k, v in metrics.items():
            values.setdefault(f"{stage}_{k}", []).append(v)
    w = np.asarray(sizes, dtype=np.float64)
    return {k: float(np.sum(np.asarray(v, dtype=np.float64) * w) / w.sum()) for k, v in values.items()}


# ----------------------------------------------------------------------
# Jitted training step
# ----------------------------------------------------------------------
def make_train_fns(model: GAMAutoReg, optimizer, H: int, ema_momentum: float):
    """
    compute_grads(tv, ntv, buf, future_s, w, lo, hi) -> grads, ntv, loss, mse
        buf      [n, max_lag]  last values before the forecast start (z space)
        future_s [n, H]        true future (standard-scaled)
        w        [n, H]        observed weights
        lo, hi   [n]           per-row MinMax parameters
    apply_grads(tv, opt_vars, ema, grads) -> tv, opt_vars, ema
    """

    def loss_fn(tv, ntv, buf, future_s, w, lo, hi):
        z_pred, ntv, reg = model.rollout(tv, ntv, buf, H, training=True)
        pred_s = ChannelMinMaxScaler.inverse_rows(z_pred, lo, hi)        # back to standard-scaled
        se = (pred_s - future_s) ** 2
        se_w = jnp.where(w != 0, se * w, 0.0)                            # probts weighted_average
        per_row = se_w.sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)
        mse = per_row.mean()
        clip = model.cfg.feedback_clip
        frac_clipped = (jnp.mean(jnp.abs(z_pred) >= clip) if clip is not None
                        else jnp.asarray(0.0))  # share of predictions at the bound
        return mse + reg, (ntv, mse, frac_clipped)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    @jax.jit
    def compute_grads(tv, ntv, buf, future_s, w, lo, hi):
        (loss, (ntv, mse, frac_clipped)), grads = grad_fn(tv, ntv, buf, future_s, w, lo, hi)
        return grads, ntv, loss, mse, frac_clipped

    @jax.jit
    def apply_grads(tv, opt_vars, ema, grads):
        tv, opt_vars = optimizer.stateless_apply(opt_vars, grads, tv)
        ema = [ema_momentum * e + (1.0 - ema_momentum) * v for e, v in zip(ema, tv)]
        return tv, opt_vars, ema

    return compute_grads, apply_grads


def prepare_batch(model: GAMAutoReg, past, future, obs):
    """Windows [B, *, C] -> jnp arrays for compute_grads (rows = window x channel)."""
    past_rows, future_rows = to_series(past), to_series(future)
    lo, hi = model.scaler.row_params(past_rows.shape[0])
    buf = model.initial_buffer(model.scaler.transform_rows(past_rows, lo, hi))
    return (jnp.asarray(buf), jnp.asarray(future_rows), jnp.asarray(series_weights(obs)),
            jnp.asarray(lo), jnp.asarray(hi))


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_gam(model: GAMAutoReg, data: ProbTSData, cfg: TrainConfig) -> TrainResult:
    if data.val is None:
        raise ValueError("A validation split is needed (model selection and RidgeCV).")
    t_start = time.time()
    H, freq = data.meta.prediction_length, data.meta.freq
    rng = np.random.default_rng(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)

    def log(rec):
        if cfg.verbose:
            print(json.dumps(rec), flush=True)

    # 1. scaler
    model.fit_scaler(data.train)

    # 2. optimiser state (keras stateless API)
    optimizer = keras.optimizers.AdamW(learning_rate=cfg.learning_rate,
                                       weight_decay=cfg.weight_decay)
    optimizer.build(model.net.trainable_variables)
    opt_vars = [v.value for v in optimizer.variables]
    tv, ntv = list(model.tv), list(model.ntv)
    ema = [jnp.array(v) for v in tv]
    compute_grads, apply_grads = make_train_fns(model, optimizer, H, cfg.ema_momentum)
    accum = max(int(cfg.accumulate_grad_batches), 1)

    best = {"score": np.inf, "epoch": -1, "tv": ema, "ntv": ntv}
    history = []
    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        losses, mses, clipped = [], [], []
        acc, n_acc = None, 0
        for _ in range(cfg.batches_per_epoch):
            batch = prepare_batch(model, *data.train.sample_batch(rng, cfg.batch_size))
            grads, ntv, loss, mse, frac_clipped  = compute_grads(tv, ntv, *batch)
            acc = grads if acc is None else [a + g for a, g in zip(acc, grads)]
            n_acc += 1
            if n_acc == accum:
                tv, opt_vars, ema = apply_grads(tv, opt_vars, ema, [g / accum for g in acc])
                acc, n_acc = None, 0
            losses.append(float(loss))
            mses.append(float(mse))
            clipped.append(float(frac_clipped))
        if n_acc > 0:                                         # leftover accumulated batches
            tv, opt_vars, ema = apply_grads(tv, opt_vars, ema, [g / n_acc for g in acc])

        # 3. validation with the EMA weights, Dense(1) head
        model.set_params(ema, ntv)
        val = evaluate_probts(model, data.val, data.scaler, freq, cfg.eval_batch_size,
                              "val", cfg.quantiles_num, use_ridge=False)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "train_mse": float(np.mean(mses)),
               "frac_clipped": float(np.mean(clipped)), "val_ND": val["val_ND"],
               "val_CRPS": val["val_CRPS"], "epoch_time_s": round(time.time() - t0, 2)}
        history.append(rec)
        log(rec)
        if np.isfinite(val["val_CRPS"]) and val["val_CRPS"] < best["score"]:
            best = {"score": float(val["val_CRPS"]), "epoch": epoch,
                    "tv": [jnp.array(e) for e in ema], "ntv": ntv}

    # 4. best weights -> model; ridge head on rollout features
    model.set_params(best["tv"], best["ntv"])
    result = TrainResult(best_epoch=best["epoch"], best_score=best["score"],
                         selection_metric="val_CRPS", history=history)
    result.val_dense = evaluate_probts(model, data.val, data.scaler, freq, cfg.eval_batch_size,
                                       "val", cfg.quantiles_num, use_ridge=False)
    if cfg.fit_ridge:
        result.ridge = model.fit_ridge(data.train, data.val)
        result.val_ridge = evaluate_probts(model, data.val, data.scaler, freq, cfg.eval_batch_size,
                                           "val", cfg.quantiles_num, use_ridge=True)
    log({"best_epoch": result.best_epoch, "val_CRPS_dense": result.val_dense["val_CRPS"],
         "val_CRPS_ridge": result.val_ridge["val_CRPS"] if result.val_ridge else None,
         "ridge_alpha": result.ridge["alpha"] if result.ridge else None})
    result.train_time_s = time.time() - t_start
    return result