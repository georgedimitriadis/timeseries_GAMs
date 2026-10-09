"""
Step 3 — training loop for the autoregressive GAM, no teacher forcing.
General: used by both the ProbTS and the dysts pipelines.

    result = train_gam(model, train, val, TrainConfig(...), evaluate,
                       select_metric="val_CRPS", log_metrics=("val_ND", "val_CRPS"))

    evaluate(model, windows, stage=..., use_ridge=...) -> dict of metrics
        pipeline-specific (for_probts/evaluation.py, for_dysts/evaluation.py);
        keys are prefixed with the stage, e.g. "val_CRPS". Lower = better for
        select_metric.

What it does
------------
1. Fits the per-channel MinMax scaler on the train series (model.fit_scaler).
2. Trains the ec network + Dense(1) head on free-running rollouts:
     - batch      : TrainWindows.sample_batch -> rows (one per window x channel)
     - forward    : model.rollout(training=True) over train.prediction_length steps,
                    each prediction fed back as the newest input
     - loss       : MSE in the train-series space (standard-scaled for ProbTS,
                    original for dysts), weighted by the observed mask (as ProbTS
                    Forecaster.get_weighted_loss), plus the EquationLayer penalty
     - optimiser  : keras AdamW (stateless, on the JAX backend) + EMA of the
                    weights; EMA weights are used for every evaluation, as
                    AdamW(use_ema=True) + SwapEMAWeights in ECBase._fit
     - accumulate_grad_batches
3. After every epoch: evaluate(val) with the Dense(1) head; the epoch with the
   lowest select_metric is kept.
4. Puts the best weights in the model and fits the RidgeCV head on rollout
   features (model.fit_ridge); reports val metrics with both heads.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import keras
import numpy as np

from our_models.gam_autoreg.gam_model import ChannelMinMaxScaler, GAMAutoReg
from our_models.gam_autoreg.windows import EvalWindows, TrainWindows, series_weights, to_series


# ----------------------------------------------------------------------
# Config / result
# ----------------------------------------------------------------------
@dataclass
class TrainConfig:
    max_epochs: int = 50                  # (ProbTS: trainer.max_epochs)
    batches_per_epoch: int = 100          # (ProbTS: trainer.limit_train_batches)
    batch_size: int = 32                  # windows per batch; rows = batch_size * C
    accumulate_grad_batches: int = 1      # (ProbTS: trainer.accumulate_grad_batches)
    eval_batch_size: int = 32             # (ProbTS: data.test_batch_size, metric averaging per batch)
    learning_rate: float = 0.01           # ECBase default
    weight_decay: float = 0.004           # keras AdamW default
    ema_momentum: float = 0.99            # keras optimizer EMA default
    quantiles_num: int = 20               # ProbTS Evaluator (unused for dysts)
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
                        else jnp.asarray(0.0))                           # share of predictions at the bound
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
def train_gam(model: GAMAutoReg, train: TrainWindows, val: EvalWindows, cfg: TrainConfig,
              evaluate: Callable[..., dict], select_metric: str = "val_CRPS",
              log_metrics: Sequence[str] = ()) -> TrainResult:
    """
    train          TrainWindows (training rollouts of train.prediction_length steps)
    val            EvalWindows  (epoch selection and RidgeCV fold)
    evaluate       evaluate(model, windows, stage=..., use_ridge=...) -> dict
    select_metric  key of the evaluate() output used to pick the epoch (lower = better)
    log_metrics    extra evaluate() keys written to the epoch log
    """
    if val is None:
        raise ValueError("A validation split is needed (model selection and RidgeCV).")
    t_start = time.time()
    H = train.prediction_length
    rng = np.random.default_rng(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)

    def log(rec):
        if cfg.verbose:
            print(json.dumps(rec), flush=True)

    # 1. scaler
    model.fit_scaler(train)

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
            batch = prepare_batch(model, *train.sample_batch(rng, cfg.batch_size))
            grads, ntv, loss, mse, frac_clipped = compute_grads(tv, ntv, *batch)
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
        val_m = evaluate(model, val, stage="val", use_ridge=False)
        score = val_m[select_metric]
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "train_mse": float(np.mean(mses)),
               "frac_clipped": float(np.mean(clipped)),
               **{k: val_m[k] for k in log_metrics if k != select_metric},
               select_metric: score, "epoch_time_s": round(time.time() - t0, 2)}
        history.append(rec)
        log(rec)
        if score is not None and np.isfinite(score) and score < best["score"]:
            best = {"score": float(score), "epoch": epoch,
                    "tv": [jnp.array(e) for e in ema], "ntv": ntv}

    # 4. best weights -> model; ridge head on rollout features
    model.set_params(best["tv"], best["ntv"])
    result = TrainResult(best_epoch=best["epoch"], best_score=best["score"],
                         selection_metric=select_metric, history=history)
    result.val_dense = evaluate(model, val, stage="val", use_ridge=False)
    if cfg.fit_ridge:
        result.ridge = model.fit_ridge(train, val)
        result.val_ridge = evaluate(model, val, stage="val", use_ridge=True)
    log({"best_epoch": result.best_epoch, f"{select_metric}_dense": result.val_dense[select_metric],
         f"{select_metric}_ridge": result.val_ridge[select_metric] if result.val_ridge else None,
         "ridge_alpha": result.ridge["alpha"] if result.ridge else None})
    result.train_time_s = time.time() - t_start
    return result
