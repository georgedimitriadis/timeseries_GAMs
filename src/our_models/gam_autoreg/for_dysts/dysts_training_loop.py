"""
Step 3 (dysts) — training loop for the dysts GAM (DystsGAM), autoregressive,
no teacher forcing.

    result = train_dysts_gam(model, data.train, data.val, DystsTrainConfig(...), evaluate)

    evaluate(model, windows, stage=..., use_ridge=...) -> dict
        for_dysts/dysts_evaluation.evaluate_dysts with dt / lambda_max fixed (functools.partial)

What it does
------------
1. Fits the per-coordinate MinMax scaler on the train series (model.fit_scaler).
2. Trains the D networks (+ Dense(1) heads) together on free-running rollouts:
     - epoch      : one pass over the training windows in random order (as a Darts
                    epoch); batches of batch_size windows; the last incomplete batch
                    is dropped (constant shapes -> one compilation)
     - forward    : model.loss -> model.rollout(training=True) over train.prediction_length
                    steps, all D predictions fed back together
     - loss       : MSE on z values (mean over coordinates) + EquationLayer penalties
     - optimiser  : keras AdamW (stateless, JAX backend), optional global_clipnorm,
                    + EMA of the weights; EMA weights are used for every evaluation
                    (as AdamW(use_ema=True) + SwapEMAWeights in ECBase._fit)
3. After every epoch: evaluate(val) with the Dense(1) heads; the weights of the epoch
   with the lowest val sMAPE are kept (select_metric="val_smape"). Optional early
   stopping after `patience` epochs without improvement.
4. Puts the best weights in the model and fits one RidgeCV head per coordinate on
   rollout features (model.fit_ridge); reports val metrics with both heads.
"""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import keras
import numpy as np

from our_models.gam_autoreg.for_dysts.dysts_gam_model import DystsGAM
from our_models.gam_autoreg.windows import EvalWindows, TrainWindows


# ----------------------------------------------------------------------
# Config / result
# ----------------------------------------------------------------------
@dataclass
class DystsTrainConfig:
    max_epochs: int = 100                   # Darts default n_epochs (N-BEATS)
    batch_size: int = 32                    # Darts default batch_size
    batches_per_epoch: Optional[int] = None # None = one pass over the training windows
    accumulate_grad_batches: int = 1
    learning_rate: float = 0.01             # ECBase default
    weight_decay: float = 0.004             # keras AdamW default
    ema_momentum: float = 0.99              # keras optimizer EMA default
    global_clipnorm: Optional[float] = None # gradient clipping (None = off)
    patience: Optional[int] = None          # early stopping on val_smape (None = off)
    select_metric: str = "val_smape"        # lower = better
    fit_ridge: bool = True
    seed: int = 0
    verbose: bool = True


@dataclass
class DystsTrainResult:
    best_epoch: int
    best_score: float
    selection_metric: str
    epochs_run: int = 0
    history: list = field(default_factory=list)
    val_dense: Optional[dict] = None        # val metrics, Dense(1) heads
    val_ridge: Optional[dict] = None        # val metrics, ridge heads
    ridge: Optional[list] = None            # per coordinate: alpha, rows
    train_time_s: float = 0.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def epoch_batches(n_windows: int, batch_size: int, rng: np.random.Generator,
                  batches_per_epoch: Optional[int] = None):
    """
    Window indices for one epoch: a random permutation cut into full batches
    (the last incomplete batch is dropped). batches_per_epoch caps the number of
    batches; if it asks for more than one pass, further permutations are appended.
    """
    n_full = n_windows // batch_size
    if n_full == 0:
        raise ValueError(f"batch_size {batch_size} > number of training windows {n_windows}")
    n_batches = n_full if batches_per_epoch is None else int(batches_per_epoch)
    batches = []
    while len(batches) < n_batches:
        perm = rng.permutation(n_windows)
        batches += [perm[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
    return batches[:n_batches]


def make_train_fns(model: DystsGAM, optimizer, ema_momentum: float):
    """
    compute_grads(tv, ntv, buf, future_z, w) -> grads, ntv, loss, mse, frac_clipped
    apply_grads(tv, opt_vars, ema, grads)    -> tv, opt_vars, ema
    """
    grad_fn = jax.value_and_grad(model.loss, has_aux=True)

    @jax.jit
    def compute_grads(tv, ntv, buf, future_z, w):
        (loss, (ntv, mse, frac_clipped)), grads = grad_fn(tv, ntv, buf, future_z, w)
        return grads, ntv, loss, mse, frac_clipped

    @jax.jit
    def apply_grads(tv, opt_vars, ema, grads):
        tv, opt_vars = optimizer.stateless_apply(opt_vars, grads, tv)
        ema = [ema_momentum * e + (1.0 - ema_momentum) * v for e, v in zip(ema, tv)]
        return tv, opt_vars, ema

    return compute_grads, apply_grads


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_dysts_gam(model: DystsGAM, train: TrainWindows, val: EvalWindows,
                    cfg: DystsTrainConfig, evaluate: Callable[..., dict]) -> DystsTrainResult:
    if val is None:
        raise ValueError("A validation set is needed (epoch selection and RidgeCV).")
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)
    sel = cfg.select_metric

    def log(rec):
        if cfg.verbose:
            print(json.dumps(rec), flush=True)

    # 1. scaler
    model.fit_scaler(train)

    # 2. optimiser state (keras stateless API), all D networks together
    optimizer = keras.optimizers.AdamW(learning_rate=cfg.learning_rate,
                                       weight_decay=cfg.weight_decay,
                                       global_clipnorm=cfg.global_clipnorm)
    optimizer.build(model.trainable_variables)
    opt_vars = [v.value for v in optimizer.variables]
    tv, ntv = list(model.tv), list(model.ntv)
    ema = [jnp.array(v) for v in tv]
    compute_grads, apply_grads = make_train_fns(model, optimizer, cfg.ema_momentum)
    accum = max(int(cfg.accumulate_grad_batches), 1)

    best = {"score": np.inf, "epoch": -1, "tv": ema, "ntv": ntv}
    history, since_best = [], 0
    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        losses, mses, clipped = [], [], []
        acc, n_acc = None, 0
        for idx in epoch_batches(len(train), cfg.batch_size, rng, cfg.batches_per_epoch):
            past, future, obs = train.windows(train.starts[idx])
            buf, future_z, w = model.prepare_batch(past, future, obs)
            grads, ntv, loss, mse, frac_clipped = compute_grads(tv, ntv, buf, future_z, w)
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

        # 3. validation with the EMA weights, Dense(1) heads
        model.set_params(ema, ntv)
        val_m = evaluate(model, val, stage="val", use_ridge=False)
        score = val_m.get(sel)
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "train_mse": float(np.mean(mses)), "frac_clipped": float(np.mean(clipped)),
               sel: score, "val_valid_horizon_lyap": val_m.get("val_valid_horizon_lyap"),
               "epoch_time_s": round(time.time() - t0, 2)}
        history.append(rec)
        log(rec)
        if score is not None and np.isfinite(score) and score < best["score"]:
            best = {"score": float(score), "epoch": epoch,
                    "tv": [jnp.array(e) for e in ema], "ntv": ntv}
            since_best = 0
        else:
            since_best += 1
        if cfg.patience is not None and since_best >= cfg.patience:
            log({"early_stop": epoch, "best_epoch": best["epoch"]})
            break

    if best["epoch"] < 0:
        raise RuntimeError(f"No epoch had a finite {sel}; see the epoch log.")

    # 4. best weights -> model; ridge heads on rollout features
    model.set_params(best["tv"], best["ntv"])
    result = DystsTrainResult(best_epoch=best["epoch"], best_score=best["score"],
                              selection_metric=sel, epochs_run=len(history), history=history)
    result.val_dense = evaluate(model, val, stage="val", use_ridge=False)
    if cfg.fit_ridge:
        ridge = model.fit_ridge(train, val)
        result.ridge = [{k: r[k] for k in ("alpha", "n_train_rows", "n_val_rows")} for r in ridge]
        result.val_ridge = evaluate(model, val, stage="val", use_ridge=True)
    log({"best_epoch": result.best_epoch, f"{sel}_dense": result.val_dense.get(sel),
         f"{sel}_ridge": result.val_ridge.get(sel) if result.val_ridge else None,
         "ridge_alphas": [r["alpha"] for r in result.ridge] if result.ridge else None})
    result.train_time_s = time.time() - t_start
    return result
