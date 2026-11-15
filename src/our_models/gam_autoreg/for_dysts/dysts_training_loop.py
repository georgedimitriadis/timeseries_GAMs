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
   Rollback: an epoch is bad if a batch has a non-finite loss / gradient or a gradient
   norm > rollback_grad_norm (the epoch is stopped at that batch, before the update),
   or if its mean frac_clipped > rollback_frac_clipped or the weights are non-finite.
   On a bad epoch e, training goes back two epochs: weights, optimiser state and EMA
   are reset to the end of epoch e-2 (to the initial state if e < 2), the records of
   epochs e-1 and e are dropped, and training continues from epoch e-1 (epoch 0 if
   e < 2; to the oldest kept state if e-2 is older than the last rollback_keep_epochs
   states, which can happen when a redone epoch is bad again). The batch generator is not reset, so the repeated epochs get new batch
   orders; the dropout seed state in ntv is not restored either. When epoch e has
   been bad more than max_epoch_retries times: training stops; the best epoch so far
   is used (as early stopping). The best weights are kept even if their epoch is redone.
3. After every epoch: evaluate(val) with the Dense(1) heads; the weights of the epoch
   with the lowest val sMAPE are kept (select_metric="val_smape"); with
   select_smoothing = k > 1 the score of epoch c is the select_stat (mean, median or
   max) of the val sMAPE of the k epochs centred on c (c - (k-1)//2 .. c + k//2), known k//2 epochs later; the EMA weights of
   the last (k-1)//2 + 1 epochs are kept for this. If no full window is available
   (training stopped early), the epoch with the lowest raw val sMAPE is used.
   select_last_epochs = N > 0: only epochs among the last N epochs of training (counted
   back from the last accepted epoch) can be picked; the best of them is tracked during
   training with a sliding-window minimum (EMA weights kept only for epochs that can still
   become that minimum). If none of the last N epochs has a score, the best epoch over
   all epochs is used.
   Optional early stopping after `patience` epochs without improvement of that score.
   Learning rate: constant, or cosine decay from learning_rate to lr_min over
   max_epochs (lr_schedule="cosine"); the schedule follows the optimiser step counter,
   so a rollback also restores the learning rate.
   If `test` windows are given, evaluate(test) also runs after every epoch (Dense(1)
   heads), for monitoring only: it is not used to pick the epoch.
   result.val_results / result.test_results: val / test sMAPE per accepted epoch
   (same order as result.history; epochs undone by a rollback are removed).
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
    learning_rate: float = 0.01             # ECBase default (initial value with a schedule)
    lr_schedule: str = "constant"           # "constant" | "cosine" (cosine decay over max_epochs)
    lr_min: float = 0.0                     # final learning rate of the cosine schedule
    weight_decay: float = 0.004             # keras AdamW default
    ema_momentum: float = 0.99              # keras optimizer EMA default
    global_clipnorm: Optional[float] = None # gradient clipping (None = off)
    patience: Optional[int] = None          # early stopping on val_smape (None = off)
    # rollback: on a bad epoch e, go back to the end of epoch e-2 (initial state if e < 2) and
    # continue from there with new batch orders; when epoch e is bad more than max_epoch_retries
    # times -> training stops, the best epoch so far is used
    max_epoch_retries: int = 5
    rollback_keep_epochs: int = 10                    # end-of-epoch states kept for going back (+ initial)
    rollback_grad_norm: Optional[float] = 100.0       # bad if a batch grad norm exceeds it (None = off)
    rollback_frac_clipped: Optional[float] = 0.1      # bad if the epoch mean frac_clipped exceeds it (None = off)
    select_metric: str = "val_smape"        # lower = better
    select_smoothing: int = 1               # pick the epoch with the lowest select_stat of select_metric
                                            # over the k epochs centred on it; 1 = raw
    select_stat: str = "mean"               # "mean" | "median" | "max" (max: worst epoch of the window)
    select_last_epochs: int = -1            # pick only among the last N epochs of training (-1 = all)
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
    rollbacks: list = field(default_factory=list)   # {"epoch", "retry", "reason", "restart_epoch"} per bad run
    stopped_diverged: Optional[dict] = None         # {"epoch", "reason"} if stopped after max_epoch_retries
    val_results: list = field(default_factory=list)  # val sMAPE per epoch (Dense(1) heads)
    val_results_smoothed: list = field(default_factory=list)  # centred select_stat per epoch (None at the ends)
    test_results: list = field(default_factory=list) # test sMAPE per epoch (Dense(1) heads; monitoring only)
    train_loss_results: list = field(default_factory=list)  # mean train loss per epoch (MSE + penalties)
    train_mse_results: list = field(default_factory=list)   # mean train MSE per epoch (z space)
    val_dense: Optional[dict] = None        # val metrics, Dense(1) heads
    val_ridge: Optional[dict] = None        # val metrics, ridge heads
    ridge: Optional[list] = None            # per coordinate: alpha, rows
    train_time_s: float = 0.0


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def all_finite(arrays) -> bool:
    return all(bool(jnp.all(jnp.isfinite(a))) for a in arrays)


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
    compute_grads(tv, ntv, buf, future_z, w) -> grads, ntv, loss, mse, frac_clipped, grad_norm
    apply_grads(tv, opt_vars, ema, grads)    -> tv, opt_vars, ema
    """
    grad_fn = jax.value_and_grad(model.loss, has_aux=True)

    @jax.jit
    def compute_grads(tv, ntv, buf, future_z, w):
        (loss, (ntv, mse, frac_clipped)), grads = grad_fn(tv, ntv, buf, future_z, w)
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads))   # global norm (diagnostic)
        return grads, ntv, loss, mse, frac_clipped, grad_norm

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
                    cfg: DystsTrainConfig, evaluate: Callable[..., dict],
                    test: Optional[EvalWindows] = None) -> DystsTrainResult:
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
    accum = max(int(cfg.accumulate_grad_batches), 1)
    n_batches = len(epoch_batches(len(train), cfg.batch_size, np.random.default_rng(0),
                                  cfg.batches_per_epoch))
    updates_per_epoch = -(-n_batches // accum)                # ceil
    if cfg.lr_schedule == "cosine":
        lr = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg.learning_rate,
            decay_steps=max(cfg.max_epochs * updates_per_epoch, 1),
            alpha=cfg.lr_min / cfg.learning_rate)
    elif cfg.lr_schedule == "constant":
        lr = cfg.learning_rate
    else:
        raise ValueError(f"lr_schedule {cfg.lr_schedule!r}: use 'constant' or 'cosine'")
    optimizer = keras.optimizers.AdamW(learning_rate=lr,
                                       weight_decay=cfg.weight_decay,
                                       global_clipnorm=cfg.global_clipnorm)
    optimizer.build(model.trainable_variables)
    opt_vars = [v.value for v in optimizer.variables]
    it_idx = next((i for i, v in enumerate(optimizer.variables) if v is optimizer.iterations), None)

    def current_lr(ov):                                       # learning rate of the next update (log)
        if not callable(lr):
            return float(lr)
        return float(lr(ov[it_idx])) if it_idx is not None else None
    tv, ntv = list(model.tv), list(model.ntv)
    ema = [jnp.array(v) for v in tv]
    compute_grads, apply_grads = make_train_fns(model, optimizer, cfg.ema_momentum)

    def run_epoch(tv, ntv, opt_vars, ema, batches):
        """One epoch. Returns (tv, ntv, opt_vars, ema, stats, bad_reason or None)."""
        losses, mses, clipped, gnorms = [], [], [], []
        acc, n_acc = None, 0
        stats = lambda: {"train_loss": float(np.mean(losses)) if losses else None,
                         "train_mse": float(np.mean(mses)) if mses else None,
                         "frac_clipped": float(np.mean(clipped)) if clipped else None,
                         "grad_norm_median": float(np.median(gnorms)) if gnorms else None,
                         "grad_norm_max": float(np.max(gnorms)) if gnorms else None}
        for b, idx in enumerate(batches):
            past, future, obs = train.windows(train.starts[idx])
            buf, future_z, w = model.prepare_batch(past, future, obs)
            grads, ntv_new, loss, mse, frac_clipped, grad_norm = compute_grads(tv, ntv, buf, future_z, w)
            loss, grad_norm = float(loss), float(grad_norm)
            if not (np.isfinite(loss) and np.isfinite(grad_norm)):
                return tv, ntv, opt_vars, ema, stats(), f"non-finite loss/gradient (batch {b})"
            if cfg.rollback_grad_norm is not None and grad_norm > cfg.rollback_grad_norm:
                return tv, ntv, opt_vars, ema, stats(), f"grad norm {grad_norm:.3g} (batch {b})"
            ntv = ntv_new
            acc = grads if acc is None else [a + g for a, g in zip(acc, grads)]
            n_acc += 1
            if n_acc == accum:
                tv, opt_vars, ema = apply_grads(tv, opt_vars, ema, [g / accum for g in acc])
                acc, n_acc = None, 0
            losses.append(loss)
            mses.append(float(mse))
            clipped.append(float(frac_clipped))
            gnorms.append(grad_norm)
        if n_acc > 0:                                         # leftover accumulated batches
            tv, opt_vars, ema = apply_grads(tv, opt_vars, ema, [g / n_acc for g in acc])
        st = stats()
        if cfg.rollback_frac_clipped is not None and st["frac_clipped"] > cfg.rollback_frac_clipped:
            return tv, ntv, opt_vars, ema, st, f"frac_clipped {st['frac_clipped']:.3g}"
        if not (all_finite(tv) and all_finite(ema)):
            return tv, ntv, opt_vars, ema, st, "non-finite weights"
        return tv, ntv, opt_vars, ema, st, None

    best = {"score": np.inf, "epoch": -1, "tv": ema, "ntv": ntv}       # selection (smoothed)
    best_raw = dict(best)                                               # fallback: raw val score
    k_sm = max(int(cfg.select_smoothing), 1)
    stat_fn = {"mean": np.mean, "median": np.median, "max": np.max}.get(cfg.select_stat)
    if stat_fn is None:
        raise ValueError(f"select_stat {cfg.select_stat!r}: use 'mean', 'median' or 'max'")
    half = (k_sm - 1) // 2                                              # centre = half epochs back
    recent_ema = {}                                    # epoch -> (EMA weights, ntv), last half+1 epochs
    n_last = int(cfg.select_last_epochs)
    cand = []      # select_last_epochs > 0: (score, centre, scored_at, tv, ntv), scores increasing
    history, rollbacks, since_best, stopped = [], [], 0, None
    snapshots = {-1: (tv, opt_vars, ema)}      # end-of-epoch states (tv, opt_vars, ema); -1 = initial
    n_bad = {}                                 # epoch -> number of bad runs
    epoch = 0
    while epoch < cfg.max_epochs:
        t0 = time.time()
        start = snapshots[epoch - 1]                          # state at the end of the previous epoch
        batches = epoch_batches(len(train), cfg.batch_size, rng, cfg.batches_per_epoch)
        tv_e, ntv, opt_e, ema_e, st, bad = run_epoch(start[0], ntv, start[1], start[2], batches)
        if bad is not None:
            n_bad[epoch] = n_bad.get(epoch, 0) + 1
            back = max(epoch - 2, -1)                         # go back two epochs
            if back not in snapshots:                         # older than the kept states:
                back = min(k for k in snapshots if k >= 0)    # oldest kept epoch
            rollbacks.append({"epoch": epoch, "retry": n_bad[epoch], "reason": bad,
                              "restart_epoch": back + 1})
            log({"epoch": epoch, "retry": n_bad[epoch], "rollback": bad,
                 "restart_epoch": back + 1, **st})
            if n_bad[epoch] > cfg.max_epoch_retries:
                stopped = {"epoch": epoch, "reason": f"{n_bad[epoch]} bad runs (last: {bad})"}
                log({"stop_diverged": epoch, "reason": stopped["reason"], "best_epoch": best["epoch"]})
                break
            history = [h for h in history if h["epoch"] <= back]
            snapshots = {k: v for k, v in snapshots.items() if k <= back}
            recent_ema = {k: v for k, v in recent_ema.items() if k <= back}
            cand = [c for c in cand if c[2] <= back]          # scores computed from undone epochs
            epoch = back + 1
            continue
        tv, opt_vars, ema = tv_e, opt_e, ema_e
        snapshots[epoch] = (tv, opt_vars, ema)
        snapshots = {k: v for k, v in snapshots.items()                      # initial + last kept
                     if k == -1 or k > epoch - cfg.rollback_keep_epochs}

        # 3. validation with the EMA weights, Dense(1) heads
        model.set_params(ema, ntv)
        val_m = evaluate(model, val, stage="val", use_ridge=False)
        raw = val_m.get(sel)
        test_m = evaluate(model, test, stage="test", use_ridge=False) if test is not None else {}
        rec = {"epoch": epoch, **st, "retries": n_bad.get(epoch, 0), "lr": current_lr(opt_vars),
               sel: raw, f"{sel}_smoothed": None,
               "val_valid_horizon_lyap": val_m.get("val_valid_horizon_lyap"),
               "test_smape": test_m.get("test_smape"),
               "test_valid_horizon_lyap": test_m.get("test_valid_horizon_lyap"),
               "epoch_time_s": round(time.time() - t0, 2)}
        history.append(rec)
        finite = lambda x: x is not None and np.isfinite(x)
        if finite(raw) and raw < best_raw["score"]:
            best_raw = {"score": float(raw), "epoch": epoch, "tv": [jnp.array(e) for e in ema], "ntv": ntv}

        # smoothed selection: select_stat over the k_sm epochs centred on epoch c = epoch - half
        recent_ema[epoch] = ([jnp.array(e) for e in ema], ntv)
        recent_ema = {k: v for k, v in recent_ema.items() if k >= epoch - half}
        improved = False
        if len(history) >= k_sm:
            window = [h[sel] for h in history[-k_sm:]]
            centre = history[-1 - half]
            if all(finite(w) for w in window) and centre["epoch"] in recent_ema:
                score = float(stat_fn(window))
                centre[f"{sel}_smoothed"] = score
                rec["smoothed_centre"] = centre["epoch"]
                rec["smoothed_score"] = score
                w_tv, w_ntv = recent_ema[centre["epoch"]]
                if score < best["score"]:
                    best = {"score": score, "epoch": centre["epoch"], "tv": w_tv, "ntv": w_ntv}
                    improved = True
                if n_last > 0:                                # sliding-window minimum
                    while cand and cand[-1][0] >= score:
                        cand.pop()
                    cand.append((score, centre["epoch"], epoch, w_tv, w_ntv))
        if n_last > 0:
            cand = [c for c in cand if c[1] > epoch - n_last]
        log(rec)
        since_best = 0 if improved else since_best + 1
        if cfg.patience is not None and since_best >= cfg.patience:
            log({"early_stop": epoch, "best_epoch": best["epoch"]})
            break
        epoch += 1

    if n_last > 0 and history:                # best among the last n_last epochs of training
        last = history[-1]["epoch"]
        cand = [c for c in cand if c[1] > last - n_last]
        if cand:
            score, c_ep, _, w_tv, w_ntv = cand[0]
            best = {"score": score, "epoch": c_ep, "tv": w_tv, "ntv": w_ntv}
        else:
            log({"select_last_epochs": n_last, "note": "no scored epoch in that range; "
                 "using the best epoch over all epochs", "best_epoch": best["epoch"]})
    if best["epoch"] < 0:                     # fewer than select_smoothing finite epochs
        best = best_raw
    if best["epoch"] < 0:
        raise RuntimeError(f"No epoch had a finite {sel}; see the epoch log.")

    # 4. best weights -> model; ridge heads on rollout features
    model.set_params(best["tv"], best["ntv"])
    result = DystsTrainResult(best_epoch=best["epoch"], best_score=best["score"],
                              selection_metric=sel, epochs_run=len(history), history=history,
                              rollbacks=rollbacks, stopped_diverged=stopped,
                              val_results=[h[sel] for h in history],
                              val_results_smoothed=[h[f"{sel}_smoothed"] for h in history],
                              test_results=[h["test_smape"] for h in history],
                              train_loss_results=[h["train_loss"] for h in history],
                              train_mse_results=[h["train_mse"] for h in history])
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