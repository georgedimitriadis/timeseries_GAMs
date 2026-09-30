import numpy as np


def smape_0_200(y_true, y_pred, eps=1e-10):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(200 * np.mean(np.abs(y_true - y_pred) /
                               (np.abs(y_true) + np.abs(y_pred) + eps)))


def smape_per_step_0_100(y_true, y_pred, eps=1e-10):
    """sMAPE at each forecast step (0-100 scale). Returns array of length T.
    y_true, y_pred: (T, D). Mean over D per step."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    per_step = 100 * np.mean(num / den, axis=1)   # mean over dims
    return per_step


def valid_horizon(y_true, y_pred, dt, lam, threshold):
    """Latest step where per-step sMAPE(0-100) < threshold.
    Returns (step, time, lyap). step=0 if step 0 already >= threshold.
    lyap=None if lam == 0."""
    per_step = smape_per_step_0_100(y_true, y_pred)
    below = per_step < threshold
    if not below.any():
        step = 0
    else:
        step = int(np.max(np.nonzero(below)[0]))
    t = step * dt
    lyap = None if (lam is None or lam == 0) else t * lam
    return step, float(t), (None if lyap is None else float(lyap)), per_step.tolist()