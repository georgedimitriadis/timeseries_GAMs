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

def smape_cumulative_0_200(y_true, y_pred, eps=1e-10):
    """Running-average sMAPE from the forecast start (0-200 scale), as Gilpin,
    Phys. Rev. Research 2023 (arXiv 2303.08011), Appendix F:
        eps(t) = 200 / t * sum_{t'=1..t} |y - yhat| / (|y| + |yhat|)
    y_true, y_pred: (T, D); mean over D per step, then running mean over steps.
    Returns array of length T (entry i = eps(t = i + 1))."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ratio = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps), axis=1)
    return 200 * np.cumsum(ratio) / np.arange(1, len(ratio) + 1)


def valid_horizon_cumulative(y_true, y_pred, dt, lam, threshold=50.0):
    """Valid forecast horizon as Gilpin 2023 (Appendix A): latest time t at which the
    running-average sMAPE (0-200 scale) is below threshold (50 in the paper).
    t counts forecast steps from 1 (eps(t) averages steps 1..t), so time = t * dt.
    Returns (steps, time, lyap, cumulative sMAPE per step). steps = 0 if eps(1) >= threshold.
    lyap = time * lambda_max (None if lam is None or 0)."""
    cum = smape_cumulative_0_200(y_true, y_pred)
    below = np.nonzero(cum < threshold)[0]
    steps = int(below.max()) + 1 if below.size else 0
    t = steps * dt
    lyap = None if (lam is None or lam == 0) else t * lam
    return steps, float(t), (None if lyap is None else float(lyap)), cum.tolist()