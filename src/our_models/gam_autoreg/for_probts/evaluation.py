"""
ProbTS metrics for the autoregressive GAM.

evaluate_probts() is passed to training_loop.train_gam (validation, with
functools.partial to fix the ProbTS arguments) and used by
src/run_gams_on_probts.py for the test set.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from probts.utils.evaluator import Evaluator
from our_models.gam_autoreg.gam_model import GAMAutoReg
from our_models.gam_autoreg.windows import EvalWindows


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
