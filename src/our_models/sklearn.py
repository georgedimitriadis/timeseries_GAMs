# ──────────────────────────────────────────────────────────────────────────────
# SklearnForecaster
#
# Plugs any scikit-learn regressor into the ProbTS benchmark:
#   • loss()     – ignores the incoming batch; loads raw data via
#                  ProbTSDatasetLoader, cuts windows, applies the same
#                  transformation as Forecaster.get_inputs, then calls
#                  sklearn_model.fit(X, y).  Runs only once (first call).
#   • forecast() – calls self.get_inputs(batch_data, 'encode') so that ALL
#                  of ProbTS's transformations (lags, embeddings, time feats)
#                  are applied automatically, then delegates to sklearn.
#
# Recommended YAML flags for sklearn usage:
#   use_lags:          true   (lag extraction is easy to replicate)
#   use_feat_idx_emb:  false  (embedding is not updated without gradients)
#   use_time_feat:     false  (requires timestamps – omit for simplicity)
#   use_scaling:       false  (TemporalScaler needs per-batch observed mask)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from probts.model.forecaster import Forecaster
from preprocessing.data_loading import ProbTSDatasetLoader   # the loader you wrote
from preprocessing.becnhmark_raw_data_transform import raw_data_transform



# ─────────────────────────────────────────────────────────────────────────────
# SklearnForecaster
# ─────────────────────────────────────────────────────────────────────────────

class SklearnForecaster(Forecaster):
    """
    Wraps any sklearn regressor inside the ProbTS Forecaster interface.

    Parameters
    ----------
    sklearn_model
        Any fitted or unfitted sklearn estimator with .fit(X, y) and
        .predict(X).  Must handle multi-output targets (y shape
        [n_samples, prediction_length * target_dim]).
    data_root_path : str
        Root path of the ProbTS datasets (same as DataManager's path).
    **kwargs
        All remaining keyword arguments are forwarded to Forecaster.__init__.
    """

    def __init__(
            self,
            sklearn_model,
            data_root_path: str,
            scaler: str,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.sklearn_model = sklearn_model
        self.data_root_path = data_root_path
        self._is_fitted = False
        self.scaler = scaler

        # history_length: how many raw steps are in past_target_cdf
        # mirrors DataManager._set_meta_parameters:
        #   history_length = context_length + max(lags_list)
        self._history_length = (
            self.max_context_length + max(self.lags_list)
            if self.lags_list
            else self.max_context_length
        )

        # Dummy parameter so Lightning can call loss.backward() cleanly
        # (zero gradient, no weight update)
        self._dummy_param = nn.Parameter(torch.zeros(1))

    # ── Training ──────────────────────────────────────────────────────────────

    def _build_training_data(self):
        """
        Load raw training arrays, cut windows, apply raw_data_transform,
        and return (X, y) ready for sklearn.fit().

        X shape: [n_windows, context_length * input_size]
        y shape: [n_windows, prediction_length * target_dim]
        """
        from probts.data.data_manager import DataManager

        # ── Load raw training arrays ──────────────────────────────────────
        loader = ProbTSDatasetLoader(
            dataset=self.dataset,
            data_root_path=self.data_root_path,
            context_length=self.max_context_length,
            prediction_length=self.max_prediction_length,
        )
        train_list, _, _ = loader.load()  # always returns lists

        # ── Apply the same scaler that ProbTSForecastModule applies to ────
        # past_target_cdf before calling loss(). Without this our raw windows
        # are in original scale while the benchmark's batches are z-scored.
        # We create a fresh DataManager to obtain its fitted scaler.
        dm = DataManager(
            dataset=self.dataset,
            path=self.data_root_path,
            context_length=self.max_context_length,
            prediction_length=self.max_prediction_length,
            split_val=False,
            scaler=self.scaler
        )


        dm_scaler = getattr(dm, 'scaler', None)

        if dm_scaler is not None:
            scaled_list = []
            for arr in train_list:
                t = torch.tensor(arr, dtype=torch.float32)
                t_scaled = dm_scaler.transform(t)
                scaled_list.append(t_scaled.detach().numpy().astype(np.float64))
            train_list = scaled_list

        # Extract frozen embedding weights once (None if not used)
        feat_idx_emb_weights = None
        if self.use_feat_idx_emb and self.feat_idx_emb is not None:
            feat_idx_emb_weights = (
                self.feat_idx_emb.weight.detach().cpu().numpy()  # [F, D]
            )

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []

        for arr in train_list:  # arr: [T, F]
            T = arr.shape[0]
            n_valid = T - self._history_length - self.max_prediction_length + 1
            if n_valid <= 0:
                continue

            for t in range(self._history_length,
                           T - self.max_prediction_length + 1):
                window = arr[t - self._history_length: t]  # [H, F]
                future = arr[t: t + self.max_prediction_length]  # [P, F]

                x = raw_data_transform(
                    window=window,
                    context_length=self.max_context_length,
                    lags_list=self.lags_list,
                    use_lags=self.use_lags,
                    use_scaling=self.use_scaling,
                    use_feat_idx_emb=self.use_feat_idx_emb,
                    feat_idx_emb_weights=feat_idx_emb_weights,
                    use_time_feat=False,  # time features need timestamps;
                    # set use_time_feat=False in YAML
                )  # [ctx, in]

                X_list.append(x.reshape(-1))  # [ctx * input_size]
                y_list.append(future.reshape(-1))  # [pred_len * F]

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        return X, y

    def loss(self, batch_data) -> torch.Tensor:
        """
        On the first call: load raw data, fit the sklearn model.
        Always returns a zero-gradient dummy loss so Lightning's
        training step completes without error.
        """
        if not self._is_fitted:
            X, y = self._build_training_data()
            self.sklearn_model.fit(X, y)
            self._is_fitted = True

        # Zero-gradient loss — keeps Lightning happy without touching weights
        return self._dummy_param.sum() * 0.0

    # ── Evaluation ────────────────────────────────────────────────────────────

    def forecast(
        self,
        batch_data=None,
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Called by ProbTS's test pipeline.

        self.get_inputs() handles ALL of ProbTS's transformations (lag
        extraction, feature-index embeddings, time features) identically to
        how a PyTorch model would see the data — no replication needed here.

        Returns
        -------
        torch.Tensor  shape [B, 1, prediction_length, target_dim]
            The 1 in dim-1 signals a point forecast (no sample dimension).
        """
        # TemporalScaler must be fitted before get_inputs() reads self.scaler.scale
        if self.use_scaling:
            self.get_scale(batch_data)

        # [B, context_length, input_size]  — exactly what a linear layer sees
        inputs = self.get_inputs(batch_data, 'encode')

        B = inputs.shape[0]
        x_np = inputs.detach().cpu().numpy().reshape(B, -1)  # [B, ctx*in]
        preds = self.sklearn_model.predict(x_np)             # [B, P*F]

        forecasts = (
            torch.tensor(preds, dtype=torch.float32)
            .reshape(B, self.max_prediction_length, self.target_dim)
            .unsqueeze(1)                                     # [B, 1, P, F]
        )
        return forecasts