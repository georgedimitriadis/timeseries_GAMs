# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# We thank the authors for their contributions.
#
# This is an auto-regressive variant of DLinear. Instead of mapping the context
# window directly to the whole prediction horizon in a single shot, the linear
# heads map the context window to a *single* next step. At inference the predicted
# step is appended to the input window and fed back in to predict the following
# step, in the same manner as the GRU forecaster.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from probts.data import ProbTSBatchData
from probts.model.forecaster import Forecaster
from probts.model.nn.arch.decomp import series_decomp


class DLinearAutoReg(Forecaster):
    def __init__(
        self,
        kernel_size: int,
        individual: bool,
        teacher_forcing: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.autoregressive = True

        # When True (default), each training step predicts the next point from the
        # ground-truth window. When False, the model is unrolled during training:
        # its own predictions are appended to the moving window and fed back in,
        # exactly as at inference time.
        self.teacher_forcing = teacher_forcing

        if self.input_size != self.target_dim:
            self.enc_linear = nn.Linear(
                in_features=self.input_size, out_features=self.target_dim
            )
        else:
            self.enc_linear = nn.Identity()

        # Decomposition Kernel Size
        self.kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual

        # Auto-regressive: each head maps the context window to a SINGLE next step.
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.target_dim):
                self.Linear_Seasonal.append(nn.Linear(self.context_length, 1))
                self.Linear_Trend.append(nn.Linear(self.context_length, 1))
        else:
            self.Linear_Seasonal = nn.Linear(self.context_length, 1)
            self.Linear_Trend = nn.Linear(self.context_length, 1)
        self.loss_fn = nn.MSELoss(reduction='none')

    def encoder(self, inputs):
        """
        Predict a single next step from a window of length `context_length`.

        inputs: [B, context_length, C]
        returns: [B, 1, C]
        """
        seasonal_init, trend_init = self.decompsition(inputs)

        # [B, C, L]
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), 1],
                dtype=seasonal_init.dtype
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), 1],
                dtype=trend_init.dtype
            ).to(trend_init.device)
            for i in range(self.target_dim):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        outputs = seasonal_output + trend_output  # [B, C, 1]
        return outputs.permute(0, 2, 1)  # [B, 1, C]

    def loss(self, batch_data):
        if self.teacher_forcing:
            outputs = self._loss_teacher_forcing(batch_data)
        else:
            outputs = self._loss_free_running(batch_data)

        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def _loss_teacher_forcing(self, batch_data):
        # Teacher forcing: build the full ground-truth sequence, then predict each
        # of the `prediction_length` next-steps from a sliding context window over
        # the true history. This mirrors the GRU forecaster, which trains on the
        # concatenated sequence, while keeping DLinear's per-step decomposition.
        inputs = self.get_inputs(batch_data, 'all')  # [B, L_ctx + L_pred, input_size]
        inputs = self.enc_linear(inputs)             # [B, L_ctx + L_pred, C]

        # For predicting step (context_length + k), the input window is
        # inputs[:, k : k + context_length, :], for k in [0, prediction_length).
        outputs = []
        for k in range(self.prediction_length):
            window = inputs[:, k: k + self.context_length, ...]
            step = self.encoder(window)  # [B, 1, C]
            outputs.append(step)
        return torch.cat(outputs, dim=1)  # [B, prediction_length, C]

    def _loss_free_running(self, batch_data):
        # No teacher forcing: unroll the model during training. Each step's own
        # prediction is appended to the moving window and fed back in to predict
        # the next step, so training matches the inference-time dynamics. Gradients
        # flow through the fed-back predictions (no detach), which is what makes
        # this differ from teacher forcing rather than just re-running forecast().
        #
        # We grow `past_target_cdf` exactly as forecast() does, and rebuild each
        # window through get_inputs(..., 'encode') so that any extra input features
        # (time features, lags, feat-idx embeddings) are handled consistently
        # regardless of the model's feature configuration.
        past_target_cdf = batch_data.past_target_cdf

        outputs = []
        for k in range(self.prediction_length):
            current_batch_data = ProbTSBatchData({
                'target_dimension_indicator': batch_data.target_dimension_indicator,
                'past_target_cdf': past_target_cdf,
                'future_time_feat': batch_data.future_time_feat[:, k: k + 1:, ...]
                if batch_data.future_time_feat is not None else None,
            }, device=batch_data.device)

            inputs = self.get_inputs(current_batch_data, 'encode')  # [B, L_ctx, input_size]
            inputs = self.enc_linear(inputs)                        # [B, L_ctx, C]
            step = self.encoder(inputs)                             # [B, 1, C]
            outputs.append(step)

            # Feed the model's OWN prediction back into the moving window.
            past_target_cdf = torch.cat((past_target_cdf, step), dim=1)

        return torch.cat(outputs, dim=1)  # [B, prediction_length, C]

    def forecast(self, batch_data, num_samples=None):
        forecasts = []
        past_target_cdf = batch_data.past_target_cdf

        for k in range(self.prediction_length):
            current_batch_data = ProbTSBatchData({
                'target_dimension_indicator': batch_data.target_dimension_indicator,
                'past_target_cdf': past_target_cdf,
                'future_time_feat': batch_data.future_time_feat[:, k: k + 1:, ...]
            }, device=batch_data.device)

            # Use the most recent context window to predict a single next step.
            inputs = self.get_inputs(current_batch_data, 'encode')  # [B, L_ctx, input_size]
            inputs = self.enc_linear(inputs)                        # [B, L_ctx, C]
            outputs = self.encoder(inputs)                          # [B, 1, C]
            forecasts.append(outputs)

            # Feed the prediction back in as the newest observation.
            past_target_cdf = torch.cat(
                (past_target_cdf, outputs), dim=1
            )

        forecasts = torch.cat(forecasts, dim=1).reshape(
            -1, self.prediction_length, self.target_dim)
        return forecasts.unsqueeze(1)