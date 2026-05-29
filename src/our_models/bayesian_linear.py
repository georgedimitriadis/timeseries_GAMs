# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from probts.model.forecaster import Forecaster


class BayesianLinearForecaster(Forecaster):
    def __init__(
            self,
            individual: bool = True,
            kl_weight: float= 1.0,
            init_sigma: float = -1,
            n_train_batches: int = 100,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.individual = individual
        self.kl_weight = kl_weight
        self.n_train_batches = n_train_batches

        if self.individual:
            self.linear = nn.ModuleList()
            self.weight_mu = nn.ParameterList()
            self.weight_logsigma = nn.ParameterList()
            for i in range(self.input_size):
                self.linear.append(nn.Linear(self.context_length, self.prediction_length))
                self.weight_mu.append(nn.Parameter(torch.zeros(self.prediction_length, self.context_length)))
                self.weight_logsigma.append(nn.Parameter(torch.full((self.prediction_length, self.context_length), init_sigma)))
        else:
            self.linear = nn.Linear(self.context_length, self.prediction_length)
            self.weight_mu = nn.Parameter(torch.zeros(self.prediction_length, self.context_length))
            self.weight_logsigma = nn.Parameter(torch.full((self.prediction_length, self.context_length), init_sigma))

        self.out_linear = nn.Linear(self.input_size, self.target_dim)
        self.loss_fn = nn.MSELoss(reduction='none')


    def forward(self, x):
        if self.individual:
            outputs = torch.zeros(x.size(0), self.prediction_length, x.size(2),
                                  dtype=x.dtype, device=x.device)
            for i in range(self.input_size):
                sigma = torch.exp(self.weight_logsigma[i])  # [T_out, T_in]
                w = self.weight_mu[i] + sigma * torch.randn_like(sigma)
                outputs[:, :, i] = F.linear(x[:, :, i], w, self.linear[i].bias)
        else:
            sigma = torch.exp(self.weight_logsigma)  # [T_out, T_in]
            w = self.weight_mu + sigma * torch.randn_like(sigma)
            outputs = F.linear(x.permute(0, 2, 1), w, self.linear.bias).permute(0, 2, 1)

        outputs = self.out_linear(outputs)
        return outputs

    def forecast(self, batch_data=None, num_samples=100):
        inputs = self.get_inputs(batch_data, 'encode')

        samples = []
        for _ in range(num_samples):
            samples.append(self(inputs))  # each call samples a new w

        forecasts = torch.stack(samples, dim=1)  # [B, num_samples, T_out, F]
        return forecasts

    def kl_loss(self):
        if self.individual:
            kl = sum(
                -0.5 * torch.sum(
                    1 + 2 * logsigma
                    - mu ** 2
                    - torch.exp(2 * logsigma)
                )
                for mu, logsigma in zip(self.weight_mu, self.weight_logsigma)
            )
        else:
            kl = -0.5 * torch.sum(
                1 + 2 * self.weight_logsigma
                - self.weight_mu ** 2
                - torch.exp(2 * self.weight_logsigma)
            )
        return kl

    def loss(self, batch_data):
        inputs = self.get_inputs(batch_data, 'encode')
        outputs = self(inputs)

        mse = self.loss_fn(batch_data.future_target_cdf, outputs)
        mse = self.get_weighted_loss(batch_data, mse).mean()
        kl = self.kl_loss() / self.n_train_batches

        return mse + self.kl_weight * kl
