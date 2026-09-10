
# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# Single-model DLinear with post-decomposition multi-resolution pointwise splines
# and an explicit affine base term.
#
# Changes relative to the bagged / accumulator variant:
# - No ensemble / no multiple members.
# - No masking / random subspaces.
# - No accumulator / no recurrence.
# - Two pointwise spline modules, one for seasonal and one for trend.
# - Each spline is multi-resolution and computes an additive sum over resolutions:
#       f(x) = (a * x + b) + f_{r1}(x) + f_{r2}(x) + ...
#   e.g. resolutions=[2, 4, 8, 16] gives
#       f(x) = (a * x + b) + f_2(x) + f_4(x) + f_8(x) + f_16(x)
# - Splines are applied AFTER series decomposition and BEFORE the DLinear heads.
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.forecaster import Forecaster
from probts.model.nn.arch.decomp import series_decomp


class MultiResolutionSpline(nn.Module):
    """
    Multi-resolution pointwise spline for tensors of shape [B, C, L], with an
    explicit affine base term.

    This module applies a channel-specific transform pointwise, independently at
    every timestep, and sums an affine branch with several spline resolutions:

        f(x) = (a * x + b) + sum_{r in resolutions} f_r(x)

    where each f_r is a first-order (triangular / hat-basis) spline with r knots.

    Important properties:
    - Pointwise in time: no recurrence, no temporal mixing.
    - Shared across all timesteps: the same transform is used at every t.
    - Channel-local: each channel has its own affine and spline parameters.

    Input:
        x: [B, C, L]

    Output:
        y: [B, C, L]
    """

    def __init__(
        self,
        target_dim: int,
        resolutions=(2, 4, 16, 64),
        knot_min: float = -1.0,
        knot_max: float = 1.0,
        init_weight_std: float = 0.02,
        init_affine_weight: float = 1.0,
        init_affine_bias: float = 0.0,
    ):
        super().__init__()

        if target_dim <= 0:
            raise ValueError("target_dim must be positive.")

        if knot_max <= knot_min:
            raise ValueError("knot_max must be greater than knot_min.")

        resolutions = list(resolutions)
        if len(resolutions) == 0:
            raise ValueError("resolutions must contain at least one entry.")
        if any(int(r) != r or r < 2 for r in resolutions):
            raise ValueError("Every spline resolution must be an integer >= 2.")

        self.target_dim = target_dim
        self.resolutions = [int(r) for r in resolutions]
        self.knot_min = knot_min
        self.knot_max = knot_max

        # Explicit affine base term: a * x + b
        self.affine_weight = nn.Parameter(torch.full((target_dim,), float(init_affine_weight)))
        self.affine_bias = nn.Parameter(torch.full((target_dim,), float(init_affine_bias)))

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for r in self.resolutions:
            knots = torch.linspace(knot_min, knot_max, r)
            self.register_buffer(f"knots_{r}", knots, persistent=True)

            weight = nn.Parameter(torch.empty(target_dim, r))
            bias = nn.Parameter(torch.zeros(target_dim))

            nn.init.normal_(weight, mean=0.0, std=init_weight_std)

            self.weights.append(weight)
            self.biases.append(bias)

    @staticmethod
    def _basis(x: torch.Tensor, knots: torch.Tensor):
        """
        First-order triangular spline basis.

        Args:
            x:
                [B, C, L]
            knots:
                [K]

        Returns:
            basis:
                [B, C, L, K]
        """
        k = knots.numel()
        spacing = (knots[-1] - knots[0]) / max(k - 1, 1)
        spacing = spacing.clamp_min(torch.finfo(x.dtype).eps)

        dist = torch.abs(x.unsqueeze(-1) - knots.view(1, 1, 1, k))
        basis = F.relu(1.0 - dist / spacing)
        return basis

    def forward(self, x: torch.Tensor):
        """
        Args:
            x:
                [B, C, L]

        Returns:
            [B, C, L]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input with shape [B, C, L], got {tuple(x.shape)}.")

        _, c, _ = x.shape
        if c != self.target_dim:
            raise ValueError(
                f"Expected channel dimension {self.target_dim}, got {c}."
            )

        dtype = x.dtype
        device = x.device

        # Affine base term: a * x + b
        out = (
            x * self.affine_weight.to(device=device, dtype=dtype).view(1, c, 1)
            + self.affine_bias.to(device=device, dtype=dtype).view(1, c, 1)
        )

        # Add multi-resolution spline corrections.
        for r, weight, bias in zip(self.resolutions, self.weights, self.biases):
            knots = getattr(self, f"knots_{r}").to(device=device, dtype=dtype)
            basis = self._basis(x, knots)  # [B, C, L, r]

            contribution = torch.einsum(
                "bclk,ck->bcl",
                basis,
                weight.to(device=device, dtype=dtype),
            )
            contribution = contribution + bias.to(device=device, dtype=dtype).view(1, c, 1)
            out = out + contribution

        return out


class _SplineDLinearModel(nn.Module):
    """
    Single DLinear model with two post-decomposition multi-resolution splines:
    one for the seasonal component and one for the trend component.

    Pipeline:
        inputs -> optional encoder -> series decomposition
               -> seasonal affine+spline / trend affine+spline
               -> DLinear seasonal / trend heads
               -> sum

    Input:
        [B, context_length, input_size]

    Output:
        [B, prediction_length, target_dim]
    """

    def __init__(
        self,
        input_size: int,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        kernel_size: int,
        individual: bool,
        spline_resolutions=(2, 4, 8, 16),
        spline_knot_min: float = -2.0,
        spline_knot_max: float = 2.0,
        spline_init_weight_std: float = 0.02,
        spline_init_affine_weight: float = 1.0,
        spline_init_affine_bias: float = 0.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.target_dim = target_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.individual = individual

        if input_size != target_dim:
            self.enc_linear = nn.Linear(input_size, target_dim)
        else:
            self.enc_linear = nn.Identity()

        # Keep the original source-style attribute name for compatibility.
        self.decompsition = series_decomp(kernel_size)

        # Two separate post-decomposition affine + multi-resolution spline modules.
        self.Spline_Seasonal = MultiResolutionSpline(
            target_dim=target_dim,
            resolutions=spline_resolutions,
            knot_min=spline_knot_min,
            knot_max=spline_knot_max,
            init_weight_std=spline_init_weight_std,
            init_affine_weight=spline_init_affine_weight,
            init_affine_bias=spline_init_affine_bias,
        )
        self.Spline_Trend = MultiResolutionSpline(
            target_dim=target_dim,
            resolutions=spline_resolutions,
            knot_min=spline_knot_min,
            knot_max=spline_knot_max,
            init_weight_std=spline_init_weight_std,
            init_affine_weight=spline_init_affine_weight,
            init_affine_bias=spline_init_affine_bias,
        )

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for _ in range(target_dim):
                self.Linear_Seasonal.append(nn.Linear(context_length, prediction_length))
                self.Linear_Trend.append(nn.Linear(context_length, prediction_length))
        else:
            self.Linear_Seasonal = nn.Linear(context_length, prediction_length)
            self.Linear_Trend = nn.Linear(context_length, prediction_length)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs:
                [B, context_length, input_size]

        Returns:
            [B, prediction_length, target_dim]
        """
        if inputs.dim() != 3:
            raise ValueError(
                f"Expected input with shape [B, L, input_size], got {tuple(inputs.shape)}."
            )

        inputs = self.enc_linear(inputs)  # [B, L, target_dim]

        seasonal_init, trend_init = self.decompsition(inputs)
        # series_decomp returns [B, L, C]; DLinear heads expect [B, C, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # Apply the two pointwise affine + multi-resolution spline transforms AFTER decomposition.
        seasonal_init = self.Spline_Seasonal(seasonal_init)  # [B, C, L]
        trend_init = self.Spline_Trend(trend_init)           # [B, C, L]

        if self.individual:
            b = seasonal_init.size(0)
            c = seasonal_init.size(1)

            seasonal_output = torch.zeros(
                b,
                c,
                self.prediction_length,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                b,
                c,
                self.prediction_length,
                dtype=trend_init.dtype,
                device=trend_init.device,
            )

            for i in range(self.target_dim):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        outputs = seasonal_output + trend_output  # [B, C, H]
        outputs = outputs.permute(0, 2, 1)        # [B, H, C]
        return outputs


class MultiresSplineDLinear(Forecaster):
    """
    Single-model DLinear with two post-decomposition multi-resolution pointwise
    spline transforms (seasonal + trend), each with an explicit affine term.

    API-compatible methods:
        loss(batch_data)
        forecast(batch_data, num_samples=None)

    The model is deterministic. For compatibility with probabilistic forecasting
    interfaces, forward()/forecast() return a sample dimension of size 1 by
    default: [B, 1, prediction_length, target_dim]. If num_samples > 1, the
    deterministic forecast is repeated along the sample dimension.
    """

    def __init__(
        self,
        kernel_size: int,
        individual: bool,
        num_parallel_samples: int = 100,
        spline_resolutions=(2, 4, 16, 64),
        spline_knot_min: float = -2.0,
        spline_knot_max: float = 2.0,
        spline_init_weight_std: float = 0.02,
        spline_init_affine_weight: float = 1.0,
        spline_init_affine_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel_size = kernel_size
        self.individual = individual
        self.num_parallel_samples = num_parallel_samples
        self.spline_resolutions = list(spline_resolutions)
        self.spline_knot_min = spline_knot_min
        self.spline_knot_max = spline_knot_max
        self.spline_init_weight_std = spline_init_weight_std
        self.spline_init_affine_weight = spline_init_affine_weight
        self.spline_init_affine_bias = spline_init_affine_bias

        self.model = _SplineDLinearModel(
            input_size=self.input_size,
            target_dim=self.target_dim,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            kernel_size=kernel_size,
            individual=individual,
            spline_resolutions=spline_resolutions,
            spline_knot_min=spline_knot_min,
            spline_knot_max=spline_knot_max,
            spline_init_weight_std=spline_init_weight_std,
            spline_init_affine_weight=spline_init_affine_weight,
            spline_init_affine_bias=spline_init_affine_bias,
        )

        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, batch_data, num_samples=None):
        """
        Returns deterministic forecasts with a sample dimension.

        Shape:
            [B, S, prediction_length, target_dim]

        If num_samples is None:
            returns S=1.

        If num_samples is provided and > 1:
            repeats the deterministic forecast along the sample axis.
        """
        inputs = self.get_inputs(batch_data, "encode")
        outputs = self.model(inputs)  # [B, H, D]

        samples = outputs.unsqueeze(1)  # [B, 1, H, D]

        if num_samples is None or num_samples == 1:
            return samples

        return samples.repeat(1, num_samples, 1, 1)

    def loss(self, batch_data):
        target = batch_data.future_target_cdf  # [B, H, D]
        outputs = self.model(self.get_inputs(batch_data, "encode"))

        loss = self.loss_fn(target, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        return loss.mean()

    def forecast(self, batch_data, num_samples=None):
        """
        API-compatible forecast.

        Returns:
            [B, S, prediction_length, target_dim]
        """
        if num_samples is None:
            num_samples = 1
        return self.forward(batch_data, num_samples=num_samples)

    def mean_forecast(self, batch_data):
        """
        Returns deterministic mean forecast with sample dimension retained:
            [B, 1, prediction_length, target_dim]
        """
        return self.forward(batch_data, num_samples=1)
