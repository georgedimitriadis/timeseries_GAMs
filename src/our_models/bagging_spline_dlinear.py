

# ---------------------------------------------------------------------------------
# Portions of this file are derived from LTSF-Linear
# - Source: https://github.com/cure-lab/LTSF-Linear
# - Paper: Are Transformers Effective for Time Series Forecasting?
# - License: Apache-2.0
#
# Random-subspace bagged DLinear modification with learned decay/scale spline
# accumulator:
# - Multiple independent DLinear members.
# - Each member has a fixed random mask over the input context/features.
# - Each member may include a channel-specific learned accumulator branch:
#       l_t[d] = decay[d] * l_{t-1}[d] + scale[d] * spline_d(x_t[d])
# - forecast() returns member forecasts as an empirical distribution.
# ---------------------------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.forecaster import Forecaster
from probts.model.nn.arch.decomp import series_decomp


class LearnedDecaySplineAccumulator(nn.Module):
    """
    Channel-specific spline accumulator with learned decay and learned increment
    scale.

    Recurrence per channel d:

        l_t[d] = decay[d] * l_{t-1}[d] + scale[d] * spline_d(x_t[d])

    where:
        decay[d] = sigmoid(decay_logit[d])       in (0, 1)
        scale[d] = softplus(scale_unconstrained[d]) > 0

    The pure additive accumulator is the limiting/special case:

        decay ~= 1, scale ~= 1

    After scanning the context, l_T[d] is projected to the forecast horizon:

        accumulator_output[:, h, d] = l_T[d] * horizon_weight[d, h]
                                     + horizon_bias[d, h]

    Input:
        [B, context_length, target_dim]

    Output:
        [B, prediction_length, target_dim]

    This branch is channel-local: there is no cross-channel mixing in the
    recurrence.
    """

    def __init__(
        self,
        target_dim: int,
        prediction_length: int,
        n_knots: int = 16,
        knot_min: float = -2.0,
        knot_max: float = 2.0,
        init_decay: float = 0.98,
        init_increment_scale: float = 1.0,
        init_spline_weight_std: float = 0.02,
        init_horizon_weight_std: float = 0.02,
        normalize_by_length: bool = False,
    ):
        super().__init__()

        if n_knots < 2:
            raise ValueError("n_knots must be at least 2.")

        if knot_max <= knot_min:
            raise ValueError("knot_max must be greater than knot_min.")

        if not 0.0 < init_decay < 1.0:
            raise ValueError("init_decay must be in (0, 1).")

        if init_increment_scale <= 0.0:
            raise ValueError("init_increment_scale must be positive.")

        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.n_knots = n_knots
        self.knot_min = knot_min
        self.knot_max = knot_max
        self.normalize_by_length = normalize_by_length

        knots = torch.linspace(knot_min, knot_max, n_knots)
        self.register_buffer("knots", knots, persistent=True)

        # Channel-specific learned decay in (0, 1).
        decay_logit = torch.logit(torch.tensor(float(init_decay)))
        self.decay_logit = nn.Parameter(torch.full((target_dim,), decay_logit))

        # Channel-specific positive learned increment scale.
        # inverse softplus: x = log(exp(y) - 1)
        inv_softplus = math.log(math.expm1(float(init_increment_scale)))
        self.increment_scale_unconstrained = nn.Parameter(
            torch.full((target_dim,), inv_softplus)
        )

        # spline_d(x) = sum_k basis_k(x) * spline_weight[d, k] + spline_bias[d]
        self.spline_weight = nn.Parameter(torch.empty(target_dim, n_knots))
        self.spline_bias = nn.Parameter(torch.zeros(target_dim))
        nn.init.normal_(self.spline_weight, mean=0.0, std=init_spline_weight_std)

        # Channel-specific projection from final accumulator state to horizon.
        self.horizon_weight = nn.Parameter(torch.empty(target_dim, prediction_length))
        self.horizon_bias = nn.Parameter(torch.zeros(target_dim, prediction_length))
        nn.init.normal_(self.horizon_weight, mean=0.0, std=init_horizon_weight_std)

    def decay(self):
        return torch.sigmoid(self.decay_logit)

    def increment_scale(self):
        return F.softplus(self.increment_scale_unconstrained)

    def _basis(self, x: torch.Tensor):
        """
        Triangular first-order spline basis.

        x:
            [B, D]

        returns:
            [B, D, K]
        """
        knots = self.knots.to(device=x.device, dtype=x.dtype)
        spacing = (knots[-1] - knots[0]) / max(self.n_knots - 1, 1)
        spacing = spacing.clamp_min(torch.finfo(x.dtype).eps)

        distance = torch.abs(x.unsqueeze(-1) - knots.view(1, 1, self.n_knots))
        basis = F.relu(1.0 - distance / spacing)
        return basis

    def spline_increment(self, x_t: torch.Tensor):
        """
        Evaluate channel-specific scalar spline increments.

        x_t:
            [B, D]

        returns:
            [B, D]
        """
        basis = self._basis(x_t)  # [B, D, K]
        weight = self.spline_weight.to(device=x_t.device, dtype=x_t.dtype)
        bias = self.spline_bias.to(device=x_t.device, dtype=x_t.dtype)

        increment = torch.einsum("bdk,dk->bd", basis, weight) + bias.view(1, -1)
        return increment

    def forward(self, inputs: torch.Tensor):
        """
        inputs:
            [B, L, D]

        returns:
            [B, H, D]
        """
        B, L, D = inputs.shape

        if D != self.target_dim:
            raise ValueError(
                f"Expected target_dim={self.target_dim}, got input dimension {D}."
            )

        accumulator = torch.zeros(B, D, device=inputs.device, dtype=inputs.dtype)

        decay = self.decay().to(device=inputs.device, dtype=inputs.dtype).view(1, D)
        scale = self.increment_scale().to(device=inputs.device, dtype=inputs.dtype).view(1, D)

        # Learned leaky/additive recurrence:
        #   l_t = decay * l_{t-1} + scale * spline(x_t)
        for t in range(L):
            accumulator = decay * accumulator + scale * self.spline_increment(inputs[:, t, :])

        if self.normalize_by_length:
            accumulator = accumulator / max(L, 1)

        horizon_weight = self.horizon_weight.to(device=inputs.device, dtype=inputs.dtype)
        horizon_bias = self.horizon_bias.to(device=inputs.device, dtype=inputs.dtype)

        # [B, D, H]
        out = accumulator.unsqueeze(-1) * horizon_weight.unsqueeze(0)
        out = out + horizon_bias.unsqueeze(0)

        # [B, H, D]
        return out.permute(0, 2, 1)


class _MaskedDLinearMember(nn.Module):
    """
    One independent DLinear member with a fixed random input mask and optional
    channel-specific learned decay/scale spline accumulator branch.

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
        context_keep_prob: float = 0.8,
        feature_keep_prob: float = 1.0,
        mask_scale: bool = True,
        mask_mode: str = "time_feature",
        min_context_keep: int = 1,
        min_feature_keep: int = 1,
        use_accumulator: bool = True,
        accumulator_n_knots: int = 16,
        accumulator_knot_min: float = -2.0,
        accumulator_knot_max: float = 2.0,
        accumulator_init_decay: float = 0.98,
        accumulator_init_increment_scale: float = 1.0,
        accumulator_init_spline_weight_std: float = 0.02,
        accumulator_init_horizon_weight_std: float = 0.02,
        accumulator_normalize_by_length: bool = False,
    ):
        super().__init__()

        if not 0.0 < context_keep_prob <= 1.0:
            raise ValueError("context_keep_prob must be in (0, 1].")

        if not 0.0 < feature_keep_prob <= 1.0:
            raise ValueError("feature_keep_prob must be in (0, 1].")

        if mask_mode not in {"time", "feature", "time_feature"}:
            raise ValueError(
                "mask_mode must be one of {'time', 'feature', 'time_feature'}."
            )

        self.input_size = input_size
        self.target_dim = target_dim
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.individual = individual
        self.context_keep_prob = context_keep_prob
        self.feature_keep_prob = feature_keep_prob
        self.mask_scale = mask_scale
        self.mask_mode = mask_mode
        self.use_accumulator = use_accumulator

        if input_size != target_dim:
            self.enc_linear = nn.Linear(
                in_features=input_size,
                out_features=target_dim,
            )
        else:
            self.enc_linear = nn.Identity()

        # Keep the original misspelling for compatibility with the source style.
        self.decompsition = series_decomp(kernel_size)

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for _ in range(target_dim):
                self.Linear_Seasonal.append(
                    nn.Linear(context_length, prediction_length)
                )
                self.Linear_Trend.append(
                    nn.Linear(context_length, prediction_length)
                )
        else:
            self.Linear_Seasonal = nn.Linear(context_length, prediction_length)
            self.Linear_Trend = nn.Linear(context_length, prediction_length)

        if use_accumulator:
            self.Accumulator = LearnedDecaySplineAccumulator(
                target_dim=target_dim,
                prediction_length=prediction_length,
                n_knots=accumulator_n_knots,
                knot_min=accumulator_knot_min,
                knot_max=accumulator_knot_max,
                init_decay=accumulator_init_decay,
                init_increment_scale=accumulator_init_increment_scale,
                init_spline_weight_std=accumulator_init_spline_weight_std,
                init_horizon_weight_std=accumulator_init_horizon_weight_std,
                normalize_by_length=accumulator_normalize_by_length,
            )
        else:
            self.Accumulator = None

        input_mask = self._make_fixed_input_mask(
            context_length=context_length,
            input_size=input_size,
            context_keep_prob=context_keep_prob,
            feature_keep_prob=feature_keep_prob,
            mask_mode=mask_mode,
            mask_scale=mask_scale,
            min_context_keep=min_context_keep,
            min_feature_keep=min_feature_keep,
        )

        # Shape: [1, L, input_size], broadcastable across batch.
        self.register_buffer("input_mask", input_mask, persistent=True)

    @staticmethod
    def _ensure_at_least_k_kept(mask: torch.Tensor, k: int):
        """
        Ensure a 1D Bernoulli mask has at least k entries equal to 1.
        """
        if k <= 0:
            return mask

        n = mask.numel()
        k = min(k, n)
        current = int(mask.sum().item())

        if current >= k:
            return mask

        zero_idx = torch.nonzero(mask == 0.0, as_tuple=False).flatten()

        if zero_idx.numel() == 0:
            return mask

        need = k - current
        chosen = zero_idx[torch.randperm(zero_idx.numel())[:need]]
        mask[chosen] = 1.0

        return mask

    @classmethod
    def _make_fixed_input_mask(
        cls,
        context_length: int,
        input_size: int,
        context_keep_prob: float,
        feature_keep_prob: float,
        mask_mode: str,
        mask_scale: bool,
        min_context_keep: int,
        min_feature_keep: int,
    ):
        """
        Builds a fixed mask with shape [1, context_length, input_size].
        """
        if mask_mode in {"time", "time_feature"}:
            time_mask = torch.bernoulli(
                torch.full((context_length,), context_keep_prob)
            )
            time_mask = cls._ensure_at_least_k_kept(time_mask, min_context_keep)
        else:
            time_mask = torch.ones(context_length)

        if mask_mode in {"feature", "time_feature"}:
            feature_mask = torch.bernoulli(
                torch.full((input_size,), feature_keep_prob)
            )
            feature_mask = cls._ensure_at_least_k_kept(feature_mask, min_feature_keep)
        else:
            feature_mask = torch.ones(input_size)

        mask = time_mask.view(context_length, 1) * feature_mask.view(1, input_size)

        if mask.sum() == 0:
            mask[:] = 1.0

        if mask_scale:
            if mask_mode == "time":
                expected_keep = context_keep_prob
            elif mask_mode == "feature":
                expected_keep = feature_keep_prob
            else:
                expected_keep = context_keep_prob * feature_keep_prob

            mask = mask / max(expected_keep, 1e-6)

        return mask.view(1, context_length, input_size)

    def forward(self, inputs):
        """
        inputs:
            [B, context_length, input_size]

        returns:
            [B, prediction_length, target_dim]
        """
        inputs = inputs * self.input_mask
        inputs = self.enc_linear(inputs)  # [B, L, target_dim]

        seasonal_init, trend_init = self.decompsition(inputs)

        # [B, C, L]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            B = seasonal_init.size(0)
            C = seasonal_init.size(1)

            seasonal_output = torch.zeros(
                B,
                C,
                self.prediction_length,
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )

            trend_output = torch.zeros(
                B,
                C,
                self.prediction_length,
                dtype=trend_init.dtype,
                device=trend_init.device,
            )

            for i in range(self.target_dim):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :]
                )
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        outputs = seasonal_output + trend_output  # [B, C, H]
        outputs = outputs.permute(0, 2, 1)        # [B, H, C]

        if self.Accumulator is not None:
            accumulator_output = self.Accumulator(inputs)  # [B, H, C]
            outputs = outputs + accumulator_output

        return outputs


class BaggingSplineDlinear(Forecaster):
    """
    Bagged/random-subspace DLinear with optional channel-specific learned
    decay/scale spline accumulator.

    This keeps the original DLinear API style:

        loss(batch_data)
        forecast(batch_data, num_samples=None)

    Internally, it trains several independent DLinear members. Each member sees
    a fixed masked version of the input context/features and optionally adds a
    channel-specific accumulator branch:

        output = seasonal + trend + accumulator

    where accumulator is generated by:

        l_t[d] = decay[d] * l_{t-1}[d] + scale[d] * spline_d(x_t[d])

    forecast() returns:
        [B, S, prediction_length, target_dim]
    """

    def __init__(
        self,
        kernel_size: int,
        individual: bool,
        n_members: int = 20,
        context_keep_prob: float = 0.8,
        feature_keep_prob: float = 1.0,
        mask_mode: str = "time_feature",
        mask_scale: bool = True,
        min_context_keep: int = 1,
        min_feature_keep: int = 1,
        num_parallel_samples: int = 100,
        aggregate_loss: str = "mean",
        use_accumulator: bool = False, # True uses splines
        accumulator_n_knots: int = 16,
        accumulator_knot_min: float = -2.0,
        accumulator_knot_max: float = 2.0,
        accumulator_init_decay: float = 0.98,
        accumulator_init_increment_scale: float = 1.0,
        accumulator_init_spline_weight_std: float = 0.02,
        accumulator_init_horizon_weight_std: float = 0.02,
        accumulator_normalize_by_length: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if n_members <= 0:
            raise ValueError("n_members must be positive.")

        if aggregate_loss not in {"mean", "sum"}:
            raise ValueError("aggregate_loss must be either 'mean' or 'sum'.")

        self.kernel_size = kernel_size
        self.individual = individual
        self.n_members = n_members
        self.context_keep_prob = context_keep_prob
        self.feature_keep_prob = feature_keep_prob
        self.mask_mode = mask_mode
        self.mask_scale = mask_scale
        self.min_context_keep = min_context_keep
        self.min_feature_keep = min_feature_keep
        self.num_parallel_samples = num_parallel_samples
        self.aggregate_loss = aggregate_loss
        self.use_accumulator = use_accumulator
        self.accumulator_n_knots = accumulator_n_knots
        self.accumulator_knot_min = accumulator_knot_min
        self.accumulator_knot_max = accumulator_knot_max
        self.accumulator_init_decay = accumulator_init_decay
        self.accumulator_init_increment_scale = accumulator_init_increment_scale
        self.accumulator_init_spline_weight_std = accumulator_init_spline_weight_std
        self.accumulator_init_horizon_weight_std = accumulator_init_horizon_weight_std
        self.accumulator_normalize_by_length = accumulator_normalize_by_length

        self.members = nn.ModuleList([
            _MaskedDLinearMember(
                input_size=self.input_size,
                target_dim=self.target_dim,
                context_length=self.context_length,
                prediction_length=self.prediction_length,
                kernel_size=kernel_size,
                individual=individual,
                context_keep_prob=context_keep_prob,
                feature_keep_prob=feature_keep_prob,
                mask_scale=mask_scale,
                mask_mode=mask_mode,
                min_context_keep=min_context_keep,
                min_feature_keep=min_feature_keep,
                use_accumulator=use_accumulator,
                accumulator_n_knots=accumulator_n_knots,
                accumulator_knot_min=accumulator_knot_min,
                accumulator_knot_max=accumulator_knot_max,
                accumulator_init_decay=accumulator_init_decay,
                accumulator_init_increment_scale=accumulator_init_increment_scale,
                accumulator_init_spline_weight_std=accumulator_init_spline_weight_std,
                accumulator_init_horizon_weight_std=accumulator_init_horizon_weight_std,
                accumulator_normalize_by_length=accumulator_normalize_by_length,
            )
            for _ in range(n_members)
        ])

        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(self, batch_data, num_samples=None):
        """
        Returns member/sample forecasts.

        Shape:
            [B, S, prediction_length, target_dim]

        If num_samples is None:
            returns all n_members.

        If num_samples != n_members:
            resamples member forecasts with replacement along the sample axis.
        """
        inputs = self.get_inputs(batch_data, "encode")  # [B, L, input_size]

        member_outputs = []

        for member in self.members:
            member_outputs.append(member(inputs))       # [B, H, D]

        samples = torch.stack(member_outputs, dim=1)    # [B, M, H, D]

        if num_samples is None:
            return samples

        M = samples.size(1)

        if num_samples == M:
            return samples

        idx = torch.randint(
            low=0,
            high=M,
            size=(num_samples,),
            device=samples.device,
        )

        return samples[:, idx, :, :]                    # [B, S, H, D]

    def loss(self, batch_data):
        """
        Average MSE over independent masked DLinear members.

        Because members have disjoint parameters, averaging the scalar losses
        does not introduce gradient coupling between member-specific regressors.
        """
        inputs = self.get_inputs(batch_data, "encode")
        target = batch_data.future_target_cdf           # [B, H, D]

        member_losses = []

        for member in self.members:
            outputs = member(inputs)                    # [B, H, D]

            loss = self.loss_fn(target, outputs)        # [B, H, D]
            loss = self.get_weighted_loss(batch_data, loss)

            member_losses.append(loss.mean())

        member_losses = torch.stack(member_losses)

        if self.aggregate_loss == "sum":
            return member_losses.sum()

        return member_losses.mean()

    def forecast(self, batch_data, num_samples=None):
        """
        API-compatible forecast.

        Returns:
            [B, S, prediction_length, target_dim]
        """
        if num_samples is None:
            num_samples = self.num_parallel_samples

        return self.forward(batch_data, num_samples=num_samples)

    def mean_forecast(self, batch_data):
        """
        Optional helper.

        Returns ensemble mean with sample dimension retained:
            [B, 1, prediction_length, target_dim]
        """
        samples = self.forward(batch_data, num_samples=None)
        return samples.mean(dim=1, keepdim=True)

    def member_forecasts(self, batch_data):
        """
        Optional helper.

        Returns exactly one forecast per member:
            [B, n_members, prediction_length, target_dim]
        """
        return self.forward(batch_data, num_samples=None)
