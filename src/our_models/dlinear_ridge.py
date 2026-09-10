import numpy as np
from sklearn.linear_model import RidgeCV
from probts.model.nn.arch.decomp import series_decomp


class DLinearRidge:
    def __init__(
        self,
        kernel_size: int,
        individual: bool,
        alphas=None,
        **kwargs
    ):
        #super().__init__(**kwargs)
        context_length = kwargs['context_length']
        prediction_length = kwargs['prediction_length']
        target_dim = kwargs['target_dim']

        self.kernel_size = kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual

        self.alphas = (
            np.logspace(-6, 3, 13) if alphas is None else np.asarray(alphas)
        )

        # L and P are the per-channel time lengths. Use the model's configured
        # lengths; these define how to un-flatten X and y.
        # If lags are used then L = context_length * num_of_lags otherwise it is just context_length
        use_lags = kwargs['use_lags']
        if use_lags:
            lags_list = kwargs['lags_list']
            self.L = context_length * len(lags_list) # time length expands by lags
            self.C = target_dim  # channels stay unchanged
        else:
            self.L = context_length
            self.C = target_dim
        self.P = prediction_length

        if self.individual:
            self.ridge_seasonal = [
                RidgeCV(alphas=self.alphas) for _ in range(self.C)
            ]
            self.ridge_trend = [
                RidgeCV(alphas=self.alphas) for _ in range(self.C)
            ]
        else:
            self.ridge_seasonal = RidgeCV(alphas=self.alphas)
            self.ridge_trend = RidgeCV(alphas=self.alphas)

        self._fitted = False

    # ── reshape helpers ──────────────────────────────────────────────
    def _unflatten_x(self, X):
        """[B, L*C] (L-major, C-minor) -> [B, C, L]."""
        B = X.shape[0]
        return X.reshape(B, self.L, self.C).transpose(0, 2, 1)

    def _unflatten_y(self, y):
        """[B, P*C] (P-major, C-minor) -> [B, C, P]."""
        B = y.shape[0]
        return y.reshape(B, self.P, self.C).transpose(0, 2, 1)

    def _decompose_np(self, x_bcl):
        """x_bcl: [B, C, L] numpy -> (seasonal, trend) each [B, C, L] numpy.

        series_decomp expects a torch tensor in [B, L, C] (the orientation it
        sees inside the original encoder, before the permute). So we transpose
        back to [B, L, C], run it, and transpose the outputs to [B, C, L]."""
        import torch
        x_blc = torch.tensor(
            x_bcl.transpose(0, 2, 1), dtype=torch.float32
        )  # [B, L, C]
        seasonal_init, trend_init = self.decompsition(x_blc)
        seasonal = seasonal_init.permute(0, 2, 1).detach().cpu().numpy()
        trend = trend_init.permute(0, 2, 1).detach().cpu().numpy()
        return seasonal, trend  # [B, C, L]

    # ── analytic fit ─────────────────────────────────────────────────
    def fit(self, X, y):
        """
        X: [B, L*C]   y: [B, P*C]   both flattened L/P-major, C-minor,
        already scaled. Solves each RidgeCV in closed form on all windows.
        """
        x_bcl = self._unflatten_x(X)          # [B, C, L]
        y_bcp = self._unflatten_y(y)          # [B, C, P]
        seasonal, trend = self._decompose_np(x_bcl)   # [B, C, L] each

        B, C, L = seasonal.shape
        P = y_bcp.shape[2]

        if self.individual:
            for i in range(C):
                self.ridge_seasonal[i].fit(seasonal[:, i, :], y_bcp[:, i, :])
                self.ridge_trend[i].fit(trend[:, i, :], y_bcp[:, i, :])
        else:
            # Shared map across channels: fold C into the sample axis, exactly
            # as one nn.Linear(L -> P) applied identically to every channel.
            seasonal_flat = seasonal.reshape(B * C, L)
            trend_flat = trend.reshape(B * C, L)
            y_flat = y_bcp.reshape(B * C, P)
            self.ridge_seasonal.fit(seasonal_flat, y_flat)
            self.ridge_trend.fit(trend_flat, y_flat)
        #print(f'Seasonal fit alpha = {self.ridge_seasonal.alpha_}')
        #print(f'Trend fit alpha = {self.ridge_trend.alpha_}')
        self._fitted = True
        return self

    # ── prediction ───────────────────────────────────────────────────
    def predict(self, X):
        """X: [B, L*C] -> predictions [B, P*C] in the same flattened layout.

        Pure numpy in / numpy out. The surrounding Forecaster builds X from the
        batch (get_inputs + reshape) and reshapes this output back to
        [B, P, F], so nothing torch- or batch-related belongs here."""

        assert X.shape[1] == self.L * self.C, (
            f"expected L*C={self.L * self.C}, got {X.shape[1]}"
        )

        if not self._fitted:
            import traceback
            traceback.print_stack()
            raise RuntimeError(
                "DLinearRidge.predict called before fit — see stack above"
            )

        x_bcl = self._unflatten_x(X)  # [B, C, L]
        seasonal, trend = self._decompose_np(x_bcl)  # [B, C, L]
        B, C, L = seasonal.shape

        if self.individual:
            P = self.P
            seasonal_out = np.zeros((B, C, P), dtype=np.float64)
            trend_out = np.zeros((B, C, P), dtype=np.float64)
            for i in range(C):
                seasonal_out[:, i, :] = self.ridge_seasonal[i].predict(
                    seasonal[:, i, :]
                )
                trend_out[:, i, :] = self.ridge_trend[i].predict(trend[:, i, :])
        else:
            P = self.P
            seasonal_out = self.ridge_seasonal.predict(
                seasonal.reshape(B * C, L)
            ).reshape(B, C, P)
            trend_out = self.ridge_trend.predict(
                trend.reshape(B * C, L)
            ).reshape(B, C, P)

        out_bcp = seasonal_out + trend_out  # [B, C, P]
        # Back to the flattened layout [B, P*C] (P-major, C-minor)
        return out_bcp.transpose(0, 2, 1).reshape(B, P * C)