import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional
import os
import pickle

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

from probts.model.forecaster import Forecaster


class TsfreshBayesianRidge(Forecaster):
    """
    Forecaster that extracts tsfresh features from the context window
    and feeds the selected features into a per-horizon BayesianRidge regressor.

    Training is non-gradient: the loss() calls simply accumulate (context, future)
    pairs, and the sklearn models are fitted lazily on the first call to forecast().
    """

    def __init__(
        self,
        n_samples: int = 100,
        fdr_level: float = 0.05,
        use_tsfresh_selection: bool = False,
        sklearn_state_path: str = './sklearn_state',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.fdr_level = fdr_level
        self.no_training = False  # we need the training loop to accumulate data
        self.use_tsfresh_selection = use_tsfresh_selection
        self._sklearn_state_path = sklearn_state_path

        # Dummy parameter so Adam doesn't crash on an empty parameter list
        self._dummy = nn.Parameter(torch.zeros(1))

        # Sklearn state — populated by _fit()
        self.regressors: Optional[list] = None        # [prediction_length][target_dim]
        self.selected_features: Optional[list] = None # same shape
        self.scaler = StandardScaler()
        self._is_fitted = False

        # Training data accumulators
        self._train_contexts: list = []
        self._train_futures: list = []
        print(self._sklearn_state_path)
    # ------------------------------------------------------------------
    # tsfresh helpers
    # ------------------------------------------------------------------

    def _to_tsfresh_df(self, X: np.ndarray) -> pd.DataFrame:
        """Convert [N, T, D] array to the long-format tsfresh expects."""
        N, T, D = X.shape
        rows = []
        for i in range(N):
            for t in range(T):
                row = {"id": i, "time": t}
                for d in range(D):
                    row[f"value_{d}"] = X[i, t, d]
                rows.append(row)
        return pd.DataFrame(rows)

    def _extract_tsfresh(self, X: np.ndarray) -> pd.DataFrame:
        """Extract and impute tsfresh features from an [N, T, D] array."""
        df = self._to_tsfresh_df(X)
        features = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=MinimalFCParameters(),
            impute_function=impute,
            n_jobs=0,
            disable_progressbar=True,
        )
        impute(features)
        return features

    def _extract(self, X: np.ndarray) -> pd.DataFrame:
        """
        Manually extract simple features from [N, T, D] array.
        Avoids tsfresh multiprocessing and memory issues.
        """
        N, T, D = X.shape
        feature_dict = {}

        for d in range(D):
            Xd = X[:, :, d]  # [N, T]
            feature_dict[f'mean_{d}'] = Xd.mean(axis=1)
            #feature_dict[f'std_{d}'] = Xd.std(axis=1)
            feature_dict[f'min_{d}'] = Xd.min(axis=1)
            feature_dict[f'max_{d}'] = Xd.max(axis=1)
            #feature_dict[f'trend_{d}'] = Xd[:, -1] - Xd[:, 0]
            #feature_dict[f'abs_energy_{d}'] = (Xd ** 2).sum(axis=1)

        return pd.DataFrame(feature_dict)

    def _save_sklearn_state(self):
        state = {
            'regressors': self.regressors,
            'selected_features': self.selected_features,
            'scaler': self.scaler,
            'train_means': self._train_means,
            'train_stds': self._train_stds,
        }
        os.makedirs(self._sklearn_state_path, exist_ok=True)
        path = os.path.join(self._sklearn_state_path, 'sklearn_state.pkl')
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved sklearn state to {path}")

    def _load_sklearn_state(self):
        path = os.path.join(self._sklearn_state_path, 'sklearn_state.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(f"No sklearn state found at {path}")
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.regressors = state['regressors']
        self.selected_features = state['selected_features']
        self.scaler = state['scaler']
        self._train_means = state['train_means']
        self._train_stds = state['train_stds']
        self._is_fitted = True
        print(f"Loaded sklearn state from {path}")

    # ------------------------------------------------------------------
    # Fit sklearn models from accumulated training data
    # ------------------------------------------------------------------

    def _fit(self):
        print(f"1. _fit called, n_contexts={len(self._train_contexts)}")
        X_all = np.concatenate(self._train_contexts, axis=0)
        print(f"2. X_all shape: {X_all.shape}, size in MB: {X_all.nbytes / 1e6:.1f}")
        y_all = np.concatenate(self._train_futures, axis=0)
        print(f"3. y_all shape: {y_all.shape}, size in MB: {y_all.nbytes / 1e6:.1f}")
        N, T_pred, D = y_all.shape

        print(f"4. [TsfreshBayesianRidge] Extracting features from {N} training windows...")
        features_df = self._extract(X_all)
        print(features_df.shape)
        print('5. Features extracted')

        # Fit a single global scaler on raw features; select_features still
        # uses the raw (unscaled) values for its statistical relevance tests
        X_scaled = self.scaler.fit_transform(features_df.values)
        features_scaled_df = pd.DataFrame(
            X_scaled, columns=features_df.columns, index=features_df.index
        )
        print('6. Features scaled')

        self.regressors = []
        self.selected_features = []

        for d in range(D):
            y_d = y_all[:, :, d]  # [N, T_pred] — predict all horizons at once

            # feature selection: use first horizon as proxy
            if not self.use_tsfresh_selection:
                sel_feats = features_df.columns.tolist()
            else:
                y_proxy = pd.Series(y_d[:, 0], name="target")
                # skip feature selection if target has no variance
                if y_proxy.nunique() <= 1:
                    sel_feats = features_df.columns.tolist()
                else:
                    sel_df = select_features(
                        features_df, y_proxy, fdr_level=self.fdr_level, n_jobs=0
                    )
                    sel_feats = (
                        sel_df.columns.tolist()
                        if not sel_df.empty
                        else features_df.columns.tolist()  # fallback: keep all
                    )

            try:
                print(np.array(features_scaled_df[sel_feats].values).shape)
                reg = MultiOutputRegressor(BayesianRidge())
                reg.fit(features_scaled_df[sel_feats].values, y_d)
                self.regressors.append(reg)
                self.selected_features.append(sel_feats)
            except Exception as e:
                # SVD or other failure — store None and fall back to mean at forecast time
                print(f"Warning: fitting failed for d={d}: {e}. Falling back to mean.")
                self.regressors.append(None)
                self.selected_features.append(None)

            print(f'   7 {d}/{D} Fitted {len(sel_feats)} features')

        self._train_means = y_all.mean(axis=0)  # [T_pred, D]
        self._train_stds = y_all.std(axis=0)  # [T_pred, D]
        self._is_fitted = True
        self._train_contexts.clear()
        self._train_futures.clear()
        # Save sklearn state to disk alongside the checkpoint
        self._save_sklearn_state()
        print("8. [TsfreshBayesianRidge] Fitting complete.")

    # ------------------------------------------------------------------
    # ProbTS interface
    # ------------------------------------------------------------------

    def forward(self, inputs):
        # ProbTS calls loss() and forecast() directly; forward() is not used.
        raise NotImplementedError(
            "TsfreshBayesianRidge uses loss() and forecast(), not forward()."
        )

    def loss(self, batch_data):
        """
        Accumulate (context, future) pairs from each training batch.
        Returns a constant zero loss so the Lightning training loop runs normally.
        """

        if not self._is_fitted:
            past = self.get_inputs(batch_data, 'encode').detach().cpu().numpy()
            print(past.shape)
            future = batch_data.future_target_cdf.detach().cpu().numpy()
            print(future.shape)
            self._train_contexts.append(past)
            self._train_futures.append(future)
        return torch.tensor(0.0, requires_grad=True)

    def forecast(self, batch_data, num_samples=None):
        """
        Fit on first call (lazy), then generate probabilistic forecasts via
        posterior predictive sampling from BayesianRidge's Gaussian output.

        Returns: Tensor [batch_size, num_samples, prediction_length, var_num]
        """
        if not self._is_fitted:
            self._fit()
        else:
            # model was loaded from checkpoint — restore sklearn state from disk
            self._load_sklearn_state()

        n_samples = num_samples if num_samples is not None else self.n_samples
        past = self.get_inputs(batch_data, 'encode').cpu().numpy()
        B, _, D = past.shape
        T_pred = self.prediction_length

        features_df = self._extract(past)
        X_scaled = self.scaler.transform(features_df.values)
        features_scaled_df = pd.DataFrame(
            X_scaled, columns=features_df.columns, index=features_df.index
        )

        means = np.zeros((B, T_pred, D))
        stds = np.zeros((B, T_pred, D))

        for d in range(D):
            reg = self.regressors[d]
            if reg is None:
                # fallback: use training mean and std
                means[:, :, d] = self._train_means[:, d]
                stds[:, :, d] = self._train_stds[:, d]
            else:
                sel_feats = self.selected_features[d]
                X_sel = features_scaled_df[sel_feats].values
                # get mean and std for each horizon from each internal estimator
                for h, est in enumerate(reg.estimators_):
                    y_mean, y_std = est.predict(X_sel, return_std=True)
                    means[:, h, d] = y_mean
                    stds[:, h, d] = y_std

        # Draw posterior predictive samples: N(mu, sigma)
        eps = np.random.randn(B, n_samples, T_pred, D)
        samples = means[:, None, :, :] + stds[:, None, :, :] * eps

        device = batch_data.past_target_cdf.device
        return torch.tensor(samples, dtype=torch.float32, device=device)