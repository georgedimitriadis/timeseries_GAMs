# ---------------------------------------------------------------------------------
# VMD Decomposition Forecaster for ProbTS
#
# Architecture:
#   1. VMD (Variational Mode Decomposition) transforms the ENTIRE raw dataset
#      once on the first call, saving results to vmd_cache_path.
#      Subsequent runs load from cache — VMD never runs again for the same data.
#   2. Each batch lookup uses a value fingerprint to find the correct VMD window
#      from the pre-transformed dataset, O(1) per batch item.
#   3. A shared linear layer maps each (b, f, d) mode from T_in → T_in+T_out.
#   4. The last T_out outputs are un-normalised and summed over D → forecast.
#
# Gradient flow:
#   VMD and data loading have NO gradient.
#   Gradients flow through: Linear layer weights → output.
#
# Dependencies:
#   pip install vmdpy
# ---------------------------------------------------------------------------------

import os
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vmdpy import VMD

from probts.model.forecaster import Forecaster

log = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────────────────────────────────
# 3. Main model
# ─────────────────────────────────────────────────────────────────────────────────

class VMDDecompositionForecaster(Forecaster):
    """
    ProbTS forecaster using offline VMD decomposition.

    Args:
        data_root_path (str):
            Root folder containing all dataset subdirectories — the same value
            you pass to DataManager as root_path (e.g. /data/datasets/).
            ProbTS's get_dataset_info resolves the actual file path from
            self.dataset, so you never need to specify it per-model.
        vmd_cache_path (str):
            Where to save (and subsequently load) the VMD-transformed dataset.
            Saved as a numpy .npz file.  Delete to force recomputation.
        custom_data_file (str | None):
            Relative path inside data_root_path for datasets whose name is NOT
            in ProbTS's built-in table (e.g. 'my_project/series.csv').
            Leave as null for any built-in dataset name (etth1, traffic_ltsf…).
        num_decompositions (int):
            Number of VMD modes K.
        vmd_alpha (float):
            VMD bandwidth constraint (100–5000 typical).
        vmd_tau (float):
            Noise tolerance (0 = strict data fidelity).
        vmd_DC (int):
            1 = include DC/trend mode, 0 = exclude.
        vmd_init (int):
            Centre-frequency init: 1 = uniform, 2 = random.
        vmd_tol (float):
            Convergence tolerance.
    """

    def __init__(
        self,
        data_root_path: str,
        vmd_cache_path: str,
        custom_data_file: Optional[str] = None,
        num_decompositions: int = 5,
        vmd_alpha: float = 2000.0,
        vmd_tau: float = 0.0,
        vmd_DC: int = 0,
        vmd_init: int = 1,
        vmd_tol: float = 1e-7,
        dropout: float = 0.1,
        l2_lambda: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.data_root_path  = data_root_path
        self.vmd_cache_path  = vmd_cache_path
        self.custom_data_file = custom_data_file
        self.D               = num_decompositions
        self.vmd_alpha       = vmd_alpha
        self.vmd_tau         = vmd_tau
        self.vmd_DC          = vmd_DC
        self.vmd_init        = vmd_init
        self.vmd_tol         = vmd_tol

        # Two-layer structure mirroring LinearForecaster (individual=False):
        #   self.linear     : time projection   T_in  → T_out   (per F*D channel)
        #   self.out_linear : feature projection F*D   → F       (per time step)
        T_in  = self.max_context_length
        T_out = self.max_prediction_length
        self.linear     = nn.Linear(T_in, T_out)
        #self.out_linear = nn.Linear(self.target_dim * self.D, self.target_dim)
        self.out_linear = nn.Sequential(
            nn.Linear(self.target_dim * self.D, self.target_dim * self.D // 2),
            nn.GELU(),
            nn.Linear(self.target_dim * self.D // 2, self.target_dim),
        )
        self.dropout    = nn.Dropout(p=dropout)
        self.l2_lambda  = l2_lambda
        self.loss_fn = nn.MSELoss(reduction="none")
        self.loss_call_count = 0

        # Populated by _ensure_vmd_ready()
        # _raw_series_list  : flat list of all [T_n, F] arrays
        # _vmd_modes_list   : flat list of all [F, D, T_n] arrays
        # _train_indices    : which indices in the above lists to search during training
        # _test_indices     : which indices to search during evaluation/forecast
        # _train_cache / _test_cache : separate sign-key → (series_idx, t) caches
        self._raw_series_list: Optional[List[np.ndarray]] = None
        self._vmd_modes_list:  Optional[List[np.ndarray]] = None
        self._train_indices:   List[int] = []
        self._test_indices:    List[int] = []
        self._train_cache:     Dict[bytes, Tuple[int, int]] = {}
        self._test_cache:      Dict[bytes, Tuple[int, int]] = {}
        self._vmd_ready = False

    # ── Step 1: ensure VMD data is ready ─────────────────────────────────────────

    def _ensure_vmd_ready(self):
        if self._vmd_ready:
            return

        cache_path = Path(self.vmd_cache_path)

        # If the user supplied a directory (not a .npz file), auto-generate
        # the filename as  <dataset>_D<num_decompositions>.npz
        # e.g.  src/vmd_data  +  etth1  +  D5  →  src/vmd_data/etth1_D5.npz
        if cache_path.suffix != ".npz":
            cache_path.mkdir(parents=True, exist_ok=True)
            safe_name = self.dataset.replace("/", "_")   # GIFT names contain "/"
            cache_path = cache_path / f"{safe_name}_D{self.D}.npz"
            log.info(f"vmd_cache_path is a directory — using file: {cache_path}")

        cache = cache_path

        if cache.exists():
            log.info(f"Loading VMD cache from {cache}")
            self._load_vmd_cache(cache)
        else:
            log.info(f"VMD cache not found — computing from {self.data_root_path}")
            raw_series = self._load_raw_series()
            self._raw_series_list = raw_series
            vmd_modes = self._vmd_transform_all(raw_series)
            self._vmd_modes_list = vmd_modes
            self._save_vmd_cache(cache)

        self._vmd_ready = True
        log.info(
            f"VMD ready: {len(self._vmd_modes_list)} series  "
            f"({sum(s.shape[0] for s in self._raw_series_list)} total timesteps).  "
            f"Windows located by timestamp matching (O(T) once, then cached)."
        )

    # ── Step 1b: load raw series via ProbTS DataManager ────────────────────────

    def _load_raw_series(self) -> List[np.ndarray]:
        """
        Load the full (unsplit) dataset by delegating to DataManager, which
        handles every format ProbTS supports:
          - Long-term CSV datasets  (ETT, traffic, weather, CAISO, Nordpool…)
          - Short-term GluonTS repository datasets  (M4, electricity_nips…)
          - GIFT eval datasets

        The three cases produce different dataset_raw types which we unpack:

          DataFrame          → long-term: one contiguous [T, F] array
          GluonTS dataset    → short-term: one or more series extracted from
                               the test split (which contains the full horizon)
          GiftEvalDataset    → GIFT: one series per training item

        We always extract the FULL time span (all splits) so that the window
        index covers train, validation, and test windows equally.
        """
        import pandas as pd
        from probts.data.data_manager import DataManager, MULTI_VARIATE_DATASETS
        from gluonts.dataset.multivariate_grouper import MultivariateGrouper

        log.info(
            f"Loading raw data via DataManager: "
            f"dataset={self.dataset}  path={self.data_root_path}"
        )

        # Instantiate a minimal DataManager — same arguments the user's
        # DataManager config already provides, so nothing extra to configure.
        # split_val=False avoids unnecessary work; we only need dataset_raw.
        dm = DataManager(
            dataset=self.dataset,
            path=self.data_root_path,
            context_length=self.max_context_length,
            prediction_length=self.max_prediction_length,
            freq=self.freq,
            data_path=self.custom_data_file,   # None for all built-in datasets
            multivariate=True,
            split_val=False,
        )

        raw = dm.dataset_raw

        # ── Case 1: Long-term dataset ─────────────────────────────────────────
        # load_dataset() returns a pandas DataFrame [T, F] (full, unsplit).
        if isinstance(raw, pd.DataFrame):
            arr = raw.values.astype(np.float64)
            log.info(f"Long-term series shape: {arr.shape}")
            self._train_indices = [0]
            self._test_indices  = [0]
            return [arr]

        # ── Case 2: Short-term GluonTS repository dataset ─────────────────────
        # raw.test contains the full time horizon (train window + test future).
        # Multivariate datasets need grouping before extraction.
        if hasattr(raw, 'train') and hasattr(raw, 'test'):
            if self.dataset in MULTI_VARIATE_DATASETS:
                # We need BOTH train and test series:
                #   raw.train → covers training context windows
                #   raw.test  → covers test context windows in the test period
                # Both are stored separately so _find_window_start searches both.
                num_test_dates = int(len(raw.test) / len(raw.train))
                train_grouper = MultivariateGrouper(
                    max_target_dim=int(dm.target_dim)
                )
                test_grouper = MultivariateGrouper(
                    num_test_dates=num_test_dates,
                    max_target_dim=int(dm.target_dim)
                )
                arr_train = list(train_grouper(raw.train))[0]["target"].T.astype(np.float64)
                # Use [-1] (the longest grouped series) so the test raw data
                # covers the full test period, not just the first rolling window.
                arr_test  = list(test_grouper(raw.test))[-1]["target"].T.astype(np.float64)
                log.info(f"GluonTS MV train series: {arr_train.shape}")
                log.info(f"GluonTS MV test  series: {arr_test.shape}")
                self._train_indices = [0]
                self._test_indices  = [1]
                return [arr_train, arr_test]
            else:
                # Univariate: one array per series item in the test split.
                series_list = []
                for item in raw.test:
                    arr = np.array(item["target"], dtype=np.float64)
                    if arr.ndim == 1:
                        arr = arr[:, np.newaxis]        # [T, 1]
                    elif arr.ndim == 2:
                        arr = arr.T                     # [F, T] → [T, F]
                    series_list.append(arr)
                log.info(f"GluonTS UV: {len(series_list)} series loaded")
                self._train_indices = list(range(len(series_list)))
                self._test_indices  = list(range(len(series_list)))
                return series_list

        # ── Case 3: GIFT eval dataset ─────────────────────────────────────────
        # training_dataset items each hold a 'target' array.
        if hasattr(raw, 'training_dataset'):
            series_list = []
            for item in raw.training_dataset:
                arr = np.array(item["target"], dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                elif arr.ndim == 2 and arr.shape[0] == dm.target_dim:
                    arr = arr.T                         # [F, T] → [T, F]
                series_list.append(arr)
            log.info(f"GIFT eval: {len(series_list)} series loaded")
            self._train_indices = list(range(len(series_list)))
            self._test_indices  = list(range(len(series_list)))
            return series_list

        raise ValueError(
            f"Unrecognised dataset_raw type '{type(raw)}' for dataset "
            f"'{self.dataset}'. Cannot extract raw series for VMD."
        )

    # ── Step 2: VMD-transform entire dataset ─────────────────────────────────────

    def _vmd_transform_all(
        self, raw_series: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        VMD-transform every series in the dataset.

        Args:
            raw_series: list of [T_n, F] arrays

        Returns:
            list of [F, D, T_n] arrays  (one per series)
        """
        vmd_modes = []
        for s_idx, series in enumerate(raw_series):
            T_n, F = series.shape
            modes = np.zeros((F, self.D, T_n), dtype=np.float32)
            for f in range(F):
                signal = series[:, f].astype(np.float64)
                try:
                    u, _, _ = VMD(
                        signal,
                        self.vmd_alpha,
                        self.vmd_tau,
                        self.D,
                        self.vmd_DC,
                        self.vmd_init,
                        self.vmd_tol,
                    )                              # u: [D, T_vmd] (T_vmd ≈ T_n)
                    # vmdpy occasionally returns T_vmd = T_n ± 1; clip to T_n
                    T_out = u.shape[1]
                    n     = min(T_out, T_n)
                    modes[f, :, :n] = u[:, :n].astype(np.float32)
                except Exception as e:
                    log.warning(
                        f"VMD failed on series {s_idx} feature {f}: {e}. "
                        "Falling back to raw signal in mode 0."
                    )
                    modes[f, 0] = signal.astype(np.float32)
            vmd_modes.append(modes)
            log.info(f"  VMD: series {s_idx + 1}/{len(raw_series)} done  {T_n}×{F}")
        return vmd_modes

    # ── Step 3: persist / restore cache ──────────────────────────────────────────

    def _save_vmd_cache(self, cache_path: Path):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dict  = {f"raw_{i}": arr for i, arr in enumerate(self._raw_series_list)}
        vmd_dict  = {f"vmd_{i}": arr for i, arr in enumerate(self._vmd_modes_list)}
        idx_dict  = {
            "train_indices": np.array(self._train_indices, dtype=np.int32),
            "test_indices":  np.array(self._test_indices,  dtype=np.int32),
        }
        np.savez_compressed(cache_path, **raw_dict, **vmd_dict, **idx_dict)
        log.info(f"VMD cache saved to {cache_path}")

    def _load_vmd_cache(self, cache_path: Path):
        data = np.load(cache_path, allow_pickle=False)
        n = sum(1 for k in data.files if k.startswith("raw_"))
        self._raw_series_list = [data[f"raw_{i}"] for i in range(n)]
        self._vmd_modes_list  = [data[f"vmd_{i}"] for i in range(n)]
        self._train_indices   = data["train_indices"].tolist()
        self._test_indices    = data["test_indices"].tolist()
        log.info(f"train_indices={self._train_indices}  test_indices={self._test_indices}")


    # ── Step 4: build fingerprint → (series_idx, window_start) index ─────────────

    # Number of sign-diff steps used as the search probe.
    # 10 steps × F features gives 10×F bits — collision-proof for any real dataset
    # while keeping the BLAS matrix-vector multiply very fast.
    _N_PROBE: int = 10

    @staticmethod
    def _find_submatrix_origin(large: np.ndarray, small: np.ndarray,
                               n_probe: int = 10) -> tuple:
        """
        Find the row in `large` [T, F] where `small` [n, F] begins.

        Uses the first n_probe rows of `small` as the search key and a
        BLAS matrix-vector multiply for the scan — O(T × n_probe × F) in
        float32, which is orders of magnitude faster than correlate2d.

        Returns:
            (row,)  — start row index

        Raises:
            ValueError  if the best correlation ratio < 0.99
        """
        n_probe = min(n_probe, small.shape[0])
        probe   = small[:n_probe].astype(np.float32)        # [n_probe, F]
        large_f = large.astype(np.float32)                  # [T, F]

        T, F = large_f.shape
        if T < n_probe:
            raise ValueError("series shorter than probe")

        n_valid = T - n_probe + 1                           # number of candidate positions

        # Build sliding windows [n_valid, n_probe, F] via stride tricks (no copy)
        wins = np.lib.stride_tricks.sliding_window_view(
            large_f, (n_probe, F)
        )[:, 0, :, :]                                       # [n_valid, n_probe, F]

        # Flatten last two dims and use BLAS DGEMV: [n_valid, n_probe*F] @ [n_probe*F]
        probe_flat = probe.ravel()                          # [n_probe*F]
        wins_flat  = np.ascontiguousarray(
            wins.reshape(n_valid, n_probe * F)
        )                                                   # [n_valid, n_probe*F]
        corr = wins_flat @ probe_flat                       # [n_valid]  — pure BLAS

        row = int(np.argmax(corr))

        expected = float(np.dot(probe_flat, probe_flat))
        achieved = float(corr[row])

        if expected > 0 and (achieved / expected) < 0.99:
            raise ValueError(
                f"No definite match — ratio {achieved/expected:.3f} < 0.99"
            )

        return (row,)

    def _find_window_start(self, context: np.ndarray,
                           indices: List[int]) -> Tuple[int, int]:
        """
        Locate context window in the raw series via sign-of-diff correlation.

        Args:
            context : [ctx, F]  float32
            indices : which entries of _raw_series_list to search

        Returns:
            (series_idx, window_start)  — series_idx is an absolute index

        Raises:
            ValueError  if no match found in the given series subset
        """
        sign_ctx = np.sign(
            np.diff(context.astype(np.float32), axis=0)
        )                                                    # [ctx-1, F]

        for s_idx in indices:
            series = self._raw_series_list[s_idx]
            T   = series.shape[0]
            ctx = self.max_context_length
            if T < ctx:
                continue

            # Only include positions t where t + ctx <= T
            sign_hay = np.sign(
                np.diff(series[:T, :].astype(np.float32), axis=0)
            )[:T - ctx + ctx - 1]                           # [T-ctx+ctx-1, F] = [T-1, F]

            try:
                (row,) = self._find_submatrix_origin(
                    sign_hay, sign_ctx, n_probe=self._N_PROBE
                )
                return (s_idx, row)
            except ValueError:
                continue

        raise ValueError(
            f"_find_window_start: no match in series {indices}. "
            f"context shape={context.shape}, context[:3,:3]={context[:3, :3]}"
        )

    def _get_vmd_batch(self, batch_data,
                       indices: List[int],
                       cache:   Dict[bytes, Tuple[int, int]]) -> torch.Tensor:
        """
        Locate each batch item in the specified series subset and return modes.

        Args:
            indices : series indices to search (train or test subset)
            cache   : position cache for this subset (train or test)

        Returns: [B, F, D, ctx]  on the same device as past_target_cdf
        """
        ctx     = self.max_context_length
        device  = batch_data.past_target_cdf.device
        past_np = batch_data.past_target_cdf.cpu().numpy().astype(np.float32)
        B       = past_np.shape[0]

        batch_modes = np.zeros((B, self.target_dim, self.D, ctx), dtype=np.float32)

        for b in range(B):
            context   = past_np[b, -ctx:]                   # [ctx, F]
            signs_key = np.sign(
                np.diff(context[:10], axis=0)
            ).astype(np.int8).tobytes()

            if signs_key not in cache:
                cache[signs_key] = self._find_window_start(context, indices)

            s_idx, t = cache[signs_key]
            modes    = self._vmd_modes_list[s_idx]           # [F, D, T_total]
            t        = min(t, modes.shape[2] - ctx)
            batch_modes[b] = modes[:, :, t: t + ctx]

        return torch.tensor(batch_modes, device=device)


    # ── Core forward ─────────────────────────────────────────────────────────────

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modes : [B, F, D, T_in]  — VMD modes for the context window

        Returns:
            [B, T_out, F]

        Mirrors LinearForecaster (individual=False) with VMD modes as input:

          1. Per-mode instance normalisation (mean/std over T_in) — stabilises
             inputs regardless of raw data scale.
          2. Flatten F and D → treat as F*D independent channels: [B, F*D, T_in].
          3. self.linear     [T_in → T_out]  — time projection per channel,
             same role as LinearForecaster.linear.
          4. self.out_linear [F*D  → F]      — feature projection per time step,
             same role as LinearForecaster.out_linear.
        """
        B, Fv, D, T_in = modes.shape

        # ── 1. Per-mode normalisation (over the T_in dimension) ───────────────
        mu  = modes.mean(dim=-1, keepdim=True)              # [B, F, D, 1]
        sig = modes.std( dim=-1, keepdim=True).clamp(1e-8)  # [B, F, D, 1]
        modes_norm = (modes - mu) / sig                     # [B, F, D, T_in]

        # ── 2. Flatten F and D into one channel dimension ─────────────────────
        x = self.dropout(modes_norm.reshape(B, Fv * D, T_in))  # [B, F*D, T_in]

        # ── 3. Time projection: T_in → T_out  (applied to last dim) ──────────
        # Matches LinearForecaster: linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.linear(x)                                  # [B, F*D, T_out]

        # ── 4. Feature projection: F*D → F  (applied to last dim) ────────────
        x = x.permute(0, 2, 1)                              # [B, T_out, F*D]
        x = self.out_linear(x)                              # [B, T_out, F]

        return x                                            # [B, T_out, F]

    # ── ProbTS interface ──────────────────────────────────────────────────────────

    def loss(self, batch_data) -> torch.Tensor:
        self._ensure_vmd_ready()
        self.loss_call_count += 1

        if self.use_scaling:
            self.get_scale(batch_data)

        modes   = self._get_vmd_batch(batch_data,
                                   self._train_indices,
                                   self._train_cache)        # [B, F, D, ctx]
        outputs = self(modes)                                   # [B, T_out, F]

        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        result = loss.mean()

        # L2 regularisation on both linear layers (equivalent to weight decay)
        if self.l2_lambda > 0:
            l2 = sum(p.pow(2).sum()
                     for p in list(self.linear.parameters()) +
                              list(self.out_linear.parameters()))
            result = result + self.l2_lambda * l2

        print(f"[loss call {self.loss_call_count:>6}]  loss={result.item():.6f}")
        return result

    def forecast(self, batch_data, num_samples=None) -> torch.Tensor:
        self._ensure_vmd_ready()

        if self.use_scaling:
            self.get_scale(batch_data)

        modes     = self._get_vmd_batch(batch_data,
                                     self._test_indices,
                                     self._test_cache)       # [B, F, D, ctx]
        forecasts = self(modes).unsqueeze(1)                   # [B, 1, T_out, F]
        return forecasts


# ─────────────────────────────────────────────────────────────────────────────────
# YAML config example
# ─────────────────────────────────────────────────────────────────────────────────
#
#   model:
#     forecaster:
#       class_path: our_models.vmd.VMDDecompositionForecaster
#       init_args:
#         data_root_path:  /data/datasets      # same as DataManager root_path
#         vmd_cache_path:  /data/vmd/etth1_cache.npz
#         custom_data_file: null               # only for unknown dataset names
#         num_decompositions: 5
#         vmd_alpha: 2000.0
#     learning_rate: 0.001
#
#   data:
#     data_manager:
#       class_path: probts.data.data_manager.DataManager
#       init_args:
#         dataset: etth1                       # self.dataset — used by both
#         root_path: /data/datasets            # same folder as data_root_path
#         context_length: 96
#
# The first run calls get_dataset_info('etth1') → 'ETT-small/ETTh1.csv',
# then load_dataset('/data/datasets', 'ETT-small/ETTh1.csv', 'H'),
# VMD-transforms the full series, and writes etth1_cache.npz.
# Every subsequent run loads from cache — VMD never runs again.
# ─────────────────────────────────────────────────────────────────────────────────