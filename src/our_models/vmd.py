# ---------------------------------------------------------------------------------
# VMD Decomposition Forecaster for ProbTS
#
# Architecture:
#   1. VMD (Variational Mode Decomposition) transforms the ENTIRE raw dataset
#      once on the first call, saving results to vmd_cache_path.
#      Subsequent runs load from cache — VMD never runs again for the same data.
#   2. Each batch lookup uses a value fingerprint to find the correct VMD window
#      from the pre-transformed dataset, O(1) per batch item.
#   3. A CNN encoder maps VMD modes → per-mode A (amplitude) and Phi (phase).
#   4. Differentiable FFT synthesis applies A and Phi, sums over modes → forecast.
#
# Gradient flow:
#   VMD and data loading have NO gradient.
#   Gradients flow: Encoder → A, Phi → fft_phase_shift → output.
#
# Dependencies:
#   pip install vmdpy
# ---------------------------------------------------------------------------------

import os
import logging
from pathlib import Path
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from scipy.signal import correlate2d
import torch
import torch.nn as nn
import torch.nn.functional as F

from vmdpy import VMD

from probts.model.forecaster import Forecaster

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────────
# 1. Differentiable FFT phase shift
# ─────────────────────────────────────────────────────────────────────────────────

def fft_phase_shift(x: torch.Tensor, phi: torch.Tensor, T_out: int) -> torch.Tensor:
    """
    Shift each signal in x by phi samples (fractional, learnable), resample to T_out.

    Args:
        x   : [N, T_in]   real signals
        phi : [N]         shift in samples (fractional, differentiable)
        T_out             desired output length

    Returns: [N, T_out]

    Note: irfft(n=T_out) naturally handles T_out ≠ T_in via zero-padding or
    truncation in the frequency domain. For band-limited periodic signals (VMD
    modes), this is exact.
    """
    T_in = x.shape[-1]
    X = torch.fft.rfft(x, n=T_in)
    freqs = torch.fft.rfftfreq(T_in, device=x.device)          # [T_in//2+1]
    phase = torch.exp(
        torch.complex(
            torch.zeros_like(phi.unsqueeze(-1).expand(-1, freqs.shape[0])),
            -2.0 * torch.pi * freqs.unsqueeze(0) * phi.unsqueeze(-1),
        )
    )
    return torch.fft.irfft(X * phase, n=T_out)                 # [N, T_out]



def _gluonts_time_features(dates: "pd.DatetimeIndex", freq: str) -> np.ndarray:
    """
    Compute time features using GluonTS's own machinery so that the result
    matches the values that GluonTS puts into past_time_feat in the batch.

    Falls back to ProbTS's time_features() if gluonts is unavailable.

    Returns: float32 array [T, D]
    """
    try:
        from gluonts.time_feature import time_features_from_frequency_str
        feat_fns = time_features_from_frequency_str(freq)
        stamp = np.column_stack([f(dates) for f in feat_fns]).astype(np.float32)
        return stamp                                          # [T, D]
    except Exception:
        # Fallback to ProbTS implementation
        from probts.data.data_utils.time_features import time_features as tf_fn
        return tf_fn(dates, freq=freq).T.astype(np.float32)  # [T, D]

# ─────────────────────────────────────────────────────────────────────────────────
# 2. Mode encoder  (VMD modes → A and Phi)
# ─────────────────────────────────────────────────────────────────────────────────

class ModeEncoder(nn.Module):
    """
    Maps one feature's D VMD modes [N, D, T_in] → A [N, D], Phi [N, D].
    N = B * F (batch × features, processed jointly with shared weights).
    """

    def __init__(self, D: int, hidden: int = 64):
        super().__init__()
        self.D = D
        self.cnn = nn.Sequential(
            nn.Conv1d(D, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * D),
        )

    def forward(self, modes: torch.Tensor):
        h = self.cnn(modes).squeeze(-1)
        out = self.head(h)
        return out[:, :self.D], out[:, self.D:]                 # A_raw, Phi_raw


# ─────────────────────────────────────────────────────────────────────────────────
# 3. Raw data loader  (via ProbTS infrastructure)
# ─────────────────────────────────────────────────────────────────────────────────
# load_dataset() from probts.data.data_utils.get_datasets handles every format
# ProbTS supports (ETT CSVs, traffic, CAISO, Monash .tsf, etc.).
# get_dataset_info() maps a dataset name to its relative file path + frequency,
# so the caller only needs to supply the root folder, not the file path.
# For datasets not in get_dataset_info's built-in table, pass custom_data_file.
# ─────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────
# 4. Main model
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
        encoder_hidden (int):
            CNN encoder hidden size.
        phi_scale (float | None):
            Phase shift range ±phi_scale samples. Defaults to context_length.
        allow_negative_amplitude (bool):
            True = amplitudes can be negative (modes can cancel each other).
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
        encoder_hidden: int = 64,
        phi_scale: Optional[float] = None,
        allow_negative_amplitude: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # data_root_path: root folder containing all dataset subdirectories,
        #   e.g. /data/  (same value you pass to DataManager as root_path).
        # custom_data_file: relative path inside data_root_path for datasets
        #   not listed in get_dataset_info, e.g. 'my_project/series.csv'.
        #   Leave as None for any built-in ProbTS dataset name.
        self.data_root_path = data_root_path
        self.vmd_cache_path = vmd_cache_path
        self.custom_data_file = custom_data_file
        self.D = num_decompositions
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol
        self.allow_negative_amplitude = allow_negative_amplitude
        self.phi_scale = float(phi_scale or self.max_context_length)

        self.encoder = ModeEncoder(D=self.D, hidden=encoder_hidden)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.loss_call_count = 0

        # Populated by _ensure_vmd_ready()
        # _vmd_modes_list : List[np.ndarray]  each [F, D, T_n]
        # _raw_series_list: List[np.ndarray]  each [T_n, F]
        # _data_stamp     : np.ndarray [T_full, time_feat_dim]  (long-term datasets)
        # _position_cache : Dict[bytes, Tuple[int, int]]
        #                   time-feature key → (series_idx, window_start)
        self._vmd_modes_list:  Optional[List[np.ndarray]] = None
        self._raw_series_list: Optional[List[np.ndarray]] = None
        self._data_stamp:      Optional[np.ndarray] = None
        self._position_cache:  Dict[bytes, Tuple[int, int]] = {}
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
            log.info(f"_data_stamp before save: {None if self._data_stamp is None else self._data_stamp.shape}")
            self._save_vmd_cache(cache, raw_series, vmd_modes)

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
        # data_stamp [T, time_feat_dim] encodes the timestamp of every row
        # and is used later to locate batch windows by time, not by value.
        if isinstance(raw, pd.DataFrame):
            arr = raw.values.astype(np.float64)
            log.info(f"Long-term series shape: {arr.shape}")
            if hasattr(dm, 'data_stamp') and dm.data_stamp is not None:
                self._data_stamp = np.array(dm.data_stamp, dtype=np.float32)
                log.info(f"data_stamp shape: {self._data_stamp.shape}")
            else:
                log.warning("data_stamp not available — will fall back to z-norm search")
            return [arr]

        # ── Case 2: Short-term GluonTS repository dataset ─────────────────────
        # raw.test contains the full time horizon (train window + test future).
        # Multivariate datasets need grouping before extraction.
        if hasattr(raw, 'train') and hasattr(raw, 'test'):
            if self.dataset in MULTI_VARIATE_DATASETS:
                # Use raw.TRAIN (not raw.test) so the series starts at the
                # training data origin (e.g. 2014-01-01 for electricity_nips).
                # All context windows in training AND test batches come from
                # the training time-range, so this covers both.
                train_grouper = MultivariateGrouper(max_target_dim=int(dm.target_dim))
                grouped = list(train_grouper(raw.train))
                item = grouped[0]
                arr  = item["target"].T.astype(np.float64)          # [T_train, F]
                log.info(f"GluonTS MV series shape: {arr.shape}")

                # Build data_stamp from the item's start timestamp + freq
                T_n   = arr.shape[0]
                raw_start = item["start"]
                log.info(f"  item start={raw_start!r}  type={type(raw_start).__name__}  dm.freq={dm.freq!r}")
                try:
                    if hasattr(raw_start, 'to_timestamp'):
                        ts0 = raw_start.to_timestamp()
                    else:
                        ts0 = pd.Timestamp(raw_start)
                    dates = pd.date_range(start=ts0, periods=T_n, freq=dm.freq)
                    self._data_stamp = _gluonts_time_features(dates, dm.freq)
                    log.info(f"  data_stamp shape: {self._data_stamp.shape}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to build data_stamp for '{self.dataset}': {exc} "
                        f"start={raw_start!r} freq={dm.freq!r} T_n={T_n}"
                    ) from exc
                return [arr]
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
                # Build data_stamp from the first item (all share the same timeline)
                first     = list(raw.test)[0]
                T_n       = len(series_list[0])
                raw_start = first["start"]
                log.info(f"  item start={raw_start!r}  type={type(raw_start).__name__}  dm.freq={dm.freq!r}")
                try:
                    if hasattr(raw_start, 'to_timestamp'):
                        ts0 = raw_start.to_timestamp()
                    else:
                        ts0 = pd.Timestamp(raw_start)
                    dates = pd.date_range(start=ts0, periods=T_n, freq=dm.freq)
                    self._data_stamp = _gluonts_time_features(dates, dm.freq)
                    log.info(f"  data_stamp shape: {self._data_stamp.shape}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to build data_stamp for '{self.dataset}': {exc} "
                        f"start={raw_start!r} freq={dm.freq!r} T_n={T_n}"
                    ) from exc
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

    def _save_vmd_cache(
        self,
        cache_path: Path,
        raw_series: List[np.ndarray],
        vmd_modes: List[np.ndarray],
    ):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_dict  = {f"raw_{i}": arr for i, arr in enumerate(raw_series)}
        vmd_dict  = {f"vmd_{i}": arr for i, arr in enumerate(vmd_modes)}
        stamp_kw  = {"data_stamp": self._data_stamp} if self._data_stamp is not None else {}
        np.savez_compressed(cache_path, **raw_dict, **vmd_dict, **stamp_kw)
        log.info(f"VMD cache saved to {cache_path}")

    def _load_vmd_cache(self, cache_path: Path):
        data = np.load(cache_path, allow_pickle=False)
        n = sum(1 for k in data.files if k.startswith("raw_"))
        self._raw_series_list = [data[f"raw_{i}"] for i in range(n)]
        self._vmd_modes_list  = [data[f"vmd_{i}"] for i in range(n)]
        if "data_stamp" in data.files:
            self._data_stamp = data["data_stamp"]
            log.info(f"Loaded data_stamp {self._data_stamp.shape} from cache")
        else:
            log.warning("Cache has no data_stamp — delete cache to regenerate with timestamp support")

    # ── Step 4: build fingerprint → (series_idx, window_start) index ─────────────

    @staticmethod
    def _find_submatrix_origin(large: np.ndarray, small: np.ndarray) -> tuple:
        """
        Find the row position in `large` [T, F] where `small` [n, F] begins,
        using 2-D cross-correlation as a fast similarity measure.

        Both matrices must have the same number of columns (features F).
        The search is along the time axis only (columns are never shifted).

        Returns:
            (row,)  — start row of `small` inside `large`

        Raises:
            ValueError  if no close match is found (best normalised correlation < 0.99)
        """
        large_f = large.astype(np.float64)
        small_f = small.astype(np.float64)

        # correlate2d with mode='valid' slides small over large along axis-0;
        # result shape is (T - n + 1, 1) because columns always align fully.
        corr = correlate2d(large_f, small_f, mode='valid')  # [T-n+1, 1]
        row  = int(np.argmax(corr[:, 0]))

        # Normalised check: perfect sign match → corr == sum(small**2)
        expected = float(np.sum(small_f ** 2))
        achieved = float(corr[row, 0])

        if expected > 0 and (achieved / expected) < 0.99:
            raise ValueError(
                f"No definite match found — best correlation {achieved:.2f} / "
                f"expected {expected:.2f} (ratio {achieved/expected:.3f} < 0.99). "
                f"The context window sign pattern does not appear in the raw series."
            )

        return (row,)

    def _find_window_start(self, context: np.ndarray) -> Tuple[int, int]:
        """
        Find where this context window appears in the raw series by correlating
        the sign-of-differences patterns.

        sign(diff) is invariant to any monotone per-feature transformation
        (z-score, CDF normalisation, mean-scaling, etc.), so this works even
        when past_target_cdf has been normalised by the DataManager.

        Steps:
          1. Compute sign(diff(context))  →  [ctx-1, F]  probe pattern
          2. Compute sign(diff(series))   →  [T-1, F]    haystack
          3. Use 2-D cross-correlation to find the row where the probe appears
          4. Raise ValueError if no definite match (correlation ratio < 0.99)

        Args:
            context : [ctx, F]  float32 context window from the batch

        Returns:
            (series_idx, window_start)

        Raises:
            ValueError  if no matching position is found in any series
        """
        # Use all ctx-1 difference steps for maximum discriminability
        sign_ctx = np.sign(
            np.diff(context.astype(np.float64), axis=0)
        )                                                    # [ctx-1, F]

        for s_idx, series in enumerate(self._raw_series_list):
            T = series.shape[0]
            ctx = self.max_context_length
            if T < ctx:
                continue

            sign_raw = np.sign(
                np.diff(series[:T - ctx + ctx, :].astype(np.float64), axis=0)
            )                                                # [T-1, F]

            # Restrict haystack so t + ctx <= T (only valid start positions)
            max_valid_row = T - ctx                         # sign index max_valid_row
            # sign_raw[t : t+ctx-1] corresponds to series[t : t+ctx]
            sign_hay = sign_raw[:max_valid_row + ctx - 1]   # [max_valid_row + ctx - 1, F]

            try:
                (row,) = self._find_submatrix_origin(sign_hay, sign_ctx)
                return (s_idx, row)
            except ValueError:
                continue

        raise ValueError(
            f"_find_window_start: context window not found in any raw series. "
            f"context shape={context.shape}, "
            f"context[:3, :3]={context[:3, :3]}"
        )

    def _get_vmd_batch(self, batch_data) -> torch.Tensor:
        """
        For each item in the batch, locate its position in the pre-transformed
        dataset then slice the corresponding VMD modes.

        Returns: [B, F, D, ctx]  on the same device as past_target_cdf
        """
        ctx    = self.max_context_length
        device = batch_data.past_target_cdf.device
        past_np = batch_data.past_target_cdf.cpu().numpy().astype(np.float32)
        B       = past_np.shape[0]

        batch_modes = np.zeros((B, self.target_dim, self.D, ctx), dtype=np.float32)

        for b in range(B):
            context = past_np[b, -ctx:]                      # [ctx, F]

            # Cache key: exact sign pattern — no rounding, no precision issues
            signs_key = np.sign(
                np.diff(context[:10], axis=0)
            ).astype(np.int8).tobytes()

            if signs_key not in self._position_cache:
                self._position_cache[signs_key] = self._find_window_start(context)

            s_idx, t = self._position_cache[signs_key]
            modes = self._vmd_modes_list[s_idx]              # [F, D, T_total]
            T_n   = modes.shape[2]
            t     = min(t, T_n - ctx)                        # clamp: never overshoot
            batch_modes[b] = modes[:, :, t: t + ctx]         # [F, D, ctx]

        return torch.tensor(batch_modes, device=device)


    # ── Core synthesis (differentiable) ──────────────────────────────────────────

    def _synthesize(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modes: [B, F, D, T_in]

        Returns:
            [B, F, T_out]
        """
        B, Fv, D, T_in = modes.shape
        T_out = self.max_prediction_length

        modes_bf = modes.view(B * Fv, D, T_in)
        A_raw, Phi_raw = self.encoder(modes_bf)             # each [B*F, D]

        A   = A_raw if self.allow_negative_amplitude else F.softplus(A_raw)
        Phi = torch.tanh(Phi_raw) * self.phi_scale

        modes_bfd = modes_bf.reshape(B * Fv * D, T_in)
        phi_bfd   = Phi.reshape(B * Fv * D)

        shifted = fft_phase_shift(modes_bfd, phi_bfd, T_out)   # [B*F*D, T_out]
        shifted = shifted.view(B, Fv, D, T_out)

        output = (A.view(B, Fv, D).unsqueeze(-1) * shifted).sum(dim=2)  # [B, F, T_out]
        return output

    def forward(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modes: [B, F, D, T_in]   — pre-looked-up VMD modes

        Returns:
            [B, T_out, F]

        RevIN-style per-feature normalisation is applied before synthesis and
        reversed afterwards.  This keeps the encoder inputs in a well-conditioned
        range regardless of the raw data scale (e.g. kWh in the thousands for
        electricity datasets), preventing the exploding-loss cold-start problem.
        """
        # ── Per-feature normalisation (mean/std across D and T) ───────────────
        # modes: [B, F, D, T_in]
        mu  = modes.mean(dim=(-2, -1), keepdim=True)              # [B, F, 1, 1]
        sig = modes.std( dim=(-2, -1), keepdim=True).clamp(1e-8)  # [B, F, 1, 1]
        modes_norm = (modes - mu) / sig                            # [B, F, D, T_in]

        # ── Synthesis in normalised space ─────────────────────────────────────
        out_norm = self._synthesize(modes_norm)                    # [B, F, T_out]

        # ── Un-normalise back to original scale ───────────────────────────────
        mu_2d  = mu.squeeze(-1).squeeze(-1)                        # [B, F]
        sig_2d = sig.squeeze(-1).squeeze(-1)                       # [B, F]
        output = out_norm * sig_2d.unsqueeze(-1) + mu_2d.unsqueeze(-1)  # [B, F, T_out]

        return output.permute(0, 2, 1)                             # [B, T_out, F]

    # ── ProbTS interface ──────────────────────────────────────────────────────────

    def loss(self, batch_data) -> torch.Tensor:
        self._ensure_vmd_ready()
        self.loss_call_count += 1

        if not getattr(self, '_batch_diagnosed', False):
            '''
            self._batch_diagnosed = True
            print("\n=== batch_data fields ===")
            for name in vars(batch_data):
                val = getattr(batch_data, name)
                if hasattr(val, 'shape'):
                    print(f"  {name}: {val.shape}  dtype={val.dtype}")
                else:
                    print(f"  {name}: {type(val).__name__} = {val}")
            print("=========================\n")
            # Save past_target_cdf to disk for inspection
            save_path = Path(self.vmd_cache_path).parent / "past_target_cdf_sample.npy"
            np.save(save_path, batch_data.past_target_cdf.cpu().numpy())
            log.info(f"Saved past_target_cdf sample to {save_path}")
        '''
        if self.use_scaling:
            self.get_scale(batch_data)

        modes   = self._get_vmd_batch(batch_data)              # [B, F, D, ctx]
        outputs = self(modes)                                   # [B, T_out, F]

        loss = self.loss_fn(batch_data.future_target_cdf, outputs)
        loss = self.get_weighted_loss(batch_data, loss)
        result = loss.mean()

        print(f"[loss call {self.loss_call_count:>6}]  loss={result.item():.6f}")
        return result

    def forecast(self, batch_data, num_samples=None) -> torch.Tensor:
        self._ensure_vmd_ready()

        if self.use_scaling:
            self.get_scale(batch_data)

        modes     = self._get_vmd_batch(batch_data)            # [B, F, D, ctx]
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
#         encoder_hidden: 64
#         phi_scale: null
#         allow_negative_amplitude: true
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