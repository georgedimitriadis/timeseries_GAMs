"""
ProbTSDatasetLoader
====================
Loads any ProbTS benchmark dataset and returns (train, val, test) splits as
numpy arrays — without requiring a config file path.

How metadata is resolved (in priority order)
---------------------------------------------
1. Explicit keyword arguments passed to the constructor always win.
2. For short-term GluonTS datasets, ``freq`` and ``prediction_length`` are
   read directly from ``dataset.metadata`` (the same source ProbTS uses
   internally), and ``context_length`` defaults to
   ``metadata.prediction_length`` when not given.
3. For long-term CSV datasets (ETT, traffic, weather, CAISO, Nordpool …),
   ``freq``, the fixed train/val/test boundaries, and a sensible default
   ``context_length`` / ``prediction_length`` are looked up from the
   ``LONGTERM_DATASET_INFO`` table embedded in this module.
4. GIFT eval dataset names (``"gift/…"``) carry their own metadata inside
   the dataset object itself, so nothing extra is needed.

This mirrors exactly what ProbTS does when you pass a dataset name on the CLI:
long-term datasets require explicit context/prediction lengths at that CLI
level, but the freq and split boundaries are always fixed per dataset.

Returned splits
---------------
All splits are lists of float64 numpy arrays shaped [T, F].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known multivariate GluonTS datasets (mirrors ProbTS internals)
# ---------------------------------------------------------------------------
MULTI_VARIATE_DATASETS = {
    "electricity_nips",
    "solar_nips",
    "exchange_rate_nips",
    "traffic_nips",
    "taxi_30min",
    "wiki-rolling_nips",
    "wind_farms_without_missing",
}

# ---------------------------------------------------------------------------
# Long-term dataset metadata
#
# Split boundaries are the canonical fixed values used by ProbTS / Autoformer
# / PatchTST / iTransformer for these datasets.  They are absolute time-step
# indices, not ratios, because the community always uses the same cuts.
#
# Keys: dataset name as recognised by ProbTS DataManager
# Values:
#   freq              – pandas offset alias
#   n_features        – number of variate columns in the CSV
#   train_end         – last training timestep (exclusive)
#   val_end           – last validation timestep (exclusive); test goes to end
#   default_context   – context length used when not supplied by the caller
#   default_pred      – prediction length used when not supplied by the caller
# ---------------------------------------------------------------------------
LONGTERM_DATASET_INFO: Dict[str, dict] = {
    # ETT hourly (17,420 total rows: 8,640 train / 2,880 val / 2,880+1,020 test)
    "etth1": dict(freq="H",  n_features=7,  train_end=8640,  val_end=11520, default_context=336, default_pred=96),
    "etth2": dict(freq="H",  n_features=7,  train_end=8640,  val_end=11520, default_context=336, default_pred=96),
    # ETT 15-min (69,680 total rows: 34,560 / 11,520 / 11,520+11,520 test)
    "ettm1": dict(freq="15T", n_features=7, train_end=34560, val_end=46080, default_context=336, default_pred=96),
    "ettm2": dict(freq="15T", n_features=7, train_end=34560, val_end=46080, default_context=336, default_pred=96),
    # Traffic (hourly, 17,544 rows — standard 70/10/20 split)
    "traffic_ltsf":     dict(freq="H",  n_features=862, train_end=12280, val_end=13896, default_context=336, default_pred=96),
    # Electricity (hourly, 26,304 rows — standard 70/10/20)
    "electricity_ltsf": dict(freq="H",  n_features=321, train_end=18413, val_end=21044, default_context=336, default_pred=96),
    # Exchange rate (daily, 7,588 rows — standard 70/10/20)
    "exchange_ltsf":    dict(freq="D",  n_features=8,   train_end=5312,  val_end=6071,  default_context=336, default_pred=96),
    # National illness (weekly, 966 rows — standard 70/10/20)
    "illness_ltsf":     dict(freq="W",  n_features=7,   train_end=676,   val_end=772,   default_context=36,  default_pred=24),
    # Weather (10-min, 52,696 rows — standard 70/10/20)
    "weather_ltsf":     dict(freq="10T", n_features=21, train_end=36887, val_end=42157, default_context=336, default_pred=96),
    # CAISO (hourly, 8,760 rows — standard 70/10/20)
    "caiso":            dict(freq="H",  n_features=10,  train_end=6132,  val_end=7008,  default_context=336, default_pred=96),
    # Nordpool (hourly, 17,544 rows — standard 70/10/20)
    "nordpool":         dict(freq="H",  n_features=18,  train_end=12280, val_end=13896, default_context=336, default_pred=96),
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProbTSDatasetLoader:
    """
    Load a ProbTS benchmark dataset and expose train / val / test splits.

    Parameters
    ----------
    dataset : str
        Dataset name as used by ProbTS, e.g. ``"etth1"``, ``"m4_hourly"``,
        ``"electricity_nips"``, ``"gift/ett1/H/long"``.
    data_root_path : str | Path
        Root directory that contains the dataset files (the ``path`` arg
        passed to DataManager for long-term CSV datasets; GluonTS datasets
        are fetched automatically to the GluonTS cache).
    context_length : int, optional
        Override the context (look-back) length.  For long-term datasets a
        sensible default is taken from ``LONGTERM_DATASET_INFO`` when omitted.
        For short-term datasets it defaults to the dataset's own
        ``prediction_length`` from GluonTS metadata.
    prediction_length : int, optional
        Override the prediction (forecast) length.  Defaults as above.
    freq : str, optional
        Override the frequency string.  Rarely needed; defaults come from
        ``LONGTERM_DATASET_INFO`` or GluonTS metadata.
    **kwargs
        Any remaining DataManager arguments (``multivariate``,
        ``data_path`` for custom .tsf files, etc.).

    Examples
    --------
    >>> loader = ProbTSDatasetLoader("etth1", "./datasets", prediction_length=192)
    >>> train, val, test = loader.load()
    >>> print(train[0].shape)   # (8640, 7)

    >>> loader = ProbTSDatasetLoader("m4_hourly", "./datasets")
    >>> train, val, test = loader.load()

    >>> loader = ProbTSDatasetLoader("gift/ett1/H/long", "./datasets")
    >>> train, val, test = loader.load()
    """

    def __init__(
        self,
        dataset: str,
        data_root_path: str | Path,
        context_length: Optional[int] = None,
        prediction_length: Optional[int] = None,
        freq: Optional[str] = None,
        **kwargs,
    ):
        self.dataset        = dataset
        self.data_root_path = Path(data_root_path)
        self.multivariate   = bool(kwargs.pop("multivariate", True))
        self.custom_data_file = kwargs.pop("data_path", None)
        self._extra_dm_kwargs = kwargs

        # These three may remain None until _resolve_metadata() fills them in
        # (which happens lazily inside load()).
        self._context_length_override    = context_length
        self._prediction_length_override = prediction_length
        self._freq_override              = freq

        # Resolved values — filled by _resolve_metadata()
        self.context_length:    Optional[int] = None
        self.prediction_length: Optional[int] = None
        self.freq:              Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Load the dataset and return ``(train, val, test)`` splits.

        Each split is a list of 2-D ``float64`` arrays shaped ``[T, F]``.
        """
        raw_series, dataset_type, meta = self._load_raw_series()
        self._apply_metadata(meta)
        train, val, test = self._split(raw_series, dataset_type, meta)
        return train, val, test

    # ------------------------------------------------------------------
    # Dataset-type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_longterm(dataset_name: str) -> bool:
        return dataset_name in LONGTERM_DATASET_INFO

    @staticmethod
    def _is_gift(dataset_name: str) -> bool:
        return dataset_name.startswith("gift/")

    # ------------------------------------------------------------------
    # Raw series loading
    # ------------------------------------------------------------------

    def _load_raw_series(self):
        """
        Instantiate DataManager, extract full unsplit arrays, and collect
        whatever metadata it surfaces.

        Returns
        -------
        series    : raw series data (format depends on dataset_type)
        dtype     : str  "longterm" | "gluonts_mv" | "gluonts_uv" | "gift"
        meta      : dict  with keys freq, prediction_length, context_length,
                    train_end, val_end (last two only for long-term datasets)
        """
        from probts.data.data_manager import DataManager  # lazy import

        # ── Determine what to pass to DataManager ─────────────────────────
        # For short-term / GIFT datasets, DataManager sets freq and
        # prediction_length on itself from GluonTS metadata, so we can pass
        # dummy values that get overwritten.  For long-term datasets we need
        # real values; pull them from the lookup table if not overridden.
        lt_info = LONGTERM_DATASET_INFO.get(self.dataset)

        init_ctx  = self._context_length_override
        init_pred = self._prediction_length_override
        init_freq = self._freq_override

        if lt_info is not None:
            # Long-term: fill missing args from the lookup table
            init_ctx  = init_ctx  or lt_info["default_context"]
            init_pred = init_pred or lt_info["default_pred"]
            init_freq = init_freq or lt_info["freq"]
        else:
            # Short-term / GIFT: DataManager will set the real values after
            # loading; we pass whatever the user gave us (or placeholder 1s
            # that will be ignored for metadata resolution).
            init_ctx  = init_ctx  or 1
            init_pred = init_pred or 1
            init_freq = init_freq or "H"   # DataManager overwrites this

        log.info(
            f"Loading via DataManager: dataset={self.dataset}  "
            f"path={self.data_root_path}  freq={init_freq}  "
            f"ctx={init_ctx}  pred={init_pred}"
        )

        dm = DataManager(
            dataset=self.dataset,
            path=str(self.data_root_path),
            context_length=init_ctx,
            prediction_length=init_pred,
            freq=init_freq,
            data_path=self.custom_data_file,
            multivariate=self.multivariate,
            split_val=False,
            **self._extra_dm_kwargs,
        )

        raw = dm.dataset_raw

        # ── Collect metadata that DataManager resolved ────────────────────
        # For GluonTS datasets dm.freq / dm.prediction_length are set from
        # dataset.metadata after loading.
        resolved_freq = getattr(dm, "freq", init_freq) or init_freq
        resolved_pred = getattr(dm, "prediction_length", init_pred) or init_pred
        resolved_ctx  = getattr(dm, "context_length",    init_ctx)  or init_ctx

        meta: dict = {
            "freq": resolved_freq,
            "prediction_length": int(resolved_pred),
            "context_length": int(resolved_ctx),
        }

        # ── Case 1: Long-term (pandas DataFrame) ──────────────────────────
        if isinstance(raw, pd.DataFrame):
            arr = raw.values.astype(np.float64)
            log.info(f"Long-term series shape: {arr.shape}")
            if lt_info:
                meta["train_end"] = lt_info["train_end"]
                meta["val_end"]   = lt_info["val_end"]
            return [arr], "longterm", meta

        # ── Case 2: Short-term GluonTS ─────────────────────────────────────
        if hasattr(raw, "train") and hasattr(raw, "test"):
            # Prefer metadata from the dataset object itself
            if hasattr(raw, "metadata") and raw.metadata is not None:
                meta["freq"]              = str(raw.metadata.freq)
                meta["prediction_length"] = int(raw.metadata.prediction_length)

            if self.dataset in MULTI_VARIATE_DATASETS:
                from gluonts.dataset.multivariate_grouper import MultivariateGrouper

                num_test_dates = int(len(raw.test) / len(raw.train))
                train_grouper = MultivariateGrouper(max_target_dim=int(dm.target_dim))
                test_grouper  = MultivariateGrouper(
                    num_test_dates=num_test_dates,
                    max_target_dim=int(dm.target_dim),
                )
                arr_train = list(train_grouper(raw.train))[0]["target"].T.astype(np.float64)
                arr_test  = list(test_grouper(raw.test))[-1]["target"].T.astype(np.float64)
                log.info(f"GluonTS MV  train={arr_train.shape}  test={arr_test.shape}")
                
                return (arr_train, arr_test), "gluonts_mv", meta
            else:
                train_list, test_list = [], []
                for item in raw.train:
                    arr = np.array(item["target"], dtype=np.float64)
                    arr = arr[:, np.newaxis] if arr.ndim == 1 else arr.T
                    train_list.append(arr)
                for item in raw.test:
                    arr = np.array(item["target"], dtype=np.float64)
                    arr = arr[:, np.newaxis] if arr.ndim == 1 else arr.T
                    test_list.append(arr)
                log.info(f"GluonTS UV  {len(train_list)} train / {len(test_list)} test series")
                return (train_list, test_list), "gluonts_uv", meta

        # ── Case 3: GIFT eval dataset ──────────────────────────────────────
        if hasattr(raw, "training_dataset"):
            series_list = []
            for item in raw.training_dataset:
                arr = np.array(item["target"], dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                elif arr.ndim == 2 and arr.shape[0] == dm.target_dim:
                    arr = arr.T
                series_list.append(arr)
            # GIFT datasets expose freq in the items themselves
            if series_list:
                first_item = next(iter(raw.training_dataset))
                if "freq" in first_item:
                    meta["freq"] = str(first_item["freq"])
            log.info(f"GIFT eval: {len(series_list)} series")
            return series_list, "gift", meta

        raise ValueError(
            f"Unrecognised dataset_raw type '{type(raw)}' for dataset "
            f"'{self.dataset}'."
        )

    def _apply_metadata(self, meta: dict) -> None:
        """Write resolved metadata back onto self, respecting user overrides."""
        self.freq = (
            self._freq_override
            if self._freq_override is not None
            else meta["freq"]
        )
        self.prediction_length = (
            self._prediction_length_override
            if self._prediction_length_override is not None
            else meta["prediction_length"]
        )
        self.context_length = (
            self._context_length_override
            if self._context_length_override is not None
            else meta["context_length"]
        )

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _split(self, raw_series, dataset_type: str, meta: dict):
        if dataset_type == "longterm":
            return self._split_longterm(raw_series[0], meta)
        if dataset_type == "gluonts_mv":
            return self._split_gluonts_mv(*raw_series, meta)
        if dataset_type == "gluonts_uv":
            train_list, test_list = raw_series
            return self._split_gluonts_uv(train_list, test_list, meta)
        if dataset_type == "gift":
            return self._split_gift(raw_series)
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # ── Long-term ──────────────────────────────────────────────────────────

    def _split_longterm(self, arr: np.ndarray, meta: dict):
        """
        Use the canonical fixed split boundaries from LONGTERM_DATASET_INFO
        when available.  Fall back to 70/10/20 ratios otherwise.
        """
        T = arr.shape[0]

        if "train_end" in meta and "val_end" in meta:
            train_end = int(meta["train_end"])
            val_end   = int(meta["val_end"])
        else:
            train_end = round(T * 0.70)
            val_end   = round(T * 0.80)

        # Clamp to valid range
        train_end = max(1, min(train_end, T - 2))
        val_end   = max(train_end + 1, min(val_end, T - 1))

        log.info(
            f"Long-term split: T={T} "
            f"train=[0:{train_end}] val=[{train_end}:{val_end}] test=[{val_end}:{T}]"
        )
        return [arr[:train_end]], [arr[train_end:val_end]], [arr[val_end:]]

    # ── GluonTS multivariate ───────────────────────────────────────────────

    def _split_gluonts_mv(self, arr_train: np.ndarray, arr_test: np.ndarray, meta: dict):
        """
        Train/test boundary is already encoded in the two grouped arrays.
        Carve a validation slice from the tail of the train array equal to
        one prediction_length.
        """
        val_len = int(meta["prediction_length"])
        if arr_train.shape[0] > val_len:
            train_arr = arr_train[:-val_len]
            val_arr   = arr_train[-val_len:]
        else:
            log.warning(
                f"Train array ({arr_train.shape[0]} steps) shorter than "
                f"prediction_length ({val_len}); validation will be empty."
            )
            train_arr = arr_train
            val_arr   = arr_train[:0]

        log.info(
            f"GluonTS MV split: "
            f"train={train_arr.shape} val={val_arr.shape} test={arr_test.shape}"
        )
        return [train_arr], [val_arr], [arr_test]

    # ── GluonTS univariate ─────────────────────────────────────────────────

    def _split_gluonts_uv(self, train_list, test_list, meta: dict):
        """
        raw.train  → training arrays (minus a held-out val tail)
        raw.test   → test arrays (already include the forecast horizon)
        val        → last prediction_length steps of each training series
        """
        val_len = int(meta["prediction_length"])
        train_out, val_out = [], []

        for arr in train_list:
            T = arr.shape[0]
            eff = min(val_len, T - 1) if T > 1 else 0
            if eff > 0:
                train_out.append(arr[:-eff])
                val_out.append(arr[-eff:])
            else:
                train_out.append(arr)
                val_out.append(arr[:0])

        log.info(
            f"GluonTS UV split: "
            f"{len(train_out)} train / {len(val_out)} val / {len(test_list)} test series"
        )
        return train_out, val_out, test_list

    # ── GIFT eval ──────────────────────────────────────────────────────────

    def _split_gift(self, series_list: List[np.ndarray]):
        """Split each GIFT series 70 / 10 / 20 along the time axis."""
        train_out, val_out, test_out = [], [], []
        for arr in series_list:
            T = arr.shape[0]
            train_end = round(T * 0.70)
            val_end   = round(T * 0.80)
            train_end = max(1, min(train_end, T - 2))
            val_end   = max(train_end + 1, min(val_end, T - 1))
            train_out.append(arr[:train_end])
            val_out.append(arr[train_end:val_end])
            test_out.append(arr[val_end:])

        log.info(f"GIFT split: {len(series_list)} series, ratios 0.70/0.10/0.20")
        return train_out, val_out, test_out


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_probts_dataset(
    dataset: str,
    data_root_path: str | Path,
    context_length: Optional[int] = None,
    prediction_length: Optional[int] = None,
    freq: Optional[str] = None,
    **kwargs,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    One-shot helper. Returns ``(train, val, test)`` for any ProbTS dataset.

    Long-term examples
    ------------------
    >>> train, val, test = load_probts_dataset("etth1", "./datasets")
    >>> train, val, test = load_probts_dataset("etth1", "./datasets", prediction_length=720)

    Short-term examples
    -------------------
    >>> train, val, test = load_probts_dataset("m4_hourly", "./datasets")
    >>> train, val, test = load_probts_dataset("electricity_nips", "./datasets")

    GIFT example
    ------------
    >>> train, val, test = load_probts_dataset("gift/ett1/H/long", "./datasets")
    """
    loader = ProbTSDatasetLoader(
        dataset, data_root_path, context_length, prediction_length, freq, **kwargs
    )
    train, val, test = loader.load()

    if len(train) == 1:
        train = train[0]
    if len(val) == 1:
        val = val[0]
    if len(test) == 1:
        test = test[0]

    return train, val, test