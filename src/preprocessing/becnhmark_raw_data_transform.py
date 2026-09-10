from __future__ import annotations
import numpy as np
from typing import List, Optional
# ─────────────────────────────────────────────────────────────────────────────
# Standalone transform — mirrors Forecaster.get_inputs(batch_data, 'encode')
# ─────────────────────────────────────────────────────────────────────────────

def raw_data_transform(
    window: np.ndarray,                        # [history_length, F]
    context_length: int,
    lags_list: List[int],
    use_lags: bool,
    use_scaling: bool = False,
    use_feat_idx_emb: bool = False,
    feat_idx_emb_weights: Optional[np.ndarray] = None,  # [F, emb_dim]
    use_time_feat: bool = False,
    time_feat_slice: Optional[np.ndarray] = None,        # [context_length, D]
) -> np.ndarray:
    """
    Replicates Forecaster.get_inputs(batch_data, 'encode') on a single raw
    window (no batch dimension).

    Parameters
    ----------
    window : [history_length, F]
        Raw values exactly as the InstanceSplitter would place in
        past_target_cdf (history_length = context_length + max(lags_list)).
    context_length : int
        Number of time steps in the model's look-back window.
    lags_list : list[int]
        Lag indices, e.g. [1, 24, 168] for hourly data.
    use_lags : bool
        Whether to apply lag extraction (mirrors Forecaster.use_lags).
    use_scaling : bool
        Whether to apply TemporalScaler normalisation (mirrors
        Forecaster.use_scaling).  Each feature is divided by its mean
        absolute value over the context window — same as TemporalScaler.
    use_feat_idx_emb : bool
        Whether to append feature-index embeddings.
    feat_idx_emb_weights : [F, emb_dim], optional
        Frozen embedding matrix extracted from self.feat_idx_emb.weight.
        Required when use_feat_idx_emb=True.
    use_time_feat : bool
        Whether to append Fourier time features.
    time_feat_slice : [context_length, D], optional
        Pre-computed time features for the context window.
        Required when use_time_feat=True.

    Returns
    -------
    np.ndarray  shape [context_length, input_size]
        Ready to be flattened and fed to a sklearn model.
    """
    parts: List[np.ndarray] = []

    # ── 0. Temporal scale (mirrors TemporalScaler.fit + Forecaster.get_scale) ─
    # scale[f] = mean(|context_window[:, f]|) — per-feature, same formula
    # as GluonTS TemporalScaler (no observation mask needed for dense data).
    if use_scaling:
        ctx = window[-context_length:, :]                          # [ctx, F]
        scale = np.abs(ctx).mean(axis=0, keepdims=True).clip(1e-8) # [1, F]
    else:
        scale = None

    # ── 1. Main sequence (mirrors get_input_sequence, mode='encode') ─────────
    if use_lags:
        lagged = []
        for lag in lags_list:
            begin = -lag - context_length
            end = -lag if lag > 0 else None
            lv = window[begin:end, :]  # [ctx, F]
            lagged.append(lv / scale if scale is not None else lv)
        seq = np.concatenate(lagged, axis=0)  # [ctx*n_lags, F]
    else:
        seq = window[-context_length:, :]  # [ctx, F]
        if scale is not None:
            seq = seq / scale
    parts.append(seq)

    # ── 2. Feature-index embedding (mirrors get_input_feat_idx_emb) ──────────
    # The embedding is frozen (no gradient updates for sklearn), so we use
    # whatever weights the nn.Embedding was initialised with.
    if use_feat_idx_emb:
        if feat_idx_emb_weights is None:
            raise ValueError(
                "feat_idx_emb_weights must be provided when use_feat_idx_emb=True. "
                "Pass self.feat_idx_emb.weight.detach().cpu().numpy()."
            )
        # weights: [F, D] → flatten to [1, F*D] → tile to [ctx, F*D]
        emb_flat = feat_idx_emb_weights.reshape(1, -1)
        parts.append(np.tile(emb_flat, (context_length, 1)))  # [ctx, F*D]

    # ── 3. Time features (mirrors get_input_time_feat, mode='encode') ────────
    if use_time_feat:
        if time_feat_slice is None:
            raise ValueError(
                "time_feat_slice must be provided when use_time_feat=True."
            )
        parts.append(time_feat_slice)  # [ctx, D]

    return np.concatenate(parts, axis=-1).astype(np.float32)  # [ctx, input_size]
