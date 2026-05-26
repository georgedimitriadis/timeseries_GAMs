# ---------------------------------------------------------------------------------
# VMD Decomposition Forecaster for ProbTS
#
# Architecture:
#   1. VMD (Variational Mode Decomposition) splits each feature's history into
#      K oscillatory modes — this is a fixed, non-learnable signal processing step.
#   2. A small CNN encoder maps the K modes of each feature to per-mode amplitude
#      multipliers A[f,d] and phase shifts Phi[f,d].
#   3. A differentiable FFT-based synthesis layer applies A and Phi to each mode,
#      then sums across modes to produce the forecast for each feature.
#
# Gradient flow:
#   VMD has NO gradient (it runs as a numpy op on CPU).
#   Gradients flow through: Encoder → A, Phi → FFT phase shift → output.
#
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, List

from vmdpy import VMD

from probts.model.forecaster import Forecaster


# ─────────────────────────────────────────────────────────────────────────────────
# Building block 1: Differentiable FFT-based phase shift
# ─────────────────────────────────────────────────────────────────────────────────

def fft_phase_shift(x: torch.Tensor, phi: torch.Tensor, T_out: int) -> torch.Tensor:
    """
    Shift signal x by phi samples (fractional), and resample to length T_out.

    Args:
        x   : [N, T_in]   real signals
        phi : [N]         shift in samples (can be fractional and learnable)
        T_out             desired output length

    Returns:
        [N, T_out]

    Notes:
        - irfft(n=T_out) with T_out > T_in  →  zero-pad in freq = periodic extrapolation.
        - irfft(n=T_out) with T_out < T_in  →  low-pass truncation.
        - Both are exact for purely periodic (band-limited) signals, which VMD modes are.
    """
    T_in = x.shape[-1]
    X = torch.fft.rfft(x, n=T_in)                              # [N, T_in//2+1] complex

    # freqs in cycles/sample: [T_in//2+1]
    freqs = torch.fft.rfftfreq(T_in, device=x.device)

    # Shift theorem: shift by phi samples ↔ multiply by e^{-j 2π f phi}
    # phi: [N] → [N, 1]   freqs: [T_in//2+1] → broadcast [N, T_in//2+1]
    phase = torch.exp(
        torch.complex(
            torch.zeros_like(phi.unsqueeze(-1).expand(-1, freqs.shape[0])),
            -2.0 * torch.pi * freqs.unsqueeze(0) * phi.unsqueeze(-1)
        )
    )
    X_shifted = X * phase

    return torch.fft.irfft(X_shifted, n=T_out)                 # [N, T_out]


# ─────────────────────────────────────────────────────────────────────────────────
# Building block 2: Encoder  (VMD modes → A and Phi per mode)
# ─────────────────────────────────────────────────────────────────────────────────

class ModeEncoder(nn.Module):
    """
    Maps one feature's D VMD modes (each of length T_in) to amplitude and phase
    parameters for each mode.

    Input  per (batch × feature):  [B*F, D, T_in]
    Output per (batch × feature):  A [B*F, D],  Phi [B*F, D]

    Architecture: lightweight 1-D CNN over time → global pool → MLP head.
    The CNN captures temporal patterns within each mode; the pool makes it
    length-agnostic (handles variable context_length cleanly).
    """

    def __init__(self, D: int, hidden: int = 64):
        super().__init__()
        self.D = D

        self.cnn = nn.Sequential(
            # Treat D modes as input channels; learn cross-mode features too
            nn.Conv1d(D, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # → [B*F, hidden, 1]
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * D),  # → [A_raw_1…D, Phi_raw_1…D]
        )

    def forward(self, modes: torch.Tensor):
        """modes: [N, D, T_in]  where N = B*F"""
        h = self.cnn(modes).squeeze(-1)   # [N, hidden]
        out = self.head(h)                # [N, 2*D]
        A_raw   = out[:, :self.D]         # [N, D]
        Phi_raw = out[:, self.D:]         # [N, D]
        return A_raw, Phi_raw


# ─────────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────────

class VMDDecompositionForecaster(Forecaster):
    """
    ProbTS forecaster that decomposes each feature's history with VMD, then
    learns per-mode amplitude multipliers and phase shifts to reconstruct the
    forecast horizon.

    Args:
        num_decompositions (int):
            Number of VMD modes K.  More modes = richer decomposition but slower
            preprocessing and more parameters.
        vmd_alpha (float):
            VMD bandwidth constraint.  Larger → narrower modes (less overlap).
            Typical range: 100 – 5000.
        vmd_tau (float):
            VMD noise tolerance.  0 = no noise assumed (strict data fidelity).
        vmd_DC (int):
            1 = include a zero-frequency (DC/trend) mode, 0 = exclude.
        vmd_init (int):
            Initialisation of centre frequencies: 1 = uniform, 2 = random.
        vmd_tol (float):
            Convergence tolerance for VMD iterations.
        encoder_hidden (int):
            Hidden size of the CNN encoder.
        phi_scale (float | None):
            Maximum phase shift in samples: shifts are clamped to
            [-phi_scale, +phi_scale] via tanh.  Defaults to context_length.
        allow_negative_amplitude (bool):
            If True, amplitudes can be negative (mode can flip sign).
            If False, softplus keeps them strictly positive.
    """

    def __init__(
        self,
        num_decompositions: int = 5,
        vmd_alpha: float = 2000.0,
        vmd_tau: float = 0.0,
        vmd_DC: int = 0,
        vmd_init: int = 1,
        vmd_tol: float = 1e-7,
        encoder_hidden: int = 64,
        phi_scale: float = None,
        allow_negative_amplitude: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.D = num_decompositions
        self.vmd_alpha = vmd_alpha
        self.vmd_tau = vmd_tau
        self.vmd_DC = vmd_DC
        self.vmd_init = vmd_init
        self.vmd_tol = vmd_tol
        self.allow_negative_amplitude = allow_negative_amplitude

        # phi bounded to ±phi_scale samples (defaults to full context window)
        self.phi_scale = float(phi_scale or self.max_context_length)

        # Encoder: shared weights across all features
        self.encoder = ModeEncoder(D=self.D, hidden=encoder_hidden)

        self.loss_fn = nn.MSELoss(reduction="none")

        self.loss_call_count = 0
        self._vmd_cache: dict = {}  # hash(input) → decomposed tensor

    # ── VMD decomposition (no gradient) ──────────────────────────────────────────

    def _vmd_decompose(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decompose raw input into VMD modes.

        Args:
            x : [B, T_in, F]   (ProbTS layout: batch × time × features)

        Returns:
            modes : [B, F, D, T_in]   (float32 tensor on same device as x)

        Implementation note:
            VMD (vmdpy) is a numpy routine — we detach, move to CPU, loop over
            (batch, feature) pairs, then move the result back to the original device.
            This is the main runtime bottleneck; see the caching note below.
        """
        print('      Calling _vmd_decompose')

        key = hash(x.cpu().numpy().tobytes())
        if key in self._vmd_cache:
            print(f"      VMD cache HIT  (cache size: {len(self._vmd_cache)})")
            return self._vmd_cache[key].to(x.device)

        print(f"      VMD cache MISS (cache size: {len(self._vmd_cache)})")

        B, T, F = x.shape
        x_np = x.detach().cpu().numpy().astype(np.float64)  # vmdpy expects float64

        modes_np = np.zeros((B, F, self.D, T), dtype=np.float32)

        for b in range(B):
            for f in range(F):
                signal = x_np[b, :, f]
                try:
                    u, _, _ = VMD(
                        signal,
                        self.vmd_alpha,
                        self.vmd_tau,
                        self.D,
                        self.vmd_DC,
                        self.vmd_init,
                        self.vmd_tol,
                    )
                    # u: [D, T]
                    modes_np[b, f, :, :] = u.astype(np.float32)
                except Exception:
                    # Fallback: put the raw signal in mode 0, leave rest as zeros.
                    # Happens on very short or constant signals.
                    modes_np[b, f, 0, :] = signal.astype(np.float32)

        result = torch.tensor(modes_np, device=x.device) # [B, F, D, T]
        self._vmd_cache[key] = result.cpu()  # store on CPU to save GPU memory
        return result

    # ── Differentiable synthesis ──────────────────────────────────────────────────

    def _synthesize(self, modes: torch.Tensor) -> torch.Tensor:
        """
        Predict A and Phi from modes, apply phase shift and scaling, sum over modes.

        Args:
            modes : [B, F, D, T_in]

        Returns:
            output : [B, F, T_out]
        """
        print('      Calling _synthesize')
        B, F, D, T_in = modes.shape
        T_out = self.max_prediction_length

        # ── Encode: run encoder once over all (B × F) feature instances ─────────
        modes_bf = modes.view(B * F, D, T_in)          # [B*F, D, T_in]
        A_raw, Phi_raw = self.encoder(modes_bf)         # each [B*F, D]

        # ── Constrain parameters ─────────────────────────────────────────────────
        if self.allow_negative_amplitude:
            # Unconstrained: modes can add or subtract (good for oscillatory signals)
            A = A_raw
        else:
            # Strictly positive: each mode only adds to the forecast
            A = F.softplus(A_raw)

        # Phi bounded to ±phi_scale via tanh → healthy gradients, no runaway shifts
        Phi = torch.tanh(Phi_raw) * self.phi_scale      # [B*F, D]

        # ── Apply phase shift to every (batch, feature, mode) triple ─────────────
        modes_bfd = modes_bf.reshape(B * F * D, T_in)  # [B*F*D, T_in]
        phi_bfd   = Phi.reshape(B * F * D)             # [B*F*D]

        shifted = fft_phase_shift(modes_bfd, phi_bfd, T_out)  # [B*F*D, T_out]
        shifted = shifted.view(B, F, D, T_out)

        # ── Weighted sum over modes ───────────────────────────────────────────────
        A = A.view(B, F, D)                             # [B, F, D]
        output = (A.unsqueeze(-1) * shifted).sum(dim=2) # [B, F, T_out]
        return output

    # ── Core forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, T_in, F]   scaled input sequence from get_inputs()

        Returns:
            [B, T_out, F]
        """
        print('   Calling forward')
        modes  = self._vmd_decompose(x)          # [B, F, D, T_in]  — no grad
        output = self._synthesize(modes)          # [B, F, T_out]    — grad flows here
        return output.permute(0, 2, 1)            # [B, T_out, F]

    # ── ProbTS interface ──────────────────────────────────────────────────────────

    def forecast(self, batch_data, num_samples=None):
        """
        Called by ProbTS evaluator.  Returns [B, 1, T_out, F] (1 = deterministic).
        """
        if self.use_scaling:
            self.get_scale(batch_data)

        inputs    = self.get_inputs(batch_data, "encode")  # [B, T_in, F]
        forecasts = self(inputs).unsqueeze(1)              # [B, 1, T_out, F]

        return forecasts

    def loss(self, batch_data):
        """
        Called by ProbTS trainer.  Returns scalar MSE loss.
        """
        self.loss_call_count += 1
        print(f'Calling loss {self.loss_call_count}')

        inputs  = self.get_inputs(batch_data, "encode")   # [B, T_in, F]
        outputs = self(inputs)                             # [B, T_out, F]

        target = batch_data.future_target_cdf

        loss = self.loss_fn(target, outputs)               # [B, T_out, F]
        loss = self.get_weighted_loss(batch_data, loss)    # apply observation mask
        result = loss.mean()

        return result


# ─────────────────────────────────────────────────────────────────────────────────
# Usage note
# ─────────────────────────────────────────────────────────────────────────────────
#
# Register in your ProbTS config like any other model:
#
#   model:
#     _target_: path.to.vmd_decomposition_forecaster.VMDDecompositionForecaster
#     num_decompositions: 5
#     vmd_alpha: 2000
#     encoder_hidden: 64
#     phi_scale: null          # defaults to context_length
#     allow_negative_amplitude: true
#
# ── Performance note on VMD ───────────────────────────────────────────────────────
#
# VMD runs on CPU with numpy inside every forward/loss call.  For large batches or
# long sequences this will bottleneck training.  Two mitigations:
#
#   Option A – Offline cache (recommended for production):
#     Run a single-pass script over your dataset to compute and save the VMD modes
#     as a new data field, then skip _vmd_decompose() entirely and read modes
#     directly from batch_data.
#
#   Option B – In-memory LRU cache (quick win for repeated batches):
#     Add a dict self._vmd_cache = {} and key on hash(x.cpu().numpy().tobytes()).
#     Useful when the same context windows are seen across epochs (e.g. few-shot).
#
# ─────────────────────────────────────────────────────────────────────────────────