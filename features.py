"""
Feature extraction pipelines for ECoG finger movement decoding.

Two pipelines:
    1. Spectral (Welch PSD → 6-band powers) — LightGBM / SpectralSVM input
    2. CSP      (bandpass + CSP spatial filter + log-variance) — LDA input
"""

import time
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal as sg

from .config import CSPConfig, FeatureConfig


# ═════════════════════════════════════════════════════════════════════
# 1. Spectral feature extraction
# ═════════════════════════════════════════════════════════════════════

def extract_spectral_features(
    ecog: np.ndarray,
    cfg: FeatureConfig,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 6-band spectral power features via Welch's method.

    For each sliding window (250 ms, step 40 ms), compute band power for
    6 frequency bands on every channel.

    Parameters
    ----------
    ecog : np.ndarray, shape (n_channels, n_time)
        Raw ECoG signal at 1000 Hz.
    cfg : FeatureConfig
        Feature extraction hyperparameters.
    verbose : bool
        If True, print extraction progress and timing.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_channels × n_bands)
        Spectral features — one row per sliding window.
    t_window : np.ndarray, shape (n_windows,)
        Start index of each window in original time (1000 Hz samples),
        for label alignment.

    Notes
    -----
    Bands: LMP(0.3–4), Alpha(8–13), Beta(13–30), Low-Gamma(30–55),
           High-Gamma1(65–115), High-Gamma2(125–175) Hz.
    """
    n_ch, n_time = ecog.shape
    win_samples = cfg.window_samples
    step_samples = cfg.step_samples
    bands = list(cfg.spectral_bands.values())  # list of (lo, hi)
    n_bands = len(bands)
    fs = cfg.fs

    n_windows = (n_time - win_samples) // step_samples
    n_features = n_ch * n_bands
    X = np.zeros((n_windows, n_features), dtype=np.float32)
    t_start = np.arange(0, n_windows * step_samples, step_samples)

    if verbose:
        t0 = time.time()
        print(f"  Extracting spectral features: {n_windows} windows "
              f"× {n_ch} channels × {n_bands} bands ...")

    for wi in range(n_windows):
        start = wi * step_samples
        segment = ecog[:, start:start + win_samples]          # (n_ch, win_samples)

        # Welch PSD on all channels simultaneously
        freqs, psd = sg.welch(segment, fs=fs, nperseg=win_samples, axis=1)

        # Sum power in each frequency band
        for bi, (lo, hi) in enumerate(bands):
            mask = (freqs >= lo) & (freqs <= hi)
            X[wi, bi::n_bands] = np.sum(psd[:, mask], axis=1)  # (n_ch,)

    if verbose:
        elapsed = time.time() - t0
        print(f"  Done: {elapsed:.1f}s  →  X.shape = {X.shape}")
        print(f"  Feature rate: {n_windows / elapsed:.0f} windows/s")

    return X, t_start


# ═════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════
# 2. FIR bandpass filter (shared utility for CSP)
# ═════════════════════════════════════════════════════════════════════

def _design_bandpass(lo: float, hi: float, fs: int, numtaps: int = 201) -> np.ndarray:
    """Design a zero-phase FIR bandpass filter (linear phase, no distortion).

    Used by CSP feature extraction to isolate the high-gamma band before
    spatial filtering.

    Parameters
    ----------
    lo, hi : float
        Passband cutoff frequencies in Hz.
    fs : int
        Sampling rate in Hz.
    numtaps : int
        Filter length (odd number for Type I linear-phase filter).

    Returns
    -------
    b : np.ndarray, shape (numtaps,)
        FIR filter coefficients.
    """
    return sg.firwin(
        numtaps, [lo, hi], pass_zero=False, fs=fs
    )


# 3. CSP feature extraction  (Blankertz et al., 2008)
# ═════════════════════════════════════════════════════════════════════

def _cov_regularized(X: np.ndarray, reg: float = 0.1) -> np.ndarray:
    """Regularized sample covariance matrix.

    C_reg = (1 − reg)·(XXᵀ/N) + reg·avg(diag)·I

    Parameters
    ----------
    X : np.ndarray, shape (n_channels, n_samples)
    reg : float
        Regularization weight toward identity.

    Returns
    -------
    C : np.ndarray, shape (n_channels, n_channels)
    """
    n_ch, n_t = X.shape
    C_raw = (X @ X.T) / n_t
    trace_avg = np.trace(C_raw) / n_ch
    return (1 - reg) * C_raw + reg * trace_avg * np.eye(n_ch)


def _csp_fit_one(
    X_class1: np.ndarray,    # (n_trials, n_ch, n_time) for class 1
    X_class2: np.ndarray,    # (n_trials, n_ch, n_time) for class 2
    n_components: int = 3,
    cov_reg: float = 0.1,
) -> np.ndarray:
    """Fit CSP spatial filters for a single one-vs-rest binary problem.

    Solves the generalised eigenvalue problem:
        C₁ W = λ (C₁ + C₂) W

    Eigenvalues λ lie in [0, 1]: λ → 1 marks spatial filters that maximise
    class-1 variance while minimising class-2 variance, and λ → 0 the reverse.
    The n_components filters with the LARGEST λ (class-1 side) and the
    n_components with the SMALLEST λ (class-2 side) are therefore returned.

    Parameters
    ----------
    X_class1 : np.ndarray, shape (n_trials_1, n_ch, n_time)
    X_class2 : np.ndarray, shape (n_trials_2, n_ch, n_time)
    n_components : int
        Number of CSP component pairs (output dim = 2·n_components).
    cov_reg : float
        Regularization for covariance estimation.

    Returns
    -------
    W_csp : np.ndarray, shape (n_ch, 2·n_components)
        Selected CSP spatial filters.
    """
    n_ch = X_class1.shape[1]
    n_t1, n_t2 = len(X_class1), len(X_class2)

    # ── Vectorized covariance: batch compute all trials at once ──
    # For each trial: C = (X @ X.T) / n_time
    # Stack trials: (n_trials, n_ch, n_time) → einsum('tci,tcj->cij')
    def _batch_cov(X_trials, reg):
        n_tr = len(X_trials)
        C_sum = np.zeros((n_ch, n_ch), dtype=np.float64)
        for i in range(n_tr):
            X_i = X_trials[i]  # (n_ch, n_time)
            C_raw = (X_i @ X_i.T) / X_i.shape[1]
            trace_avg = np.trace(C_raw) / n_ch
            C_sum += (1 - reg) * C_raw + reg * trace_avg * np.eye(n_ch)
        return C_sum / n_tr

    C1 = _batch_cov(X_class1, cov_reg)
    C2 = _batch_cov(X_class2, cov_reg)

    # Generalised eigenvalue decomposition: C1 W = λ (C1 + C2) W
    C_sum = C1 + C2
    # Use scipy.linalg.eigh for generalized EV (numpy 2.x removed this)
    from scipy.linalg import eigh as gen_eigh
    eigvals, eigvecs = gen_eigh(C1, C_sum + 1e-10 * np.eye(n_ch))

    # Sort by λ descending (λ=1 → class-1 discriminative, λ=0 → class-2)
    idx = np.argsort(-eigvals)
    eigvecs = eigvecs[:, idx]

    # Select n_components from each side of the spectrum
    n_select = min(n_components, n_ch // 2)
    W_csp = np.hstack([
        eigvecs[:, :n_select],        # largest λ → class 1
        eigvecs[:, -n_select:],       # smallest λ → class 2
    ])

    return W_csp


def extract_csp_features(
    ecog: np.ndarray,
    y: np.ndarray,
    cfg: CSPConfig,
    feat_cfg: FeatureConfig,
    verbose: bool = False,
    csp_filters: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """Extract CSP + log-variance features (Blankertz et al., 2008).

    Supports two modes:
    — **Train mode** (csp_filters=None): fit CSP on ``(ecog, y)``;
      returns (X_csp, t_window, csp_filters).
    — **Test mode**  (csp_filters provided): apply pre-fitted CSP filters
      to ``ecog``; ``y`` must still be provided for window count alignment
      but is NOT used for fitting. Returns (X_csp, t_window, None).

    Pipeline:
        1. Bandpass filter (65–175 Hz high-gamma band) on raw ECoG.
        2. Sliding-window segmentation into "trials" (250 ms, 40 ms step).
        3. For each class (One-vs-Rest): fit CSP (train) / apply CSP (test).
        4. Apply each CSP filter bank → compute log-variance per component.
        5. Concatenate features from all OvR problems → (n_windows, n_classes × 2·nc).

    Parameters
    ----------
    ecog : np.ndarray, shape (n_channels, n_time)
        Raw ECoG at 1000 Hz.
    y : np.ndarray, shape (n_windows,)
        Class labels per window (used for fitting CSP in train mode;
        in test mode, only length matters for window count).
    cfg : CSPConfig
    feat_cfg : FeatureConfig
    verbose : bool
    csp_filters : list of np.ndarray, optional
        Pre-fitted CSP spatial filters (one per class). If None, CSP is
        fitted from scratch (train mode).

    Returns
    -------
    X_csp : np.ndarray, shape (n_windows, n_classes × 2·n_components)
    t_window : np.ndarray, shape (n_windows,)
    csp_filters_out : list of np.ndarray or None
        CSP filters (train mode) or None (test mode).
    """
    n_ch, n_time = ecog.shape
    fs = feat_cfg.fs
    win_samples = int(cfg.window_ms * fs / 1000)      # CSP-specific window
    step_samples = int(cfg.step_ms * fs / 1000)        # stride (shared with FeatureConfig)
    n_comp = cfg.n_components
    n_windows = (n_time - win_samples) // step_samples
    n_actual = min(n_windows, len(y))
    is_train = (csp_filters is None)

    if verbose:
        mode = "Fit+Extract" if is_train else "Apply only"
        t0 = time.time()
        print(f"  Extracting CSP features ({mode}): {n_actual} windows "
              f"× {n_ch} ch, freq {cfg.freq_band}, {n_comp} comps/class ...")

    # ── Step 1: Bandpass filter ──
    b = _design_bandpass(cfg.freq_band[0], cfg.freq_band[1], fs, numtaps=201)
    ecog_filt = sg.filtfilt(b, [1.0], ecog, axis=1)

    # ── Step 2: Segment into windows ──
    segments = np.zeros((n_actual, n_ch, win_samples), dtype=np.float64)
    for wi in range(n_actual):
        start = wi * step_samples
        segments[wi] = ecog_filt[:, start:start + win_samples]

    # ── Determine number of classes ──
    if is_train:
        n_classes = len(np.unique(y[:n_actual]))
    else:
        n_classes = len(csp_filters)

    # ── Step 3-5: OvR CSP + log-variance ──
    features_per_class = []
    csp_filters_out = [] if is_train else None

    for ci in range(n_classes):
        if is_train:
            # ── Fit CSP on training data ──
            mask_ci = (y[:n_actual] == ci)
            mask_other = (y[:n_actual] != ci)

            if mask_ci.sum() < 2 or mask_other.sum() < 2:
                W_csp = np.zeros((n_ch, 2 * n_comp), dtype=np.float64)
            else:
                W_csp = _csp_fit_one(
                    segments[mask_ci], segments[mask_other],
                    n_components=n_comp,
                )
            csp_filters_out.append(W_csp)
        else:
            # ── Apply pre-fitted CSP filters ──
            W_csp = csp_filters[ci]

        # ── Vectorized log-variance: project all windows at once ──
        # segments: (n_windows, n_ch, win_samples)
        # W_csp:   (n_ch, n_filters)
        # proj:    (n_windows, n_filters, win_samples) via einsum
        proj = np.einsum('cf,wct->wft', W_csp, segments)
        var = np.var(proj, axis=2)                          # (n_windows, n_filters)
        feats_ci = np.log(var + 1e-10).astype(np.float32)   # log transform

        features_per_class.append(feats_ci)

    X_csp = np.hstack(features_per_class).astype(np.float32)
    t_window = np.arange(n_actual) * step_samples

    if verbose:
        elapsed = time.time() - t0
        print(f"  Done: {elapsed:.1f}s  →  X.shape = {X_csp.shape}")

    return X_csp, t_window, csp_filters_out
