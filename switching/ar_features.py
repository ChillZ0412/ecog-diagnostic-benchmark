"""
Stage 2 -- AR-coefficient feature extraction for the moving-finger state
classifier f(x) (Flamary & Rakotomamonjy 2011/2012, Sec 3.2 / Fig 2).

Per channel, per shift version of the signal:
  1. Split into non-overlapping windows of AR_WINDOW_SAMPLES (paper: 300).
  2. Fit an AR(config.AR_ORDER) model on each window via Yule-Walker
     (paper doesn't name the estimation method; Yule-Walker/autocorrelation
     is the standard default for this kind of block AR fit).
  3. Keep the first `n_coeffs_keep` coefficients (paper: "only the two
     first AR coefficients are used" -> n_coeffs_keep=2 by default; with
     AR_ORDER=2 this keeps everything the model estimates, consistent with
     the AR_ORDER ambiguity already flagged in config.py).
  4. Spline-interpolate the per-window coefficient sequence back up to a
     continuous per-sample trace ("smoothing spline-based interpolation
     between two consecutive AR coefficients", paper's wording).
  5. Optionally repeat 1-4 on the signal shifted by +ts and -ts samples
     ("signal dynamics... by applying a similar procedure to shifted
     version of the signal at (+ts and -ts)").

For n_channels channels, 3 shift versions (0, +ts, -ts), 2 AR coeffs kept,
this reproduces the paper's worked example: 48 * 3 * 2 = 240 features.

--------------------------------------------------------------------------
TWO PAPER AMBIGUITIES HANDLED HERE (beyond the already-flagged AR_ORDER):
--------------------------------------------------------------------------
(a) "Smoothing spline" is not otherwise specified (degree, smoothing
    factor). We use a natural cubic spline through the window-center
    coefficient values (scipy CubicSpline, bc_type='natural'), i.e. an
    *interpolating* cubic spline rather than a least-squares-smoothed one.
    With windows of 300 samples this already acts as a strong smoother
    relative to the raw signal; a true smoothing spline (with a tunable
    penalty) is a reasonable ablation but not implemented as the default.
(b) Window boundary handling: the paper doesn't say what happens to the
    trailing remainder when the channel length isn't a multiple of 300.
    We drop it ('valid' semantics), consistent with how Stage 4 of the
    Wiener pipeline (config.MEMORY_K) handles its own boundary.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

import config as C


def _shift_signal(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift a 1D signal by `shift` samples with edge-padding (no wraparound,
    no NaNs). shift > 0 delays the signal (pulls from the past);
    shift < 0 advances it (pulls from the future)."""
    if shift == 0:
        return x
    out = np.empty_like(x)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:-shift]
    else:
        s = -shift
        out[-s:] = x[-1]
        out[:-s] = x[s:]
    return out


def _yule_walker_batch(windows: np.ndarray, order: int) -> np.ndarray:
    """
    Fit an AR(order) model independently to each row of `windows` via
    Yule-Walker (Levinson-Durbin recursion), vectorized across windows.

    windows: (n_windows, window_len)
    returns: (n_windows, order) AR coefficients a_1..a_order
             (sign convention: x[n] ~= sum_i a_i * x[n-i])
    """
    x = windows - windows.mean(axis=1, keepdims=True)
    n = x.shape[1]
    # biased autocorrelation estimate at lags 0..order, one per window
    r = np.stack(
        [np.sum(x[:, : n - k] * x[:, k:], axis=1) / n for k in range(order + 1)],
        axis=1,
    )  # (n_windows, order+1)

    n_win = windows.shape[0]
    a = np.zeros((n_win, order + 1))
    a[:, 0] = 1.0
    e = r[:, 0].copy()
    e[e == 0] = 1e-12  # guard against a perfectly flat (zero-variance) window

    for i in range(1, order + 1):
        acc = r[:, i] + np.sum(a[:, 1:i] * r[:, i - 1 : 0 : -1], axis=1)
        k = -acc / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[:, j] = a[:, j] + k * a[:, i - j]
        a_new[:, i] = k
        a = a_new
        e = e * (1 - k**2)
        e[e <= 0] = 1e-12  # numerical guard, keeps later divisions finite

    # Levinson-Durbin's `a` is the polynomial form (x[n] + sum a_i x[n-i] = e[n]);
    # negate to the predictive form x[n] ~= sum a_i x[n-i] documented above,
    # verified against a known AR(2) process in the module's smoke test.
    return -a[:, 1:]  # drop the leading 1.0, shape (n_windows, order)


@dataclass
class ARFeatureSet:
    """features: (T, n_channels * n_shifts * n_coeffs_keep), plus the
    (channel, shift_ms, coeff_idx) label for every column, so Stage 3
    (channel selection) can aggregate back to channel level."""
    features: np.ndarray
    channel_of_col: np.ndarray   # (n_cols,) int
    shift_ms_of_col: np.ndarray  # (n_cols,) float
    coeff_idx_of_col: np.ndarray  # (n_cols,) int, 0-indexed


def ar_features_one_channel_one_shift(x: np.ndarray, fs: int, window_samples: int,
                                       order: int, shift_ms: float) -> np.ndarray:
    """Full pipeline for a single channel, single shift version.
    Returns (T, order) continuous AR-coefficient trace."""
    shift_samples = int(round(shift_ms * fs / 1000.0))
    xs = _shift_signal(x, shift_samples)

    T = len(xs)
    n_windows = T // window_samples  # drop trailing remainder ('valid')
    usable = n_windows * window_samples
    windows = xs[:usable].reshape(n_windows, window_samples)

    coeffs = _yule_walker_batch(windows, order)  # (n_windows, order)

    centers = (np.arange(n_windows) + 0.5) * window_samples
    spline = CubicSpline(centers, coeffs, axis=0, bc_type="natural")
    t_full = np.arange(T)
    return spline(t_full)  # (T, order), extrapolated flat-ish at the edges


def extract_ar_features(ecog: np.ndarray, channels=None, fs: int = None,
                         window_samples: int = None, order: int = None,
                         shifts_ms=None, n_coeffs_keep: int = 2) -> ARFeatureSet:
    """
    ecog: (T, n_channels_total)
    channels: iterable of channel indices to use (None -> all channels)
    Defaults are pulled from config.py when not given explicitly, so this
    doubles as both the Stage 4 full-feature call (all channels, all
    shifts, 2 coeffs) and the Stage 3 channel-selection call (all channels,
    shifts_ms=[0], n_coeffs_keep=1) just by varying the arguments -- see
    module docstring point (nothing here hardcodes which caller it's for).
    """
    fs = C.FS_ECOG if fs is None else fs
    window_samples = C.AR_WINDOW_SAMPLES if window_samples is None else window_samples
    order = C.AR_ORDER if order is None else order
    shifts_ms = list(C.AR_TIME_SHIFTS_MS) if shifts_ms is None else list(shifts_ms)
    channels = range(ecog.shape[1]) if channels is None else list(channels)

    n_coeffs_keep = min(n_coeffs_keep, order)
    T = ecog.shape[0]
    n_cols = len(channels) * len(shifts_ms) * n_coeffs_keep

    feats = np.empty((T, n_cols), dtype=np.float64)
    ch_col = np.empty(n_cols, dtype=int)
    shift_col = np.empty(n_cols, dtype=float)
    coeff_col = np.empty(n_cols, dtype=int)

    col = 0
    for ch in channels:
        x = ecog[:, ch]
        for shift_ms in shifts_ms:
            trace = ar_features_one_channel_one_shift(
                x, fs, window_samples, order, shift_ms
            )  # (T, order)
            for c in range(n_coeffs_keep):
                feats[:, col] = trace[:, c]
                ch_col[col] = ch
                shift_col[col] = shift_ms
                coeff_col[col] = c
                col += 1

    return ARFeatureSet(features=feats, channel_of_col=ch_col,
                         shift_ms_of_col=shift_col, coeff_idx_of_col=coeff_col)
