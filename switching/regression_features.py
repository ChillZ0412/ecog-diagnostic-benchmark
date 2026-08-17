"""
Stage 5 -- Regression features for H_k, the per-state finger-flexion
regressors (Flamary & Rakotomamonjy 2011/2012, Sec 3.3).

Paper: "we use filtered time-samples as features... All channels have been
filtered with a Savitsky-Golay (third order, 0.4s width) low-pass filter.
Then, x_t is composed of the concatenation of the time samples at t, t-tau
and t+tau for all smoothed signals at all channels."

So for n_channels channels this produces a 3*n_channels-dim feature vector
per valid time t (three temporal taps, each n_channels wide).

--------------------------------------------------------------------------
ONE MORE PAPER GAP HANDLED HERE
--------------------------------------------------------------------------
config.SG_WINDOW_SEC=0.4 at FS_ECOG=1000Hz is exactly 400 samples, but
scipy.signal.savgol_filter requires an ODD window length. The paper doesn't
address this (its window width is presumably fine at whatever their
implementation's default rounding did). We round up to 401 samples --
documented in `_sg_window_samples()` rather than silently choosing 399,
so it's easy to find if this ever needs to be revisited.
--------------------------------------------------------------------------
BOUNDARY HANDLING
--------------------------------------------------------------------------
Samples without a full [t-tau, t+tau] window (the first and last `tau`
samples) are dropped ('valid' semantics), matching the convention used
elsewhere in this project (Wiener Stage 4's memory stack, Stage 2's AR
window remainder). `build_regression_features` returns the valid index
range so the caller can slice the target (finger flexion) to match.
"""
import numpy as np
from scipy.signal import savgol_filter

import config as C


def _sg_window_samples(fs: int, window_sec: float) -> int:
    n = int(round(window_sec * fs))
    if n % 2 == 0:
        n += 1  # savgol_filter requires an odd window length
    return n


def savgol_smooth_all_channels(ecog: np.ndarray, fs: int = None,
                                window_sec: float = None,
                                polyorder: int = None) -> np.ndarray:
    """ecog: (T, n_channels). Returns smoothed array of the same shape."""
    fs = C.FS_ECOG if fs is None else fs
    window_sec = C.SG_WINDOW_SEC if window_sec is None else window_sec
    polyorder = C.SG_POLYORDER if polyorder is None else polyorder
    window_samples = _sg_window_samples(fs, window_sec)
    return savgol_filter(ecog, window_length=window_samples,
                          polyorder=polyorder, axis=0)


def build_regression_features(ecog_smoothed: np.ndarray, tau_samples: int,
                               channels=None):
    """
    ecog_smoothed: (T, n_channels_total), already Savitzky-Golay filtered.
    tau_samples: the tau lag/lead, in samples (paper's tau, tuned by
                 validation -- config.TAU_MS is None until that's done).
    channels: which channels to include (None -> all).

    Returns (X, (lo, hi)):
      X: (hi-lo, n_channels_used * 3), columns ordered
         [all channels @ t-tau, all channels @ t, all channels @ t+tau]
      (lo, hi): the original time-index range these rows correspond to,
                so the caller can slice the target the same way, e.g.
                y_aligned = finger_flexion[lo:hi]
    """
    channels = range(ecog_smoothed.shape[1]) if channels is None else list(channels)
    sig = ecog_smoothed[:, channels]
    T = sig.shape[0]

    if tau_samples == 0:
        # degenerate case (tau=0): still return 3*n_channels dims so
        # downstream code doesn't need a special case, but note the three
        # taps are then identical (t-0 == t == t+0).
        X = np.concatenate([sig, sig, sig], axis=1)
        return X, (0, T)

    lo, hi = tau_samples, T - tau_samples
    x_minus = sig[0: T - 2 * tau_samples]      # value at t - tau
    x_t = sig[tau_samples: T - tau_samples]    # value at t
    x_plus = sig[2 * tau_samples: T]           # value at t + tau
    X = np.concatenate([x_minus, x_t, x_plus], axis=1)
    return X, (lo, hi)
