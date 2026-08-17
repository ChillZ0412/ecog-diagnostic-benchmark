"""
Stage 3 — Amplitude-modulation features + target downsampling
(Liang & Bougrain 2012, Section 2.2.1.2).

The AM feature, straight from the paper's equation (1):

    x(t_n) = SUM over the window of v(t)^2

with a non-overlapping window of dt = 40 ms. At fs = 1000 Hz that is exactly
40 samples per window, so the feature rate becomes 1000/40 = 25 Hz -- which is
the dataglove's native rate. That is the whole point of choosing 40 ms.

--------------------------------------------------------------------------
THE ALIGNMENT PROBLEM
--------------------------------------------------------------------------
The glove is STORED at 1000 Hz but was recorded at 25 Hz and upsampled. To get
back to 25 Hz we must pick one value per 40-sample block, and the correct
choice depends on how the upsampling was done:

  * zero-order hold (each 25 Hz sample repeated 40x)
        -> the block is constant, so first/mean/last/centre are IDENTICAL
           and the choice does not matter at all
  * interpolated
        -> the four choices differ, and picking wrongly introduces a sub-window
           timing error against the ECoG features

`describe_glove_structure()` measures this directly rather than assuming it:
it reports the ratio of within-block std to overall std. A ratio of ~0 means
zero-order hold. Always run it once per dataset before trusting the features.

--------------------------------------------------------------------------
SCALE / CONDITIONING WARNING (matters in Stage 5)
--------------------------------------------------------------------------
Raw ECoG here has std ~ 4e3 ADC units. Squaring gives ~1.8e7, and summing 40
samples gives AM features of order 1e9. Stacking k=25 lags and forming X^T X
would produce entries of order 1e22 -- far beyond float64's ~1e-16 relative
precision, so a naive normal-equation solve is numerically hopeless.

Two consequences, both handled later:
  * Stage 5 must solve via SVD/`pinv(X) @ d`, never by building X^T X.
  * `standardize=True` (optional here) z-scores each feature. In exact
    arithmetic this does not change a linear model's predictions, but it
    massively improves conditioning. The paper does not mention it; it is
    kept OFF by default for fidelity and exposed as an ablation.

--------------------------------------------------------------------------
FEATURE LAYOUT
--------------------------------------------------------------------------
Features are band-major: all channels of 'sub', then 'gamma', then
'fastgamma'. For subject 1 that is 3 x 62 = 186 features, matching the
paper's stated count. `feature_names` and `paper_feature_id()` let you compare
selected features against the paper's Figure 3 notation (channel, band) with
band 1=sub, 2=gamma, 3=fastgamma and MATLAB 1-based channel numbering.
"""
from typing import List, Tuple

import numpy as np

import config as C
from filters import decompose_bands


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def describe_glove_structure(glove: np.ndarray, win: int = None) -> dict:
    """
    Determine empirically whether the glove is zero-order-held at 25 Hz.

    Returns a dict with `within_block_std`, `overall_std`, `ratio`, and
    `zero_order_hold` (True if the signal is constant within each block).
    """
    win = C.AM_WINDOW_SAMPLES if win is None else win
    n = (glove.shape[0] // win) * win
    blocks = glove[:n].reshape(-1, win, glove.shape[1])

    within = float(blocks.std(axis=1).mean())
    overall = float(glove.std(axis=0).mean())
    ratio = within / overall if overall > 0 else float("nan")

    first, mean, last = blocks[:, 0, :], blocks.mean(axis=1), blocks[:, -1, :]
    return {
        "within_block_std": within,
        "overall_std": overall,
        "ratio": ratio,
        "zero_order_hold": ratio < 1e-6,
        "max_first_minus_mean": float(np.abs(first - mean).max()),
        "max_first_minus_last": float(np.abs(first - last).max()),
        "n_blocks": blocks.shape[0],
    }


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def amplitude_modulation(band_signal: np.ndarray, win: int = None) -> np.ndarray:
    """
    Paper eq. (1): sum of squared voltage over non-overlapping windows.

    band_signal : (T, n_ch)  ->  returns (T // win, n_ch)
    """
    win = C.AM_WINDOW_SAMPLES if win is None else win
    n = (band_signal.shape[0] // win) * win
    x = band_signal[:n].astype(np.float64)
    return (x ** 2).reshape(-1, win, x.shape[1]).sum(axis=1)


def extract_am_features(ecog: np.ndarray,
                        standardize: bool = False,
                        causal: bool = False,
                        apply_notch: bool = None,
                        return_stats: bool = False):
    """
    Full Stage 2 + 3: band decomposition -> AM features at 25 Hz.

    ecog : (T, n_ch) at 1000 Hz
    returns X : (T // 40, n_ch * n_bands), feature_names : list[(band, ch)]

    Bands are computed one at a time and immediately reduced to 25 Hz, so peak
    memory stays near a single band rather than three.
    """
    n_ch = ecog.shape[1]
    cols, names = [], []

    for band_name, filtered in decompose_bands(ecog, causal=causal,
                                               apply_notch=apply_notch):
        cols.append(amplitude_modulation(filtered))
        names.extend((band_name, ch) for ch in range(n_ch))
        del filtered

    X = np.concatenate(cols, axis=1)

    stats = None
    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd
        stats = (mu, sd)

    if return_stats:
        return X, names, stats
    return X, names


def downsample_target(glove: np.ndarray,
                      method: str = None,
                      win: int = None) -> np.ndarray:
    """
    Reduce the 1000 Hz glove trace to one value per 40 ms block (i.e. 25 Hz).

    method: 'first' | 'mean' | 'last' | 'center'
    If the glove is zero-order-held (the usual case) all methods agree exactly.
    """
    method = C.TARGET_DOWNSAMPLE if method is None else method
    win = C.AM_WINDOW_SAMPLES if win is None else win

    n = (glove.shape[0] // win) * win
    blocks = glove[:n].astype(np.float64).reshape(-1, win, glove.shape[1])

    if method == "first":
        return blocks[:, 0, :]
    if method == "last":
        return blocks[:, -1, :]
    if method == "center":
        return blocks[:, win // 2, :]
    if method == "mean":
        return blocks.mean(axis=1)
    raise ValueError(f"unknown downsample method: {method!r}")


def align(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Trim features and targets to a common number of 25 Hz samples."""
    n = min(X.shape[0], y.shape[0])
    return X[:n], y[:n]


# ---------------------------------------------------------------------------
# Naming helpers (for comparing against the paper's Figure 3)
# ---------------------------------------------------------------------------
_BAND_TO_PAPER_ID = {"sub": 1, "gamma": 2, "fastgamma": 3}


def paper_feature_id(name: Tuple[str, int]) -> Tuple[int, int]:
    """
    Convert ('fastgamma', 0) -> (1, 3), i.e. the paper's (channel, band)
    notation with 1-based MATLAB channel numbering and band 1/2/3.
    """
    band, ch = name
    return (ch + 1, _BAND_TO_PAPER_ID[band])


def format_features(names: List[Tuple[str, int]], idx) -> str:
    """Pretty-print selected feature indices in the paper's notation."""
    return " ".join(f"({c},{b})" for c, b in (paper_feature_id(names[i]) for i in idx))
