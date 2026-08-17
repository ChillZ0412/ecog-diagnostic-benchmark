"""
Stage 6 -- Per-state ridge regression H_k (Flamary & Rakotomamonjy 2011/2012,
Sec 3.3 "Learning finger flexion models", Eq. 3).

For each state k (1..N_STATES: 5 fingers + rest -- the paper's own Table 1
reports sample counts for k=1..6 INCLUDING rest, so H_6/rest is trained the
same way as the 5 finger models), fit:

    min_{H_k}  ||Y_k - X_k H_k||_F^2 + lambda_k ||H_k||_F^2

using only the samples where state==k, extracted via a boolean mask applied
AFTER computing the continuous t-tau/t/t+tau features (Stage 5) -- masking
before would let the tau-taps silently reach across a state boundary into a
different finger's segment.

Y_k: (n_k, N_FINGERS) -- ALL 5 finger positions, not just finger k. Even
the k-th finger's own model predicts every finger's flexion simultaneously
(this matches the paper, and matches the small cross-talk between fingers
already built into synthetic_blocks.py).

Feature pruning (paper): after an initial fit, keep the M features with the
largest sum_j |H_k[i,j]| (summed absolute weight across the 5 finger
outputs), then refit restricted to just those M. M is tuned by validation
(config.H_FEATURE_PRUNE_M is None until that's done).

Ridge is solved via SVD rather than the normal equations (X^T X), for the
same numerical-stability reason flagged in the Wiener pipeline's own
Stage 5: t-tau/t/t+tau taps of a low-pass-smoothed signal are highly
collinear, so X^T X can be poorly conditioned.
"""
from dataclasses import dataclass

import numpy as np

import config as C


def extract_state_segment(X_full: np.ndarray, y_full_aligned: np.ndarray,
                           state_aligned: np.ndarray, k: int):
    """
    X_full: (T', d) regression features on the valid [lo,hi) timeline
             (Stage 5's build_regression_features output).
    y_full_aligned: (T', N_FINGERS) target, already sliced to the same [lo,hi).
    state_aligned: (T',) state labels, already sliced to the same [lo,hi).
    Returns (X_k, Y_k) restricted to samples where state_aligned == k.
    """
    mask = state_aligned == k
    return X_full[mask], y_full_aligned[mask]


def fit_ridge_svd(X: np.ndarray, Y: np.ndarray, lambda_k: float) -> np.ndarray:
    """
    Closed-form ridge regression via SVD (numerically stable regardless of
    X's conditioning): H = V diag(s/(s^2+lambda)) U^T Y.
    X: (n, d), Y: (n, m) -> H: (d, m)
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    shrink = s / (s**2 + lambda_k)
    H = (Vt.T * shrink) @ (U.T @ Y)
    return H


def prune_features(H: np.ndarray, M: int) -> np.ndarray:
    """Indices of the M features with the largest sum_j |H[i,j]|."""
    importance = np.sum(np.abs(H), axis=1)
    M = min(M, H.shape[0])
    return np.argsort(importance)[::-1][:M]


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation, paper's evaluation metric. Returns nan if either
    side has zero variance (undefined correlation), rather than raising."""
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard R^2 = 1 - SS_res/SS_tot. Unlike Pearson r, this is sensitive
    to scale/offset errors -- a prediction that's shaped right but scaled
    or shifted wrong will show a normal-looking r but a poor (even very
    negative) R^2. Matches the diagnostic role R^2 plays in the Wiener
    pipeline's summary doc (catches catastrophic divergence that r alone
    can mask).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def calibrated_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 after the OPTIMAL affine rescaling of y_pred (least-squares fit of
    y_pred_calibrated = a*y_pred + b to y_true). Mathematically equals
    pearson_r(y_true, y_pred)**2 for a single-variable affine fit -- but
    computed explicitly here (rather than just squaring pearson_r) to
    mirror the Wiener pipeline's own calibrated_r2 implementation exactly,
    so the two are directly comparable methodologically.

    Isolates "is the predicted SHAPE right" from "is the predicted SCALE
    right": a large gap between r2_score and calibrated_r2 means the model
    has the right shape but wrong scale/offset; both being poor means the
    shape itself is wrong (a more fundamental failure).
    """
    if y_pred.std() == 0:
        return float("nan")
    a, b = np.polyfit(y_pred, y_true, deg=1)
    y_pred_calibrated = a * y_pred + b
    return r2_score(y_true, y_pred_calibrated)


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error, same units as y_true (this pipeline's dataglove
    target, loaded via data_io.load_subject_clean() with zero transforms
    applied to the glove channel -- only ecog channels are touched for bad-
    channel removal). Lower is better, 0 is perfect prediction.

    Units note (verified 2026-08-04): FingerFlex's MinMaxScaler-normalized
    predictions were independently inverse-transformed and cross-validated
    against Wiener's raw data_io.load_subject() glove range (e.g. S1
    Middle: both sides give [-0.9527, +7.5789]) -- confirming this
    pipeline's glove target (identical data_io.py loading logic to
    Wiener's) shares the same original units. MAE computed here is
    directly comparable to Wiener's and FingerFlex's MAE, no conversion
    needed.

    Unlike r2_score/calibrated_r2, MAE is symmetric in its two arguments
    (|y_true - y_pred| == |y_pred - y_true|), so argument order here
    doesn't affect correctness -- kept as (y_true, y_pred) purely for
    consistency with this file's other metric functions.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


@dataclass
class FingerRegressor:
    """One H_k model for state k, with optional feature pruning."""
    state: int
    lambda_k: float
    H: np.ndarray = None                   # (d_used, N_FINGERS)
    selected_features: np.ndarray = None   # indices into the ORIGINAL feature
                                            # space, or None if no pruning was done

    def fit(self, X: np.ndarray, Y: np.ndarray, M: int = None) -> "FingerRegressor":
        if M is None:
            self.H = fit_ridge_svd(X, Y, self.lambda_k)
            self.selected_features = None
        else:
            H_full = fit_ridge_svd(X, Y, self.lambda_k)
            sel = prune_features(H_full, M)
            self.H = fit_ridge_svd(X[:, sel], Y, self.lambda_k)
            self.selected_features = sel
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xu = X if self.selected_features is None else X[:, self.selected_features]
        return Xu @ self.H
