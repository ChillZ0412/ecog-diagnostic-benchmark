"""
Stage 7 -- Full switching decoder assembly + evaluation
(Flamary & Rakotomamonjy 2011/2012, Sec 3.4 "Decoding finger movement" +
Sec 4.3 "Evaluating the switching decoder method", Table 3).

Replicates the paper's own three-tier comparison, ALL evaluated on the
held-out TEST split (never on training data):

  (a) baseline  -- single global linear model, no switching (Table 3a)
  (b) oracle    -- switching decoder using the TRUE test-set state sequence
                   (Table 3b). Legitimate only as a diagnostic upper bound
                   -- the paper's own footnote: "This is possible since the
                   finger movements on the test set are now available."
                   NOT a deployable number.
  (c) estimated -- switching decoder using states PREDICTED by the Stage 3/4
                   classifier from ECoG alone (Table 3c). This is the real,
                   deployable number -- what actually competed in the BCI
                   Competition (paper's reported r=0.427, 2nd place).

(b) and (c) reuse the SAME H_k regression models (trained once, on
oracle-segmented training data, per Eq. 3 -- that part never changes).
They differ ONLY in which state sequence drives the switch at test time.
(a) is a genuinely different, single, unswitched model.

--------------------------------------------------------------------------
INDEX ALIGNMENT (the part most likely to silently go wrong)
--------------------------------------------------------------------------
ar_features.extract_ar_features preserves the full T (spline-evaluated at
every sample, no trimming). regression_features.build_regression_features
DROPS tau_samples at each end ('valid' semantics). So classifier state
predictions (length T) must be sliced to the regression features'
[lo, hi) range before being used as a switch signal -- done explicitly
below via `pred_state[lo:hi]`, not by assuming the two already match.
"""
from dataclasses import dataclass

import numpy as np

import config as C
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import StateClassifier, build_target_matrix
from regression_features import savgol_smooth_all_channels, build_regression_features
from finger_regressor import (
    extract_state_segment, FingerRegressor, fit_ridge_svd,
    pearson_r, r2_score, calibrated_r2, mae_score,
)


def downsample_to_25hz(x: np.ndarray, window_samples: int = None) -> np.ndarray:
    """
    Block-average x (T, ...) from FS_ECOG (1000Hz) down to the dataglove's
    native 25Hz, matching the Wiener pipeline's eval_fs=25Hz protocol
    (see config.AM_WINDOW_SAMPLES, config.AM_WINDOW_MS=40.0 -- the same
    40ms/40-sample window Wiener's own AM features use).

    This matters because the raw glove signal stored at 1000Hz is itself
    just a zero-order-hold of the true 25Hz recording (see data_io.py's
    make_synthetic docstring) -- correlating at 1000Hz partly compares a
    step function against a continuously-varying prediction, which is not
    the same statistical comparison as Wiener's native-rate evaluation.
    Block-averaging the true glove signal within a 40-sample window exactly
    recovers its 25Hz value (it's constant within the block); averaging the
    prediction the same way gives the fairest like-for-like comparison.

    x: (T,) or (T, n_cols). Trailing remainder shorter than window_samples
    is dropped ('valid' semantics, consistent with the rest of this project).
    """
    window_samples = C.AM_WINDOW_SAMPLES if window_samples is None else window_samples
    T = x.shape[0]
    n_blocks = T // window_samples
    usable = n_blocks * window_samples
    if x.ndim == 1:
        return x[:usable].reshape(n_blocks, window_samples).mean(axis=1)
    else:
        return x[:usable].reshape(n_blocks, window_samples, x.shape[1]).mean(axis=1)


@dataclass
class DecoderBundle:
    """Everything fit on the TRAINING split, ready to decode any test set."""
    top_channels: np.ndarray
    classifier: StateClassifier
    Hk_models: dict            # k -> FingerRegressor
    baseline_H: np.ndarray     # single global model (Table 3a)
    tau_samples: int
    ar_shifts_ms: list


def fit_decoder(train_ecog, train_glove, train_state, *,
                 n_select_channels, lambda_s, tau_samples, lambda_k,
                 ar_shifts_ms=None, M=None) -> DecoderBundle:
    ar_shifts_ms = list(C.AR_TIME_SHIFTS_MS) if ar_shifts_ms is None else ar_shifts_ms

    # --- classifier branch (Stage 3 + 4) ---
    ranking = rank_channels(train_ecog, train_state, n_states=C.N_STATES)
    top_channels = select_top_k(ranking, k=n_select_channels)
    afs_train = extract_ar_features(train_ecog, channels=top_channels,
                                     shifts_ms=ar_shifts_ms, n_coeffs_keep=2)
    Y_cls_train = build_target_matrix(train_state, C.N_STATES)
    classifier = StateClassifier(lambda_s=lambda_s, max_iter=150, tol=1e-6)
    classifier.fit(afs_train.features, Y_cls_train)

    # --- regression branch (Stage 5 + 6) ---
    smoothed_train = savgol_smooth_all_channels(train_ecog)
    X_reg_train, (lo, hi) = build_regression_features(smoothed_train, tau_samples)
    y_train_aligned = train_glove[lo:hi]
    state_train_aligned = train_state[lo:hi]

    Hk_models = {}
    for k in range(1, C.N_STATES + 1):
        X_k, Y_k = extract_state_segment(X_reg_train, y_train_aligned, state_train_aligned, k)
        if X_k.shape[0] == 0:
            continue
        Hk_models[k] = FingerRegressor(state=k, lambda_k=lambda_k).fit(X_k, Y_k, M=M)

    # --- baseline: single global model, no switching (Table 3a) ---
    baseline_H = fit_ridge_svd(X_reg_train, y_train_aligned, lambda_k)

    return DecoderBundle(top_channels=top_channels, classifier=classifier,
                          Hk_models=Hk_models, baseline_H=baseline_H,
                          tau_samples=tau_samples, ar_shifts_ms=ar_shifts_ms)


def _decode_with_states(Hk_models, X_reg, state_sequence):
    T = X_reg.shape[0]
    y_pred = np.zeros((T, C.N_FINGERS))
    for k, model in Hk_models.items():
        mask = state_sequence == k
        if mask.sum() == 0:
            continue
        y_pred[mask] = model.predict(X_reg[mask])
    return y_pred


def evaluate(bundle: DecoderBundle, test_ecog, test_glove, test_state_true):
    """
    Returns a dict with keys 'baseline', 'oracle', 'estimated', each mapping
    to a list of N_FINGERS Pearson-r values (paper Table 3 layout), plus the
    raw predicted test-state sequence used for 'estimated' (for inspection).

    Pearson r is computed at 25Hz (config.AM_WINDOW_SAMPLES block-averaged),
    matching the Wiener pipeline's eval_fs=25Hz protocol -- see
    downsample_to_25hz()'s docstring for why. The DECODE itself (state
    switching, H_k selection) still runs at the full 1000Hz resolution;
    only the final correlation computation is downsampled.

    Also returns mae_score() (2026-08-04 addition) for each tier -- same
    units as the raw dataglove target (test_glove, straight from
    data_io.load_subject(), zero transforms applied), cross-validated
    against FingerFlex's inverse-transformed predictions to confirm both
    methods share the same original units. Directly comparable across all
    three regression methods, no conversion needed.
    """
    smoothed_test = savgol_smooth_all_channels(test_ecog)
    X_reg_test, (lo, hi) = build_regression_features(smoothed_test, bundle.tau_samples)
    y_test_aligned = test_glove[lo:hi]
    state_true_aligned = test_state_true[lo:hi]

    # (a) baseline -- single global model, no switching
    pred_baseline = X_reg_test @ bundle.baseline_H

    # (b) oracle -- true test state sequence
    pred_oracle = _decode_with_states(bundle.Hk_models, X_reg_test, state_true_aligned)

    # (c) estimated -- classifier-predicted state sequence, ECoG only
    afs_test = extract_ar_features(test_ecog, channels=bundle.top_channels,
                                    shifts_ms=bundle.ar_shifts_ms, n_coeffs_keep=2)
    pred_state_full = bundle.classifier.predict(afs_test.features)  # length T (untrimmed)
    pred_state_aligned = pred_state_full[lo:hi]                      # align to [lo,hi)
    pred_estimated = _decode_with_states(bundle.Hk_models, X_reg_test, pred_state_aligned)

    # downsample true + all three predictions to 25Hz before scoring
    y_25 = downsample_to_25hz(y_test_aligned)
    pred_baseline_25 = downsample_to_25hz(pred_baseline)
    pred_oracle_25 = downsample_to_25hz(pred_oracle)
    pred_estimated_25 = downsample_to_25hz(pred_estimated)

    r_baseline = [pearson_r(y_25[:, j], pred_baseline_25[:, j]) for j in range(C.N_FINGERS)]
    r_oracle = [pearson_r(y_25[:, j], pred_oracle_25[:, j]) for j in range(C.N_FINGERS)]
    r_estimated = [pearson_r(y_25[:, j], pred_estimated_25[:, j]) for j in range(C.N_FINGERS)]

    r2_baseline = [r2_score(y_25[:, j], pred_baseline_25[:, j]) for j in range(C.N_FINGERS)]
    r2_oracle = [r2_score(y_25[:, j], pred_oracle_25[:, j]) for j in range(C.N_FINGERS)]
    r2_estimated = [r2_score(y_25[:, j], pred_estimated_25[:, j]) for j in range(C.N_FINGERS)]

    cr2_baseline = [calibrated_r2(y_25[:, j], pred_baseline_25[:, j]) for j in range(C.N_FINGERS)]
    cr2_oracle = [calibrated_r2(y_25[:, j], pred_oracle_25[:, j]) for j in range(C.N_FINGERS)]
    cr2_estimated = [calibrated_r2(y_25[:, j], pred_estimated_25[:, j]) for j in range(C.N_FINGERS)]

    # MAE (2026-08-04 addition) -- same call-argument order (y_25 first,
    # pred second) as the r2_score/calibrated_r2 calls above, for
    # consistency. MAE is symmetric in its two arguments so this order
    # doesn't affect correctness either way, unlike r2_score/calibrated_r2
    # (see module-level note above about checking finger_regressor.py's
    # parameter order for those two).
    mae_baseline = [mae_score(y_25[:, j], pred_baseline_25[:, j]) for j in range(C.N_FINGERS)]
    mae_oracle = [mae_score(y_25[:, j], pred_oracle_25[:, j]) for j in range(C.N_FINGERS)]
    mae_estimated = [mae_score(y_25[:, j], pred_estimated_25[:, j]) for j in range(C.N_FINGERS)]

    return {
        "baseline": r_baseline,
        "oracle": r_oracle,
        "estimated": r_estimated,
        "baseline_r2": r2_baseline,
        "oracle_r2": r2_oracle,
        "estimated_r2": r2_estimated,
        "baseline_calibrated_r2": cr2_baseline,
        "oracle_calibrated_r2": cr2_oracle,
        "estimated_calibrated_r2": cr2_estimated,
        "baseline_mae": mae_baseline,
        "oracle_mae": mae_oracle,
        "estimated_mae": mae_estimated,
        "pred_state_aligned": pred_state_aligned,
        "state_true_aligned": state_true_aligned,
    }
