"""
FINAL diagnostic in this debugging chain: test whether a fixed neural-motor
lag explains why the state classifier can tell "moving vs rest" apart but
not "which finger" (see diagnose_confusion.py's findings -- ruled out:
sensor drift, missing time-shift features, non-convergence).

Motor cortex activity typically PRECEDES the physical movement by ~100-300ms.
We've been predicting state(t) from features(t); if the true relationship is
state(t + lag) ~ features(t) for some lag > 0, no amount of local ±ts
shifting (already tested, didn't help) can fix it -- that's a different
mechanism (local smoothing vs. global relabeling).

Run in the same folder as the other files, with data/ populated:

    python diagnose_lag.py

For each candidate lag, shifts the TRAIN and VALIDATION state sequences by
that amount (features stay fixed, labels move), refits Stage 3+4 the same
way as diagnose_confusion.py, and reports balanced accuracy + finger-only
accuracy. Estimated runtime: ~35 min total (9 lags x 2 subjects).
"""
import time

import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import StateClassifier, build_target_matrix, accuracy

LAG_GRID_MS = [-400, -300, -200, -100, 0, 100, 200, 300, 400]
K = 15
LAMBDA_S = 0.0
MAX_ITER = 150  # diagnose_confusion.py found 150 vs 2000 makes no difference; use 150 for speed


def shift_state(state: np.ndarray, lag_samples: int):
    """
    state_shifted[t] = state[t + lag_samples].
    Positive lag = testing "movement happens `lag` samples LATER than the
    neural signal" (i.e. features at t predict the label that will occur
    at t+lag). Returns (shifted_state, valid_slice), where valid_slice
    indexes into the ORIGINAL feature timeline (so ecog[valid_slice]
    lines up with shifted_state).
    """
    T = len(state)
    if lag_samples == 0:
        return state, slice(0, T)
    elif lag_samples > 0:
        return state[lag_samples:], slice(0, T - lag_samples)
    else:
        s = -lag_samples
        return state[:-s], slice(s, T)


def balanced_accuracy(true_state, pred_state, n_states):
    recalls = []
    for k in range(1, n_states + 1):
        mask = true_state == k
        if mask.sum() == 0:
            continue
        recalls.append(accuracy(true_state[mask], pred_state[mask]))
    return float(np.mean(recalls))


for n in [1, 2]:
    print(f"{'='*70}")
    print(f"SUBJECT {n}")
    print(f"{'='*70}")
    sd = load_subject(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

    T = len(sl.train_state)
    split = int(T * C.FS_TRAIN_FRACTION)
    fit_ecog_full, val_ecog_full = sd.train_ecog[:split], sd.train_ecog[split:]
    fit_state_full, val_state_full = sl.train_state[:split], sl.train_state[split:]

    results = []
    t_start = time.time()
    for lag_ms in LAG_GRID_MS:
        lag_samples = int(round(lag_ms * C.FS_ECOG / 1000.0))

        fit_state_shifted, fit_slice = shift_state(fit_state_full, lag_samples)
        val_state_shifted, val_slice = shift_state(val_state_full, lag_samples)
        fit_ecog = fit_ecog_full[fit_slice]
        val_ecog = val_ecog_full[val_slice]

        ranking = rank_channels(fit_ecog, fit_state_shifted, n_states=C.N_STATES)
        top_channels = select_top_k(ranking, k=K)

        afs_fit = extract_ar_features(fit_ecog, channels=top_channels,
                                       shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
        afs_val = extract_ar_features(val_ecog, channels=top_channels,
                                       shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
        Y_fit = build_target_matrix(fit_state_shifted, C.N_STATES)

        clf = StateClassifier(lambda_s=LAMBDA_S, max_iter=MAX_ITER, tol=1e-6)
        clf.fit(afs_fit.features, Y_fit)
        pred_val = clf.predict(afs_val.features)

        bal_acc = balanced_accuracy(val_state_shifted, pred_val, C.N_STATES)
        finger_mask = val_state_shifted != C.REST_STATE
        finger_acc = (accuracy(val_state_shifted[finger_mask], pred_val[finger_mask])
                      if finger_mask.sum() > 0 else float("nan"))
        results.append((lag_ms, bal_acc, finger_acc))
        print(f"  lag={lag_ms:5d}ms  bal_acc={bal_acc:.3f}  finger_only_acc={finger_acc:.3f}  "
              f"({time.time()-t_start:.0f}s elapsed)")

    best = max(results, key=lambda x: x[1])
    print()
    print(f"  BEST: lag={best[0]}ms  bal_acc={best[1]:.3f}  (chance = {1/C.N_STATES:.3f})")
    print()

print("Done. If bal_acc stays near chance (0.167) across ALL lags, the lag")
print("hypothesis is ruled out -- the bottleneck is the feature set's inability")
print("to discriminate WHICH finger, not a timing misalignment.")
