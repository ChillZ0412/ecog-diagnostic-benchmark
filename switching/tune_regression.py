"""
Validation-based hyperparameter search for the REGRESSION branch
(tau_samples, lambda_k, feature-pruning M) -- Stage 5/6/baseline only.
Does NOT touch the classifier (Stage 3/4); that's tuned separately in
tune_classifier.py, since it's much slower (group-lasso) and independent
of this branch (oracle/baseline don't need the classifier at all).

Run in the same folder as the other files, with data/ populated:

    python tune_regression.py

Splits each subject's TRAINING data chronologically into a fit portion
(first FS_TRAIN_FRACTION, reusing the same 3/5 split already used
elsewhere in this project) and a validation portion (the remaining 2/5).
Grid-searches tau_samples x lambda_k x M, scoring by the ORACLE decode's
average Pearson r on the validation portion (oracle isolates the
regression branch's own quality, independent of classifier accuracy).

This does NOT look at the real held-out TEST set at all -- selection is
entirely within the training data, matching standard validation practice.
"""
import time

import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from regression_features import savgol_smooth_all_channels, build_regression_features
from finger_regressor import extract_state_segment, FingerRegressor, fit_ridge_svd, pearson_r

TAU_GRID_MS = [20, 37, 75, 150, 300, 500]   # converted to samples via FS_ECOG below
LAMBDA_K_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
M_GRID = [None, 30, 60, 100]   # None = no pruning (use all n_channels*3 features)

for n in [1, 2, 3]:
    print(f"{'='*70}")
    print(f"SUBJECT {n}")
    print(f"{'='*70}")
    sd = load_subject(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

    T = len(sl.train_state)
    split = int(T * C.FS_TRAIN_FRACTION)
    fit_ecog, val_ecog = sd.train_ecog[:split], sd.train_ecog[split:]
    fit_glove, val_glove = sd.train_glove[:split], sd.train_glove[split:]
    fit_state, val_state = sl.train_state[:split], sl.train_state[split:]

    smoothed_fit = savgol_smooth_all_channels(fit_ecog)
    smoothed_val = savgol_smooth_all_channels(val_ecog)

    best = None
    results = []
    t_start = time.time()
    for tau_ms in TAU_GRID_MS:
        tau_samples = int(round(tau_ms * C.FS_ECOG / 1000.0))
        if tau_samples < 1:
            continue
        X_fit, (lo_f, hi_f) = build_regression_features(smoothed_fit, tau_samples)
        y_fit = fit_glove[lo_f:hi_f]
        state_fit = fit_state[lo_f:hi_f]

        X_val, (lo_v, hi_v) = build_regression_features(smoothed_val, tau_samples)
        y_val = val_glove[lo_v:hi_v]
        state_val = val_state[lo_v:hi_v]

        for lambda_k in LAMBDA_K_GRID:
            for M in M_GRID:
                models = {}
                for k in range(1, C.N_STATES + 1):
                    Xk, Yk = extract_state_segment(X_fit, y_fit, state_fit, k)
                    if Xk.shape[0] < 20:
                        continue
                    models[k] = FingerRegressor(state=k, lambda_k=lambda_k).fit(Xk, Yk, M=M)

                pred_val = np.zeros_like(y_val)
                for k, model in models.items():
                    mask = state_val == k
                    if mask.sum() == 0:
                        continue
                    pred_val[mask] = model.predict(X_val[mask])

                rs = [pearson_r(y_val[:, j], pred_val[:, j]) for j in range(C.N_FINGERS)]
                avg_r = np.nanmean(rs)
                results.append((tau_ms, lambda_k, M, avg_r))
                if best is None or avg_r > best[3]:
                    best = (tau_ms, lambda_k, M, avg_r)

    print(f"  grid search took {time.time()-t_start:.1f}s over {len(results)} combos")
    print(f"  BEST: tau_ms={best[0]}  lambda_k={best[1]}  M={best[2]}  val_oracle_avg_r={best[3]:.3f}")
    print()
    print("  top 10 combos:")
    for tau_ms, lambda_k, M, avg_r in sorted(results, key=lambda x: -x[3])[:10]:
        print(f"    tau_ms={tau_ms:4d}  lambda_k={lambda_k:8.2f}  M={str(M):>5s}  val_r={avg_r:.3f}")
    print()

print("Done. The BEST line per subject is what to plug into evaluate_decoder.fit_decoder's")
print("tau_samples/lambda_k/M arguments for the final test-set run.")
