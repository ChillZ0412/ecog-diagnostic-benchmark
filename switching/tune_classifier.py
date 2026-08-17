"""
Validation-based hyperparameter search for the CLASSIFIER branch
(n_select_channels, lambda_s) -- Stage 3/4 only. Independent of the
regression branch tuned in tune_regression.py.

Run in the same folder as the other files, with data/ populated:

    python tune_classifier.py

Same chronological fit/validation split as tune_regression.py (first
FS_TRAIN_FRACTION of train = fit, remainder = validation). Scored by
BALANCED accuracy (macro-averaged per-state recall), not raw accuracy --
raw accuracy is misleading here because rest dominates the sample count
(see Stage 4 discussion: a classifier that's perfect on states 1-5 but
weak on rest can still look mediocre or even lose to a majority-class
baseline on raw accuracy alone).

Uses a REDUCED max_iter during the search (for speed) -- the final chosen
config should be refit with the full max_iter=150 for the real test-set
run, since fewer BCD iterations may not have fully converged.
"""
import time

import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import StateClassifier, build_target_matrix, accuracy

K_GRID = [10, 15, 20, 25]
LAMBDA_S_GRID = [0.0, 1e2, 1e3, 1e4, 1e5]
SEARCH_MAX_ITER = 60  # reduced for speed during search; refit with 150 for the final run


def balanced_accuracy(true_state, pred_state, n_states):
    recalls = []
    for k in range(1, n_states + 1):
        mask = true_state == k
        if mask.sum() == 0:
            continue
        recalls.append(accuracy(true_state[mask], pred_state[mask]))
    return float(np.mean(recalls))


for n in [1, 2, 3]:
    print(f"{'='*70}")
    print(f"SUBJECT {n}")
    print(f"{'='*70}")
    sd = load_subject(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

    T = len(sl.train_state)
    split = int(T * C.FS_TRAIN_FRACTION)
    fit_ecog, val_ecog = sd.train_ecog[:split], sd.train_ecog[split:]
    fit_state, val_state = sl.train_state[:split], sl.train_state[split:]

    results = []
    best = None
    t_start = time.time()
    for K in K_GRID:
        ranking = rank_channels(fit_ecog, fit_state, n_states=C.N_STATES)
        top_channels = select_top_k(ranking, k=K)

        afs_fit = extract_ar_features(fit_ecog, channels=top_channels,
                                       shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
        afs_val = extract_ar_features(val_ecog, channels=top_channels,
                                       shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
        Y_fit = build_target_matrix(fit_state, C.N_STATES)

        for lambda_s in LAMBDA_S_GRID:
            clf = StateClassifier(lambda_s=lambda_s, max_iter=SEARCH_MAX_ITER, tol=1e-6)
            clf.fit(afs_fit.features, Y_fit)
            pred_val = clf.predict(afs_val.features)

            bal_acc = balanced_accuracy(val_state, pred_val, C.N_STATES)
            raw_acc = accuracy(val_state, pred_val)
            results.append((K, lambda_s, bal_acc, raw_acc, clf.row_sparsity()))
            if best is None or bal_acc > best[2]:
                best = (K, lambda_s, bal_acc, raw_acc, clf.row_sparsity())

    print(f"  grid search took {time.time()-t_start:.1f}s over {len(results)} combos")
    print(f"  BEST: K={best[0]}  lambda_s={best[1]}  val_balanced_acc={best[2]:.3f}  "
          f"val_raw_acc={best[3]:.3f}  row_sparsity={best[4]:.2f}")
    print()
    print("  top 10 combos (by balanced accuracy):")
    for K, lambda_s, bal_acc, raw_acc, sparsity in sorted(results, key=lambda x: -x[2])[:10]:
        print(f"    K={K:3d}  lambda_s={lambda_s:8.1f}  bal_acc={bal_acc:.3f}  "
              f"raw_acc={raw_acc:.3f}  sparsity={sparsity:.2f}")
    print()

print("Done. The BEST line per subject -> plug K into n_select_channels and")
print("lambda_s into evaluate_decoder.fit_decoder, but refit the FINAL model")
print("with max_iter=150 (not the search's reduced 60) for the real test-set run.")
