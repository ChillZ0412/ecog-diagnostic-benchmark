"""
Grid search over the AR feature time-shift ts (paper's "+ts/-ts" signal
versions, config.AR_TIME_SHIFTS_MS) -- this dimension was left at the
placeholder [0] (no shifts) through all earlier tuning. Focused on
subjects 1/2, where tune_classifier.py found balanced accuracy stuck at
random-chance (~1/6) across the ENTIRE K x lambda_s grid -- a sign the
features themselves (not the regularization) were the bottleneck.

Run in the same folder as the other files, with data/ populated:

    python tune_classifier_shifts.py

For each candidate ts, builds AR_TIME_SHIFTS_MS = [-ts, 0, +ts] (matching
the paper's 3-version worked example, 48*3*2=240) and evaluates balanced
accuracy on the validation split, at two lambda_s values (0 and a mildly
regularized value) with a fixed K=15 channels (a reasonable middle value
from the earlier K search -- this script isolates the shift dimension
first; if a clearly-better ts is found, K/lambda_s can be re-searched
around it afterward).
"""
import time

import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import StateClassifier, build_target_matrix, accuracy

K_FIXED = 15
LAMBDA_S_GRID = [0.0, 1e3]
TS_GRID_MS = [10, 20, 40, 75, 150]
SEARCH_MAX_ITER = 60


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
    fit_ecog, val_ecog = sd.train_ecog[:split], sd.train_ecog[split:]
    fit_state, val_state = sl.train_state[:split], sl.train_state[split:]

    ranking = rank_channels(fit_ecog, fit_state, n_states=C.N_STATES)
    top_channels = select_top_k(ranking, k=K_FIXED)

    results = []
    best = None
    t_start = time.time()
    for ts_ms in TS_GRID_MS:
        shifts = [-ts_ms, 0, ts_ms]
        afs_fit = extract_ar_features(fit_ecog, channels=top_channels,
                                       shifts_ms=shifts, n_coeffs_keep=2)
        afs_val = extract_ar_features(val_ecog, channels=top_channels,
                                       shifts_ms=shifts, n_coeffs_keep=2)
        Y_fit = build_target_matrix(fit_state, C.N_STATES)

        for lambda_s in LAMBDA_S_GRID:
            clf = StateClassifier(lambda_s=lambda_s, max_iter=SEARCH_MAX_ITER, tol=1e-6)
            clf.fit(afs_fit.features, Y_fit)
            pred_val = clf.predict(afs_val.features)

            bal_acc = balanced_accuracy(val_state, pred_val, C.N_STATES)
            raw_acc = accuracy(val_state, pred_val)
            results.append((ts_ms, lambda_s, bal_acc, raw_acc, clf.row_sparsity()))
            if best is None or bal_acc > best[2]:
                best = (ts_ms, lambda_s, bal_acc, raw_acc, clf.row_sparsity())

    print(f"  grid search took {time.time()-t_start:.1f}s over {len(results)} combos "
          f"(K fixed at {K_FIXED}, feature dim = {K_FIXED*3*2})")
    print(f"  BEST: ts_ms={best[0]}  lambda_s={best[1]}  val_balanced_acc={best[2]:.3f}  "
          f"val_raw_acc={best[3]:.3f}  row_sparsity={best[4]:.2f}")
    print(f"  (random chance = {1/C.N_STATES:.3f} -- compare against that)")
    print()
    print("  all combos:")
    for ts_ms, lambda_s, bal_acc, raw_acc, sparsity in sorted(results, key=lambda x: -x[2]):
        print(f"    ts_ms={ts_ms:4d}  lambda_s={lambda_s:8.1f}  bal_acc={bal_acc:.3f}  "
              f"raw_acc={raw_acc:.3f}  sparsity={sparsity:.2f}")
    print()

print("Done. If any ts clearly beats random chance (0.167), rerun tune_classifier.py's")
print("K x lambda_s grid with AR_TIME_SHIFTS_MS fixed to that ts to fine-tune around it.")
print("If NONE beat random chance even with shifts added, the bottleneck is likely")
print("deeper than the shift dimension -- worth discussing before going further.")
