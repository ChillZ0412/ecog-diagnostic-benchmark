"""
Diagnose WHICH states the classifier confuses -- full confusion matrix and
per-state recall, not just the aggregate balanced accuracy. Tests the
hypothesis that rest (not the finger classes) is the bottleneck, which
would match what we already found on synthetic data (Stage 4): no
intercept term means a "nothing is active" state has no way to pull the
argmax away from the finger classes.

Run in the same folder as the other files, with data/ populated:

    python diagnose_confusion.py

Uses lambda_s=0 (no regularization, so no degenerate all-zero-rows
collapse) and K=15 channels, with the FULL max_iter=150 (not the tuning
scripts' reduced 60) so the result reflects a properly converged model.
"""
import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import StateClassifier, build_target_matrix, accuracy

K = 15
LAMBDA_S = 0.0
MAX_ITER = 2000  # bumped way up from 150 -- both subjects hit max_iter=150
                  # WITHOUT converging in the previous run (converged=False).
                  # Testing whether letting it actually converge changes the
                  # confusion pattern (currently: almost everything -> "rest").

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
    top_channels = select_top_k(ranking, k=K)

    afs_fit = extract_ar_features(fit_ecog, channels=top_channels,
                                   shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
    afs_val = extract_ar_features(val_ecog, channels=top_channels,
                                   shifts_ms=C.AR_TIME_SHIFTS_MS, n_coeffs_keep=2)
    Y_fit = build_target_matrix(fit_state, C.N_STATES)

    clf = StateClassifier(lambda_s=LAMBDA_S, max_iter=MAX_ITER, tol=1e-6)
    clf.fit(afs_fit.features, Y_fit)
    pred_val = clf.predict(afs_val.features)

    print(f"  converged: {clf.converged_}  n_iter: {clf.n_iter_}  row_sparsity: {clf.row_sparsity():.2f}")
    print()

    # full confusion matrix: rows = true state, cols = predicted state
    conf = np.zeros((C.N_STATES, C.N_STATES), dtype=int)
    for t, p in zip(val_state, pred_val):
        conf[t - 1, p - 1] += 1
    labels = C.FINGER_NAMES + ["rest"]
    print("  confusion matrix (rows=true, cols=predicted):")
    header = "         " + "".join(f"{l[:5]:>7s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row_str = "".join(f"{conf[i,j]:7d}" for j in range(C.N_STATES))
        print(f"  {row_label:6s} {row_str}")
    print()

    # per-state recall
    print("  per-state recall:")
    for k in range(1, C.N_STATES + 1):
        mask = val_state == k
        r = accuracy(val_state[mask], pred_val[mask]) if mask.sum() > 0 else float("nan")
        label = C.FINGER_NAMES[k-1] if k <= C.N_FINGERS else "rest"
        print(f"    {label:6s} (n={mask.sum():6d}): recall={r:.3f}")

    # the key test: does accuracy differ a lot between "true state is a finger"
    # vs "true state is rest"?
    finger_mask = val_state != C.REST_STATE
    rest_mask = val_state == C.REST_STATE
    finger_acc = accuracy(val_state[finger_mask], pred_val[finger_mask])
    rest_acc = accuracy(val_state[rest_mask], pred_val[rest_mask])
    print()
    print(f"  ACCURACY WHEN TRUE STATE IS A FINGER (1-5): {finger_acc:.3f}")
    print(f"  ACCURACY WHEN TRUE STATE IS REST:            {rest_acc:.3f}")
    print(f"  (chance level for either sub-problem varies; the comparison that matters")
    print(f"   is whether one is dramatically worse than the other)")
    print()

print("Done. If finger accuracy >> rest accuracy, the bottleneck matches the")
print("synthetic-data no-intercept finding (rest has no way to win the argmax).")
print("If BOTH are near chance, the bottleneck is elsewhere (features/lag), not rest.")
