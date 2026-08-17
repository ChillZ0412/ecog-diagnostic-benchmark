"""
Run the full switching-decoder pipeline (Stage 0-7) on REAL data.
Run in the same folder as the other pipeline files, with data/ populated:

    python run_real_data.py

Does two things, in order, for each subject:
  1. A quick sanity check of the derived state labels (thresholds/debounce
     were only ever validated on synthetic data -- worth eyeballing on real
     glove traces before trusting the full ~1min fit that follows).
  2. The full fit_decoder + evaluate pipeline, printing the paper's Table 3
     layout (baseline / oracle / estimated) on the held-out TEST split.

Hyperparameters below (n_select_channels, lambda_s, tau_samples, lambda_k)
are still the SAME placeholder values used in the synthetic smoke tests --
NOT yet tuned on real validation data. Treat this run as "does the pipeline
produce sane numbers on real data", not as a final result. Proper tuning
(grid search over K, lambda_s, tau, lambda_k, M, AR_TIME_SHIFTS_MS on the
validation split) is the next step after this.
"""
import time

import numpy as np

import config as C
from data_io import load_subject
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

SUBJECTS = [1, 2, 3]

for n in SUBJECTS:
    print(f"{'='*60}")
    print(f"SUBJECT {n}")
    print(f"{'='*60}")

    sd = load_subject(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

    print()
    print("-- state label sanity check (thresholds tuned on synthetic data,")
    print("   first look at how they transfer to real glove traces) --")
    for split_name, state in [("train", sl.train_state), ("test", sl.test_state)]:
        vals, counts = np.unique(state, return_counts=True)
        dist = {int(v): round(float(c / counts.sum()), 3) for v, c in zip(vals, counts)}
        print(f"  {split_name} state distribution (1-5=finger, {C.REST_STATE}=rest): {dist}")
    print("  (if any state has ~0 samples, or rest is <10% or >95%, the")
    print("   STATE_ON_THRESHOLD_FRAC in config.py likely needs adjusting")
    print("   for real data before the fit below is worth trusting)")

    print()
    print("-- fitting full decoder (this is the slow step, ~1 min) --")
    t0 = time.time()
    bundle = fit_decoder(
        sd.train_ecog, sd.train_glove, sl.train_state,
        n_select_channels=15, lambda_s=1000.0, tau_samples=200, lambda_k=1.0,
    )
    print(f"  fit time: {time.time()-t0:.1f}s")
    print(f"  selected channels: {sorted(bundle.top_channels.tolist())}")
    print(f"  classifier row_sparsity: {bundle.classifier.row_sparsity():.2f}")

    print()
    print("-- evaluating on HELD-OUT TEST set (paper Table 3 structure) --")
    res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)

    print()
    print(f'{"finger":8s} {"(a) baseline":>14s} {"(b) oracle":>12s} {"(c) estimated":>14s}')
    for j, name in enumerate(C.FINGER_NAMES):
        print(f'{name:8s} {res["baseline"][j]:14.3f} {res["oracle"][j]:12.3f} {res["estimated"][j]:14.3f}')
    avg_b = np.nanmean(res["baseline"])
    avg_o = np.nanmean(res["oracle"])
    avg_e = np.nanmean(res["estimated"])
    print(f'{"AVERAGE":8s} {avg_b:14.3f} {avg_o:12.3f} {avg_e:14.3f}')

    state_acc = np.mean(res["pred_state_aligned"] == res["state_true_aligned"])
    print(f"  test-set state classification accuracy: {state_acc:.3f}")
    print()
    print(f"  paper reference (subject-averaged, Table 3): baseline={C.PAPER_LINEAR_BASELINE}  "
          f"oracle={C.PAPER_ORACLE_STATE}  estimated={C.PAPER_ESTIMATED_STATE}")
    print()

print("Done. Paste the full output back -- want to see per-subject numbers")
print("before deciding what to tune first.")
