"""
Local verification script for Stage 7 (evaluate_decoder.py).
Run in the same folder as the other pipeline files:

    python verify_stage7.py

This is the full end-to-end switching-decoder pipeline on synthetic data --
it takes ~1 minute (the group-lasso classifier fit dominates). It prints the
paper's Table 3 layout (baseline / oracle / estimated) evaluated on the
HELD-OUT test split, plus test-set state classification accuracy.

Known synthetic-data caveat (see project notes / discussion write-up): with
the current synthetic generator, the ECoG-to-flexion mapping is linear and
state-independent, so baseline/oracle/estimated come out nearly identical
(~0.96-0.97) -- this does NOT indicate a bug, it means the synthetic task is
too easy to show the switching model's advantage. Real data should show the
paper's actual structure (baseline << oracle > estimated).
"""
import time

import numpy as np

import config as C
from synthetic_blocks import make_synthetic_blocks
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

n = 1
sd = make_synthetic_blocks(n, seed=0)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

print("Fitting full decoder on training split (this is the slow step, ~1 min)...")
t0 = time.time()
bundle = fit_decoder(
    sd.train_ecog, sd.train_glove, sl.train_state,
    n_select_channels=15, lambda_s=1000.0, tau_samples=200, lambda_k=1.0,
)
print(f"  fit time: {time.time() - t0:.1f}s")
print(f"  selected channels: {sorted(bundle.top_channels.tolist())}")
print(f"  classifier row_sparsity: {bundle.classifier.row_sparsity():.2f}")

print()
print("Evaluating on HELD-OUT TEST set (paper Table 3 structure)...")
res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)

print()
print(f'{"finger":8s} {"(a) baseline":>14s} {"(b) oracle":>12s} {"(c) estimated":>14s}')
for j, name in enumerate(C.FINGER_NAMES):
    print(f'{name:8s} {res["baseline"][j]:14.3f} {res["oracle"][j]:12.3f} {res["estimated"][j]:14.3f}')
avg_b = np.nanmean(res["baseline"])
avg_o = np.nanmean(res["oracle"])
avg_e = np.nanmean(res["estimated"])
print(f'{"AVERAGE":8s} {avg_b:14.3f} {avg_o:12.3f} {avg_e:14.3f}')

print()
print(f"paper reference (Table 3 avg, real data): baseline={C.PAPER_LINEAR_BASELINE}  "
      f"oracle={C.PAPER_ORACLE_STATE}  estimated={C.PAPER_ESTIMATED_STATE}")
state_acc = np.mean(res["pred_state_aligned"] == res["state_true_aligned"])
print(f"test-set state classification accuracy: {state_acc:.3f}")

print()
print("RAN TO COMPLETION (no crash). See module docstring re: the synthetic-data")
print("caveat above before interpreting the baseline/oracle/estimated gap.")
