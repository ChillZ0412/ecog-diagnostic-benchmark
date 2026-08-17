"""
Quick check (few minutes) of the "insufficient training samples" hypothesis
for S3's residual middle/ring oracle catastrophic divergence (r2_score
-2.16 / -1.41 after the channel49 fix). Also checks for a TARGET-side
(glove, not ECoG) amplitude artifact in the test segment, mirroring the
electrode check's logic but applied to the regression target itself.

Run in the same folder as the other files, with data/ populated:

    python check_s3_sample_size.py
"""
import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels

n = 3
sd = load_subject_clean(n)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

print("=== 1. state sample counts (FULL 400s train set, what H_k is fit on) ===")
vals, counts = np.unique(sl.train_state, return_counts=True)
total = counts.sum()
labels = C.FINGER_NAMES + ["rest"]
for v, c in zip(vals, counts):
    label = labels[v - 1]
    print(f"  state {v} ({label:6s}): n={c:7d}  ({c/total*100:5.1f}%)")

print()
print("=== 2. state sample counts (TEST set, what oracle decode actually touches) ===")
vals_t, counts_t = np.unique(sl.test_state, return_counts=True)
total_t = counts_t.sum()
for v, c in zip(vals_t, counts_t):
    label = labels[v - 1]
    print(f"  state {v} ({label:6s}): n={c:7d}  ({c/total_t*100:5.1f}%)")

print()
print("=== 3. target (glove) amplitude check: test vs train, WITHIN each finger's own state ===")
print("    (mirrors the electrode artifact check, but on the regression TARGET, not ECoG)")
for finger_idx, finger_name in enumerate(C.FINGER_NAMES):
    state_k = finger_idx + 1
    train_mask = sl.train_state == state_k
    test_mask = sl.test_state == state_k
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"  {finger_name:8s}: no samples in train or test for this state, skipping")
        continue
    train_vals = sd.train_glove[train_mask, finger_idx]
    test_vals = sd.test_glove[test_mask, finger_idx]
    train_max = np.max(np.abs(train_vals))
    test_max = np.max(np.abs(test_vals))
    ratio = test_max / max(train_max, 1e-12)
    flag = "  <-- SUSPECT (target-side artifact?)" if ratio > 5 else ""
    print(f"  {finger_name:8s}: train_max={train_max:8.3f}  test_max={test_max:8.3f}  "
          f"ratio={ratio:6.2f}{flag}")

print()
print("=== 4. summary ===")
print("If middle/ring do NOT have notably fewer samples than other fingers in #1/#2,")
print("the sample-size hypothesis is likely WRONG. If #3 flags middle or ring, that's")
print("a target-side (glove sensor) artifact analogous to the ECoG channel49 case --")
print("a genuinely new finding, not something to just accept as 'anatomy'.")
