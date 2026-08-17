"""
Check for electrode artifacts, mirroring the Wiener team's method (their
handoff doc section 4): aggregate heavy-tail statistics BY PHYSICAL
CHANNEL (using only training-set statistics, zero test leakage), then see
whether any channel's TEST-segment values wildly exceed what the training
distribution would predict -- exactly the pattern that caused their S3
ring-finger catastrophic failure (r=-0.004 -> +0.562 after excluding the
bad channel).

Triggered by: R^2 diagnostics just found S3's baseline decode catastrophically
diverging (r2_score as low as -984) across ALL 5 fingers, and 2 of 5 fingers
in the oracle decode -- consistent with a single/few bad channel(s)
dominating the (unpruned) baseline model's predictions.

Run in the same folder as the other files, with data/ populated:

    python check_electrode_artifacts.py

For each channel, on Subject 3:
  - train_max_abs: max |value| in the TRAINING ecog for that channel
  - test_max_abs: max |value| in the TEST ecog for that channel
  - ratio: test_max_abs / train_max_abs -- a ratio >> 1 (Wiener saw ~700,000x)
    means the test segment has an amplitude excursion training never showed,
    a strong artifact signature (real neural signal doesn't suddenly jump by
    orders of magnitude; hardware/contact issues do)
Also checked on the SMOOTHED regression-feature signal (post Savitzky-Golay),
since that's what the baseline model actually consumes.
"""
import numpy as np

import config as C
from data_io import load_subject_clean
from regression_features import savgol_smooth_all_channels

SUBJECT = 3
TOP_N = 15  # show the N channels with the largest test/train ratio

sd = load_subject_clean(SUBJECT)
n_ch = sd.n_channels

print(f"Subject {SUBJECT}, {n_ch} channels")
print()

print("=== RAW ECoG: test/train max-abs-value ratio per channel ===")
train_max = np.max(np.abs(sd.train_ecog), axis=0)
test_max = np.max(np.abs(sd.test_ecog), axis=0)
ratio = test_max / np.maximum(train_max, 1e-12)
order = np.argsort(ratio)[::-1]
print(f'{"channel":>8s} {"train_max":>12s} {"test_max":>12s} {"ratio":>12s}')
for ch in order[:TOP_N]:
    flag = "  <-- SUSPECT" if ratio[ch] > 10 else ""
    print(f'{ch:8d} {train_max[ch]:12.2f} {test_max[ch]:12.2f} {ratio[ch]:12.1f}{flag}')

print()
print("=== SMOOTHED regression-feature signal: same check (what baseline actually uses) ===")
smoothed_train = savgol_smooth_all_channels(sd.train_ecog)
smoothed_test = savgol_smooth_all_channels(sd.test_ecog)
train_max_s = np.max(np.abs(smoothed_train), axis=0)
test_max_s = np.max(np.abs(smoothed_test), axis=0)
ratio_s = test_max_s / np.maximum(train_max_s, 1e-12)
order_s = np.argsort(ratio_s)[::-1]
print(f'{"channel":>8s} {"train_max":>12s} {"test_max":>12s} {"ratio":>12s}')
for ch in order_s[:TOP_N]:
    flag = "  <-- SUSPECT" if ratio_s[ch] > 10 else ""
    print(f'{ch:8d} {train_max_s[ch]:12.2f} {test_max_s[ch]:12.2f} {ratio_s[ch]:12.1f}{flag}')

print()
n_suspect = np.sum(ratio_s > 10)
print(f"Channels with smoothed test/train ratio > 10: {n_suspect}")
if n_suspect > 0:
    print(f"Suspect channel indices: {sorted(order_s[:n_suspect].tolist())}")
    print()
    print("NEXT STEP if suspects found: refit baseline_H and/or H_k excluding these")
    print("channels (mirror Wiener's fix), see if S3's catastrophic r2_score values")
    print("resolve -- do NOT just exclude and declare victory without rerunning the")
    print("full evaluate() to confirm oracle/estimated numbers actually improve.")
else:
    print("No obvious single-channel amplitude artifact found via this check.")
    print("The catastrophic divergence may come from a different source -- e.g.")
    print("insufficient ridge regularization on baseline's UNPRUNED feature set")
    print("(baseline_H uses lambda_k=1.0 with no pruning, unlike H_k which is")
    print("pruned to M=30) interacting with collinear features. Worth testing a")
    print("much larger lambda_k on baseline specifically as a next check.")
