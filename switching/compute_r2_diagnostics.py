"""
R^2 / calibrated_R^2 / MAE diagnostics for Method 2, matching the Wiener
pipeline's methodology (distinguishes "wrong shape" from "wrong
scale/offset" -- see finger_regressor.r2_score / calibrated_r2
docstrings). This was the #1 flagged gap in the handoff document.
MAE added 2026-08-04 as a 4th supplementary metric (see finger_regressor.
mae_score docstring for the units cross-validation against Wiener/FingerFlex).

Run in the same folder as the other files, with data/ populated:

    python compute_r2_diagnostics.py

Uses the SAME tuned hyperparameters as final_run.py. Specifically checks
the two doubts raised about S3:
  1. S3 baseline r=0.037 (near zero) -- shape wrong, or scale/offset wrong?
  2. S3 middle-finger oracle r=0.181 (low vs S3's other fingers) -- mild
     issue, or catastrophic divergence (like Wiener's pre-fix S3 ring
     finger, R^2 ~ -1e9) hiding under the anatomical-confound explanation?
"""
import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate

TUNED = {
    1: dict(tau_samples=75, M=30),
    2: dict(tau_samples=150, M=30),
    3: dict(tau_samples=500, M=30),
}
LAMBDA_K = 1.0
N_SELECT_CHANNELS = 15
LAMBDA_S = 0.0

all_results = {}

for n in [1, 2, 3]:
    print(f"{'='*70}")
    print(f"SUBJECT {n}")
    print(f"{'='*70}")

    sd = load_subject_clean(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)
    tau_samples, M = TUNED[n]["tau_samples"], TUNED[n]["M"]

    bundle = fit_decoder(
        sd.train_ecog, sd.train_glove, sl.train_state,
        n_select_channels=N_SELECT_CHANNELS, lambda_s=LAMBDA_S,
        tau_samples=tau_samples, lambda_k=LAMBDA_K, M=M,
    )
    res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)
    all_results[n] = res

    for tier in ["baseline", "oracle", "estimated"]:
        print(f"\n  -- {tier} --")
        print(f'  {"finger":8s} {"r":>8s} {"r2_score":>10s} {"calibrated_r2":>15s} {"MAE":>10s}  interpretation')
        for j, name in enumerate(C.FINGER_NAMES):
            r = res[tier][j]
            r2 = res[f"{tier}_r2"][j]
            cr2 = res[f"{tier}_calibrated_r2"][j]
            mae = res[f"{tier}_mae"][j]
            gap = cr2 - r2
            if r2 < -1:
                note = "CATASTROPHIC divergence (like Wiener's pre-fix S3 case)"
            elif gap > 0.3:
                note = "shape OK, scale/offset off (calibration would fix)"
            elif r2 < 0.05 and cr2 < 0.05:
                note = "shape itself is weak (not just miscalibrated)"
            else:
                note = "normal"
            print(f'  {name:8s} {r:8.3f} {r2:10.3f} {cr2:15.3f} {mae:10.3f}  {note}')

        avg_mae = np.mean(res[f"{tier}_mae"])
        print(f'  {"AVG MAE":8s} {"":8s} {"":10s} {"":15s} {avg_mae:10.3f}')

print()
print(f"{'='*70}")
print("FOCUSED CHECK ON THE TWO FLAGGED DOUBTS")
print(f"{'='*70}")

r3 = all_results[3]
i_middle = C.FINGER_NAMES.index("middle")
print(f"\n1. S3 baseline (all fingers, r near 0 overall):")
for j, name in enumerate(C.FINGER_NAMES):
    print(f"   {name:8s}: r={r3['baseline'][j]:.3f}  r2_score={r3['baseline_r2'][j]:.3f}  "
          f"calibrated_r2={r3['baseline_calibrated_r2'][j]:.3f}  mae={r3['baseline_mae'][j]:.3f}")

print(f"\n2. S3 middle finger, oracle (r=0.181, flagged as possibly more than anatomy):")
print(f"   r={r3['oracle'][i_middle]:.3f}  r2_score={r3['oracle_r2'][i_middle]:.3f}  "
      f"calibrated_r2={r3['oracle_calibrated_r2'][i_middle]:.3f}  mae={r3['oracle_mae'][i_middle]:.3f}")
print(f"   compare to S3's other fingers' oracle r2_score: "
      f"{[round(r3['oracle_r2'][j],3) for j in range(C.N_FINGERS) if j != i_middle]}")
print(f"   compare to S3's other fingers' oracle mae: "
      f"{[round(r3['oracle_mae'][j],3) for j in range(C.N_FINGERS) if j != i_middle]}")

print()
print(f"{'='*70}")
print("SUMMARY -- MAE (mean +/- SD across 3 subjects, original dataglove units)")
print(f"{'='*70}")
for tier in ["baseline", "oracle", "estimated"]:
    subj_avgs = [np.mean(all_results[n][f"{tier}_mae"]) for n in [1, 2, 3]]
    print(f"  {tier:10s}: {np.mean(subj_avgs):.4f} +/- {np.std(subj_avgs, ddof=1):.4f}  "
          f"(per-subject: {[round(v,4) for v in subj_avgs]})")
