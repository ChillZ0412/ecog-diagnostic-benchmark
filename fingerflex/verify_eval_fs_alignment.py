"""
verify_eval_fs_alignment.py

Purpose
-------
Resolves benchmark issue A: Wiener and the switching-linear-models method
report Pearson r at 25Hz (block-averaged down from their native rates), but
FingerFlex currently reports r at 100Hz (its native spectrogram/model rate).
Before the three regression methods can be put in the same benchmark table,
we need to know whether re-evaluating FingerFlex at 25Hz changes its r in a
way that matters, or whether the difference is negligible (as the switching
model team already found for their own 1000Hz -> 25Hz downsampling).

This script does NOT retrain or re-run inference. It reuses the already-saved
prediction_subX.npy / true_subX.npy arrays in res_npy/ (100Hz, test set only),
block-averages both prediction and ground truth down to 25Hz using the same
method the other two pipelines used (non-overlapping block mean, no anti-
aliasing filter -- consistent with how Wiener/switching already did their
downsampling, so the comparison is apples-to-apples), and recomputes Pearson r
per finger at both rates.

Usage
-----
Run from the directory containing res_npy/, or edit RES_NPY_DIR below.

    python verify_eval_fs_alignment.py

Output
------
A per-subject, per-finger table of r_100hz vs r_25hz and the delta, plus a
verdict against the 0.029 path_sensitivity_spread threshold that the Wiener
team already established as "this much variation is expected noise, not a
real effect." If |delta| for FingerFlex's avg_r stays under that threshold,
it's reasonable to report the 100Hz numbers as-is and note equivalence rather
than re-running the whole eval pipeline at 25Hz.
"""

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

# ---- Config -----------------------------------------------------------
RES_NPY_DIR = Path("res_npy")          # adjust if your files live elsewhere
SUBJECTS = [1, 2, 3]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
RING_FINGER_IDX = 3                     # for official_r (ring excluded), matches
                                         # the Wiener/switching-model convention
NATIVE_FS = 100                         # FingerFlex model output rate
TARGET_FS = 25                          # Wiener / switching-model eval rate
DOWNSAMPLE_FACTOR = NATIVE_FS // TARGET_FS  # = 4

# Threshold for "this difference is noise, not a real effect" -- reusing the
# Wiener team's own path_sensitivity_spread median (0.029) as the bar, since
# no FingerFlex-specific robustness number exists yet.
NOISE_THRESHOLD = 0.029


def load_subject(sub_id):
    """Load prediction/true arrays for one subject, normalize to (T, 5) shape."""
    pred = np.load(RES_NPY_DIR / f"prediction_sub{sub_id}.npy")
    true = np.load(RES_NPY_DIR / f"true_sub{sub_id}.npy")

    # Handle either (T, 5) or (5, T) storage -- transpose to (T, 5) if needed.
    if pred.shape[0] == 5 and pred.shape[1] != 5:
        pred = pred.T
    if true.shape[0] == 5 and true.shape[1] != 5:
        true = true.T

    assert pred.shape == true.shape, (
        f"Sub{sub_id}: prediction shape {pred.shape} != true shape {true.shape}"
    )
    assert pred.shape[1] == 5, f"Sub{sub_id}: expected 5 fingers, got shape {pred.shape}"
    return pred, true


def block_average_downsample(x, factor):
    """
    Non-overlapping block mean along the time axis (axis=0).
    Trims trailing samples that don't fill a full block, rather than padding,
    to avoid introducing edge artifacts -- same tradeoff the switching-model
    team made for their 1000Hz -> 25Hz downsampling.
    """
    t = x.shape[0]
    usable_len = (t // factor) * factor
    trimmed = x[:usable_len]
    reshaped = trimmed.reshape(usable_len // factor, factor, x.shape[1])
    return reshaped.mean(axis=1)


def pearson_per_finger(pred, true):
    """Return list of 5 Pearson r values, one per finger."""
    return [pearsonr(pred[:, f], true[:, f])[0] for f in range(5)]


def main():
    print(f"Downsample factor: {NATIVE_FS}Hz -> {TARGET_FS}Hz (factor={DOWNSAMPLE_FACTOR})\n")

    all_avg_100 = []
    all_avg_25 = []
    all_official_100 = []
    all_official_25 = []

    header = f"{'Subject':<10}{'Finger':<10}{'r_100Hz':>10}{'r_25Hz':>10}{'delta':>10}"
    print(header)
    print("-" * len(header))

    for sub_id in SUBJECTS:
        pred_100, true_100 = load_subject(sub_id)

        r_100 = pearson_per_finger(pred_100, true_100)

        pred_25 = block_average_downsample(pred_100, DOWNSAMPLE_FACTOR)
        true_25 = block_average_downsample(true_100, DOWNSAMPLE_FACTOR)
        r_25 = pearson_per_finger(pred_25, true_25)

        for f_idx, f_name in enumerate(FINGER_NAMES):
            delta = r_25[f_idx] - r_100[f_idx]
            print(f"S{sub_id:<9}{f_name:<10}{r_100[f_idx]:>10.3f}{r_25[f_idx]:>10.3f}{delta:>10.3f}")

        avg_100 = np.mean(r_100)
        avg_25 = np.mean(r_25)
        official_100 = np.mean([r_100[i] for i in range(5) if i != RING_FINGER_IDX])
        official_25 = np.mean([r_25[i] for i in range(5) if i != RING_FINGER_IDX])

        all_avg_100.append(avg_100)
        all_avg_25.append(avg_25)
        all_official_100.append(official_100)
        all_official_25.append(official_25)

        print(f"S{sub_id:<9}{'avg(5)':<10}{avg_100:>10.3f}{avg_25:>10.3f}{avg_25 - avg_100:>10.3f}")
        print(f"S{sub_id:<9}{'official(4)':<10}{official_100:>10.3f}{official_25:>10.3f}{official_25 - official_100:>10.3f}")
        print()

    grand_avg_100 = np.mean(all_avg_100)
    grand_avg_25 = np.mean(all_avg_25)
    grand_official_100 = np.mean(all_official_100)
    grand_official_25 = np.mean(all_official_25)
    grand_delta_avg = grand_avg_25 - grand_avg_100
    grand_delta_official = grand_official_25 - grand_official_100

    print("=" * len(header))
    print(f"Overall avg_r:      100Hz={grand_avg_100:.3f}  25Hz={grand_avg_25:.3f}  delta={grand_delta_avg:+.3f}")
    print(f"Overall official_r: 100Hz={grand_official_100:.3f}  25Hz={grand_official_25:.3f}  delta={grand_delta_official:+.3f}")
    print()

    worst_delta = max(abs(grand_delta_avg), abs(grand_delta_official))
    if worst_delta <= NOISE_THRESHOLD:
        print(f"VERDICT: max |delta| = {worst_delta:.3f} <= noise threshold ({NOISE_THRESHOLD}).")
        print("  -> 100Hz vs 25Hz difference is within the range already accepted as")
        print("     implementation noise elsewhere in the benchmark (Wiener's own")
        print("     path_sensitivity_spread median). Reasonable to report FingerFlex's")
        print("     100Hz numbers as-is and note eval_fs equivalence in the methods")
        print("     section, rather than re-running the full eval at 25Hz.")
    else:
        print(f"VERDICT: max |delta| = {worst_delta:.3f} > noise threshold ({NOISE_THRESHOLD}).")
        print("  -> The difference is large enough to matter. Recommend re-evaluating")
        print("     FingerFlex's reported r at 25Hz (using true_25/pred_25 computed here)")
        print("     rather than treating 100Hz and 25Hz numbers as interchangeable in")
        print("     the same table.")


if __name__ == "__main__":
    main()
