"""
check_fingerflex_target_scale.py

Purpose
-------
Before MAE can be added as a comparison metric alongside r / R^2 / calibration_gap,
we need to know whether FingerFlex's saved predictions (res_npy/) are still in the
MinMaxScaler-normalized [0,1] space used during training, or whether they've already
been inverse-transformed back to the original dataglove finger-flexion units.

r and R^2 are invariant to affine transforms (scaling/shifting), so they're valid to
compare regardless of which space the numbers are in. MAE is NOT invariant -- an MAE
computed in [0,1]-normalized space is not comparable to an MAE computed in raw
dataglove units, and mixing them across methods would silently produce a wrong
comparison (harder to spot than the earlier eval_fs issue, because MAE looks like it
"has units" and invites naive cross-method comparison).

Reference point: Wiener's own diagnostic notes put the ORIGINAL dataglove finger-
flexion range at roughly -1 to 3 (arbitrary units, not degrees -- this is the BCI
Competition IV Dataset 4 glove signal, plotted as "Finger flexion (a.u.)" in Wiener's
Figure 1). If FingerFlex's true/pred arrays fall inside [0, 1], they are almost
certainly still normalized. If they resemble the -1..3 range (or otherwise clearly
exceed [0,1] with negative values), they're already in original units.

Usage
-----
Run from the directory containing res_npy/, or edit RES_NPY_DIR below.

    python check_fingerflex_target_scale.py

Output
------
Per-subject min/max/mean/std for both true and predicted arrays, plus a verdict on
which space the data is in, and (if it looks safe) a same-units MAE computed per
finger as a preview -- NOT yet validated against Wiener/switching model's own target
scale, which should be checked separately (see the printed reminder at the end).
"""

import numpy as np
from pathlib import Path

RES_NPY_DIR = Path("res_npy")   # adjust if your files live elsewhere
SUBJECTS = [1, 2, 3]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

# Reference range from Wiener's own diagnostic notes (original dataglove units)
WIENER_REFERENCE_RANGE = (-1.0, 3.0)


def load_subject(sub_id):
    pred = np.load(RES_NPY_DIR / f"prediction_sub{sub_id}.npy")
    true = np.load(RES_NPY_DIR / f"true_sub{sub_id}.npy")
    # normalize to (T, 5) shape regardless of how it was saved
    if pred.shape[0] == 5 and pred.shape[1] != 5:
        pred = pred.T
    if true.shape[0] == 5 and true.shape[1] != 5:
        true = true.T
    return pred, true


def describe(arr, label):
    print(f"    {label:<12} min={arr.min():+.4f}  max={arr.max():+.4f}  "
          f"mean={arr.mean():+.4f}  std={arr.std():.4f}")


def main():
    print(f"Reference (original dataglove units, from Wiener's notes): "
          f"{WIENER_REFERENCE_RANGE[0]} to {WIENER_REFERENCE_RANGE[1]}\n")

    all_true_min, all_true_max = [], []

    for sub_id in SUBJECTS:
        pred, true = load_subject(sub_id)
        print(f"Subject {sub_id}  (shape: pred={pred.shape}, true={true.shape})")
        describe(true, "true")
        describe(pred, "prediction")
        all_true_min.append(true.min())
        all_true_max.append(true.max())
        print()

    global_min, global_max = min(all_true_min), max(all_true_max)
    print("=" * 70)
    print(f"Overall true-value range across all subjects: {global_min:+.4f} to {global_max:+.4f}")

    looks_normalized = (global_min >= -0.05) and (global_max <= 1.05)
    looks_original = (global_min < -0.3) or (global_max > 1.5)

    if looks_normalized:
        print("\nVERDICT: values fall inside ~[0, 1].")
        print("  -> Almost certainly STILL in MinMaxScaler-normalized space.")
        print("  -> MAE computed directly on these arrays is NOT comparable to")
        print("     Wiener/switching model's MAE (which are presumably in original")
        print("     dataglove units). You need to inverse_transform back to original")
        print("     units first -- check prepare_data.ipynb for the fitted")
        print("     MinMaxScaler (likely saved as a .pkl/.joblib alongside the data,")
        print("     or refit-able from the same training split) and apply")
        print("     scaler.inverse_transform() to both prediction and true arrays")
        print("     before computing MAE.")
    elif looks_original:
        print("\nVERDICT: values exceed [0, 1] (negative values and/or values > 1.5).")
        print("  -> Likely already in original dataglove units (roughly matches")
        print("     Wiener's reference range).")
        print("  -> MAE computed directly on these arrays should be safe to compare,")
        print("     PROVIDED Wiener and the switching linear model's saved")
        print("     predictions are confirmed to be in the same original units too")
        print("     (check their data_io.py / regressor.py output -- don't assume,")
        print("     verify the same way).")
    else:
        print("\nVERDICT: ambiguous -- range doesn't clearly match either pattern.")
        print("  -> Inspect prepare_data.ipynb / Lightning_BCI-autoencoder.ipynb")
        print("     directly to confirm whether inverse_transform is applied before")
        print("     saving to res_npy/, rather than relying on this heuristic.")

    print("\nREMINDER: this script only checks FingerFlex. Before using MAE as a")
    print("cross-method comparison metric, confirm Wiener's and the switching")
    print("linear model's saved predictions are ALSO in the same original units")
    print("(not some other internal representation) -- otherwise MAE will silently")
    print("compare apples to oranges even if FingerFlex checks out fine here.")

    if looks_original:
        print("\n" + "=" * 70)
        print("Preview: per-finger MAE (only meaningful if verdict above is 'original units')")
        print("=" * 70)
        for sub_id in SUBJECTS:
            pred, true = load_subject(sub_id)
            mae_per_finger = np.mean(np.abs(pred - true), axis=0)
            print(f"Subject {sub_id}: " + ", ".join(
                f"{f}={v:.4f}" for f, v in zip(FINGER_NAMES, mae_per_finger)))


if __name__ == "__main__":
    main()
