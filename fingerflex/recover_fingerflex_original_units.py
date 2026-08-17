"""
recover_fingerflex_original_units.py

Purpose
-------
FingerFlex's res_npy/ predictions and targets are in MinMaxScaler-normalized
[0,1] space (confirmed via check_fingerflex_target_scale.py). The fitted
scaler object itself was never saved to disk (prepare_data.ipynb fits it
in-memory in Cell 8 and never pickles it), so it can't be loaded directly.

BUT MinMaxScaler.fit() is just column-wise (per-finger) min/max over the
TRAINING set -- nothing fancier. This script reconstructs those exact min/max
values by replaying the same finger-flex preprocessing steps the notebook
used before fitting the scaler (interpolate_fingerflex + crop_for_time_delay,
copied verbatim from prepare_data.ipynb, run ONLY on the training glove
signal -- this is cheap, no ECoG wavelet processing needed), then applies the
inverse transform to res_npy's prediction/true arrays and recomputes MAE in
original dataglove units.

IMPORTANT: this refits the scaler on TRAINING data only, matching
prepare_data.ipynb's Cell 8 (`scaler.fit(fingerflex_data.T)` uses the training
split's fingerflex_data, not the val/test split) -- do not accidentally fit on
test data, that would silently change the recovered min/max and make the
inverse_transform wrong.

Usage
-----
Edit RAW_DATA_DIR below to point at the folder containing sub{n}_comp.mat
(the same PATH used in prepare_data.ipynb), then run from the directory
containing res_npy/:

    python recover_fingerflex_original_units.py

Output
------
Per-subject, per-finger MAE in original dataglove units, plus the recovered
min/max used for the inverse transform (sanity-check these against Wiener's
reference range of roughly -1 to 3 -- if they don't land in a similar
ballpark, something is off and should NOT be trusted yet).
"""

import numpy as np
import scipy.io
import scipy.interpolate
from pathlib import Path

# ---- Config -- EDIT THESE TWO PATHS ------------------------------------
RAW_DATA_DIR = Path("data/pure_data")   # same PATH as in prepare_data.ipynb
RES_NPY_DIR = Path("res_npy")

SUBJECTS = [1, 2, 3]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]

# Constants copied from prepare_data.ipynb (must match exactly, these are
# hyperparameters of the original pipeline, not free choices)
DOWNSAMPLE_FS = 100
TIME_DELAY_SECS = 0.2
NATIVE_GLOVE_FS = 25   # true dataglove recording rate
ECOG_FS = 1000         # rate the raw .mat file's glove channel is stored at (zero-order-held)


# ---- Functions copied verbatim from prepare_data.ipynb ------------------

def reshape_column_ecog_data(multichannel_signal: np.ndarray):
    return multichannel_signal.T  # (time, features) -> (features, time)


def interpolate_fingerflex(finger_flex, cur_fs=1000, true_fs=25, needed_hz=DOWNSAMPLE_FS, interp_type='cubic'):
    downscaling_ratio = cur_fs // true_fs
    finger_flex_true_fs = finger_flex[:, ::downscaling_ratio]
    finger_flex_true_fs = np.c_[finger_flex_true_fs, finger_flex_true_fs.T[-1]]
    upscaling_ratio = needed_hz // true_fs
    ts = np.asarray(range(finger_flex_true_fs.shape[1])) * upscaling_ratio
    interpolated_finger_flex_funcs = [
        scipy.interpolate.interp1d(ts, ch, kind=interp_type) for ch in finger_flex_true_fs
    ]
    ts_needed_hz = np.asarray(range(finger_flex_true_fs.shape[1] * upscaling_ratio)[:-upscaling_ratio])
    interpolated = np.array([[f(t) for t in ts_needed_hz] for f in interpolated_finger_flex_funcs])
    return interpolated


def crop_for_time_delay(finger_flex, time_delay_sec, fs):
    time_delay = int(time_delay_sec * fs)
    return finger_flex[..., time_delay:]


# ---- Reconstruct the training-time scaler's min/max ----------------------

def recover_train_finger_flex(subject_id, return_raw=False):
    """Replays prepare_data.ipynb's Cells 0-3 for the TRAINING split only,
    to reproduce the exact `fingerflex_data` that scaler.fit() saw."""
    mat = scipy.io.loadmat(str(RAW_DATA_DIR / f"sub{subject_id}_comp.mat"))
    train_dg = mat["train_dg"].astype("float64")
    finger_flex = reshape_column_ecog_data(train_dg)          # (5, T_raw), 1000Hz zero-order-held
    interpolated = interpolate_fingerflex(finger_flex, cur_fs=ECOG_FS,
                                           true_fs=NATIVE_GLOVE_FS, needed_hz=DOWNSAMPLE_FS)
    cropped = crop_for_time_delay(interpolated, TIME_DELAY_SECS, DOWNSAMPLE_FS)
    if return_raw:
        # Native 25Hz samples, BEFORE any cubic interpolation -- just slice out
        # the real samples from the zero-order-held 1000Hz signal (every 40th
        # sample), no spline involved. Used only to test the overshoot hypothesis.
        raw_native = finger_flex[:, ::ECOG_FS // NATIVE_GLOVE_FS]
        return cropped, raw_native
    return cropped


def load_res_npy(subject_id):
    pred = np.load(RES_NPY_DIR / f"prediction_sub{subject_id}.npy")
    true = np.load(RES_NPY_DIR / f"true_sub{subject_id}.npy")
    if pred.shape[0] == 5 and pred.shape[1] != 5:
        pred = pred.T
    if true.shape[0] == 5 and true.shape[1] != 5:
        true = true.T
    return pred, true  # (T, 5)


def main():
    print("Recovering per-finger min/max from training data (matches MinMaxScaler.fit "
          "on the training split, as in prepare_data.ipynb Cell 8)...\n")

    all_mae = {}

    for sub_id in SUBJECTS:
        train_finger_flex, raw_native = recover_train_finger_flex(sub_id, return_raw=True)
        data_min = train_finger_flex.min(axis=1)                # per-finger min, shape (5,)
        data_max = train_finger_flex.max(axis=1)                # per-finger max, shape (5,)
        scale = data_max - data_min

        raw_min = raw_native.min(axis=1)
        raw_max = raw_native.max(axis=1)

        print(f"Subject {sub_id} — raw 25Hz (pre-interpolation) vs. cubic-interpolated 100Hz range:")
        for f, rmn, rmx, mn, mx in zip(FINGER_NAMES, raw_min, raw_max, data_min, data_max):
            overshoot_lo = mn < rmn
            overshoot_hi = mx > rmx
            flag = " <- OVERSHOOT" if (overshoot_lo or overshoot_hi) else ""
            print(f"    {f:<8} raw=[{rmn:+.4f}, {rmx:+.4f}]   interpolated=[{mn:+.4f}, {mx:+.4f}]{flag}")

        pred_scaled, true_scaled = load_res_npy(sub_id)   # (T, 5), in [0,1]

        # inverse_transform: X_original = X_scaled * (max - min) + min, per finger (column)
        pred_orig = pred_scaled * scale + data_min
        true_orig = true_scaled * scale + data_min

        mae_per_finger = np.mean(np.abs(pred_orig - true_orig), axis=0)
        all_mae[sub_id] = mae_per_finger

        print(f"  MAE (original units): " + ", ".join(
            f"{f}={v:.4f}" for f, v in zip(FINGER_NAMES, mae_per_finger)))
        print(f"  MAE (5-finger mean): {mae_per_finger.mean():.4f}\n")

    print("=" * 70)
    print("Summary -- MAE in original dataglove units (mean across 5 fingers)")
    print("=" * 70)
    for sub_id in SUBJECTS:
        print(f"  Subject {sub_id}: {all_mae[sub_id].mean():.4f}")

    print("\nSanity check: if the recovered min/max above do NOT resemble Wiener's")
    print("reference range (roughly -1 to 3), STOP -- something about the raw data")
    print("path, subject indexing, or the replayed preprocessing doesn't match what")
    print("prepare_data.ipynb actually did, and these MAE numbers should not be trusted")
    print("until that's resolved.")


if __name__ == "__main__":
    main()
