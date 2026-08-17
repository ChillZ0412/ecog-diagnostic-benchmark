"""
check_wiener_glove_range.py

Purpose
-------
Cross-validation check: FingerFlex's recovered per-finger min/max (used to
inverse_transform its MinMaxScaler-normalized predictions back to original
dataglove units) came out much wider than the "-1 to 3" range Wiener's
handoff doc mentioned in passing -- but that mention was about one specific
case (Subject 3's ring finger, test segment), not a verified global range
across all subjects/fingers/splits.

This script uses Wiener's own data_io.py (completely independent of anything
FingerFlex does -- no interpolation, no reshaping, straight from the .mat
files) to print the actual global min/max of train_glove and test_glove for
all 3 subjects. If this independent source also shows a wide range (roughly
matching what FingerFlex's recovered scaler found), that confirms FingerFlex's
numbers are correct and the "-1 to 3" reference was just an incomplete
data point, not a real target. If Wiener's range comes back narrow and
tightly bounded near -1 to 3, that's a sign something in the FingerFlex
recovery script needs another look.

Usage
-----
Run from Wiener's project directory (wherever data_io.py / config.py live,
with access to the sub{n}_comp.mat / sub{n}_testlabels.mat files):

    python check_wiener_glove_range.py
"""

from data_io import load_subject

SUBJECTS = [1, 2, 3]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]


def main():
    print("Wiener's raw dataglove range (train_glove / test_glove), per subject and per finger.")
    print("Source: data_io.load_subject() -- straight from the .mat files, no transforms applied.\n")

    for n in SUBJECTS:
        sd = load_subject(n)

        print(f"Subject {n}  (train_glove shape={sd.train_glove.shape}, "
              f"test_glove shape={sd.test_glove.shape})")

        # Overall (all 5 fingers combined)
        print(f"  Overall   train: min={sd.train_glove.min():+.4f}  max={sd.train_glove.max():+.4f}"
              f"   |   test: min={sd.test_glove.min():+.4f}  max={sd.test_glove.max():+.4f}")

        # Per-finger breakdown -- assumes glove arrays are (time, 5), one column per finger.
        # If your glove arrays are (5, time) instead, swap axis=0 <-> axis=1 below.
        n_fingers = sd.train_glove.shape[1] if sd.train_glove.ndim == 2 else None
        if n_fingers == 5:
            for i, finger in enumerate(FINGER_NAMES):
                tr_min, tr_max = sd.train_glove[:, i].min(), sd.train_glove[:, i].max()
                te_min, te_max = sd.test_glove[:, i].min(), sd.test_glove[:, i].max()
                print(f"    {finger:<8} train=[{tr_min:+.4f}, {tr_max:+.4f}]   "
                      f"test=[{te_min:+.4f}, {te_max:+.4f}]")
        else:
            print("    (glove array shape doesn't look like (time, 5) -- check axis "
                  "orientation manually before trusting the per-finger breakdown above)")
        print()

    print("=" * 70)
    print("Compare this against FingerFlex's recovered per-finger range from")
    print("recover_fingerflex_original_units.py. If the ranges are in the same")
    print("ballpark (even if not identical -- different subjects/fingers/people genuinely")
    print("flex by different amounts), that confirms FingerFlex's inverse-transform is")
    print("correct and the earlier '-1 to 3' reference was just incomplete, not a real")
    print("target every finger/subject should match.")


if __name__ == "__main__":
    main()
