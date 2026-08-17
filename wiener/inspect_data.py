"""
Stage 1 — Load a subject and run a full sanity check.

Usage:
    python inspect_data.py --subject 1            # real data in ./data
    python inspect_data.py --subject 1 --synthetic # generate + inspect fake data

What it verifies (and why it matters downstream):
  * shapes & channel count match the paper (62/48/64)         -> catches wrong files
  * ECoG and glove have the SAME number of time samples       -> alignment is assumed everywhere
  * sampling rate implied by length matches 1000 Hz / 400-200 s split
  * no NaNs / Infs                                            -> FIR + least squares are NaN-fragile
  * per-finger dynamic range is non-degenerate                -> a flat finger => undefined correlation
"""
import argparse
import numpy as np

import config as C
from data_io import load_subject, make_synthetic, SubjectData


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ->  {detail}" if detail else ""))
    return ok


def inspect(sd: SubjectData) -> bool:
    print(f"\n=== Subject {sd.subject} ===")
    all_ok = True

    # --- shapes ---------------------------------------------------------------
    n_ch = sd.n_channels
    expected_ch = C.SUBJECT_CHANNELS.get(sd.subject)
    print(f"  train_ecog  {sd.train_ecog.shape}   train_glove {sd.train_glove.shape}")
    print(f"  test_ecog   {sd.test_ecog.shape}    test_glove  {sd.test_glove.shape}")

    all_ok &= _check("channel count matches paper",
                     expected_ch is None or n_ch == expected_ch,
                     f"{n_ch} channels (expected {expected_ch})")
    all_ok &= _check("glove has 5 fingers",
                     sd.train_glove.shape[1] == C.N_FINGERS == sd.test_glove.shape[1])

    # --- time alignment -------------------------------------------------------
    all_ok &= _check("train ecog/glove same length",
                     sd.train_ecog.shape[0] == sd.train_glove.shape[0],
                     f"{sd.train_ecog.shape[0]} vs {sd.train_glove.shape[0]}")
    all_ok &= _check("test ecog/glove same length",
                     sd.test_ecog.shape[0] == sd.test_glove.shape[0],
                     f"{sd.test_ecog.shape[0]} vs {sd.test_glove.shape[0]}")

    # --- sampling-rate / duration check ---------------------------------------
    train_sec = sd.train_ecog.shape[0] / C.FS_ECOG
    test_sec = sd.test_ecog.shape[0] / C.FS_ECOG
    all_ok &= _check("train duration ~400 s @1000 Hz", abs(train_sec - C.TRAIN_SECONDS) < 5,
                     f"{train_sec:.1f} s")
    all_ok &= _check("test duration ~200 s @1000 Hz", abs(test_sec - C.TEST_SECONDS) < 5,
                     f"{test_sec:.1f} s")

    # --- numerical health -----------------------------------------------------
    for name, arr in [("train_ecog", sd.train_ecog), ("train_glove", sd.train_glove),
                      ("test_ecog", sd.test_ecog), ("test_glove", sd.test_glove)]:
        finite = np.isfinite(arr).all()
        all_ok &= _check(f"{name} finite (no NaN/Inf)", finite)

    # --- per-finger movement summary -----------------------------------------
    print("  per-finger flexion range (train):")
    for f, fname in enumerate(C.FINGER_NAMES):
        col = sd.train_glove[:, f]
        rng = col.max() - col.min()
        all_ok &= _check(f"    {fname:6s} has movement", rng > 1e-6,
                         f"min={col.min():.3f}  max={col.max():.3f}  std={col.std():.3f}")

    print(f"  ECoG amplitude (train): mean|x|={np.abs(sd.train_ecog).mean():.3f}  "
          f"std={sd.train_ecog.std():.3f}")

    print(f"\n  => Subject {sd.subject}: {'ALL CHECKS PASSED' if all_ok else 'PROBLEMS FOUND'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--synthetic", action="store_true",
                    help="use generated fake data instead of ./data files")
    args = ap.parse_args()

    if args.synthetic:
        print(">> Using SYNTHETIC data (structure-only smoke test)")
        sd = make_synthetic(args.subject)
    else:
        print(f">> Loading REAL data from {C.DATA_DIR}")
        sd = load_subject(args.subject)

    inspect(sd)


if __name__ == "__main__":
    main()
