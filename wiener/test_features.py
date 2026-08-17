"""
Stage 3 verification.

Run:
    python test_features.py --synthetic     # no real data needed
    python test_features.py --subject 1     # on the real recording

Checks:
  1. GLOVE STRUCTURE — is the 1000 Hz glove a zero-order-held 25 Hz signal?
     This decides whether the downsampling choice matters at all.
  2. AM CORRECTNESS — the feature must equal sum(v^2) over each 40 ms window,
     verified against an independent slow loop.
  3. SHAPES / RATE — features must land at exactly 25 Hz and 3 x n_ch columns.
  4. DOWNSAMPLE AGREEMENT — how much do first/mean/last/center differ?
  5. SCALE — report the magnitude that Stage 5 will have to cope with.
  6. FEATURE NAMING — round-trip against the paper's (channel, band) notation.
"""
import argparse
import time

import numpy as np

import config as C
from data_io import make_synthetic
from features import (describe_glove_structure, amplitude_modulation,
                      extract_am_features, downsample_target, align,
                      paper_feature_id, format_features)


def _ok(f):
    return "OK  " if f else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seconds", type=int, default=200)
    args = ap.parse_args()

    if args.synthetic:
        sd = make_synthetic(args.subject)
        print(">> SYNTHETIC data")
    else:
        from data_io import load_subject
        sd = load_subject(args.subject)
        print(f">> REAL data, subject {args.subject}")

    n = min(args.seconds * C.FS_ECOG, sd.train_ecog.shape[0])
    ecog, glove = sd.train_ecog[:n], sd.train_glove[:n]
    all_ok = True

    # --- 1. glove structure --------------------------------------------------
    print("\n[1] Glove structure (is it a zero-order-held 25 Hz signal?)")
    info = describe_glove_structure(glove)
    print(f"      within-block std = {info['within_block_std']:.6g}")
    print(f"      overall std      = {info['overall_std']:.6g}")
    print(f"      ratio            = {info['ratio']:.6g}")
    if info["zero_order_hold"]:
        print("  [OK  ] ZERO-ORDER HOLD confirmed -> first/mean/last/center are identical,")
        print("         so the downsampling choice cannot introduce a timing error.")
    else:
        print("  [NOTE] NOT zero-order hold -> the glove was interpolated.")
        print(f"         max|first-mean| = {info['max_first_minus_mean']:.6g}")
        print(f"         max|first-last| = {info['max_first_minus_last']:.6g}")
        print("         The downsample method now matters; 'mean' is the safest default.")

    # --- 2. AM correctness ---------------------------------------------------
    print("\n[2] AM feature == sum of squares over 40 ms windows")
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((1000, 4))
    fast = amplitude_modulation(probe)
    win = C.AM_WINDOW_SAMPLES
    slow = np.array([[(probe[b * win:(b + 1) * win, c] ** 2).sum()
                      for c in range(probe.shape[1])]
                     for b in range(probe.shape[0] // win)])
    good = np.allclose(fast, slow)
    all_ok &= good
    print(f"  [{_ok(good)}] max abs difference vs reference loop = "
          f"{np.abs(fast - slow).max():.3e}")

    # --- 3. shapes and rate --------------------------------------------------
    print("\n[3] Feature extraction shapes and sampling rate")
    t0 = time.time()
    X, names = extract_am_features(ecog)
    dt = time.time() - t0
    y = downsample_target(glove)
    X, y = align(X, y)

    exp_rows = n // win
    exp_cols = ecog.shape[1] * len(C.BANDS)
    good = X.shape == (exp_rows, exp_cols) and y.shape == (exp_rows, C.N_FINGERS)
    all_ok &= good
    print(f"  [{_ok(good)}] X {X.shape}  y {y.shape}   "
          f"(expected ({exp_rows}, {exp_cols}) and ({exp_rows}, 5))")

    eff_fs = X.shape[0] / (n / C.FS_ECOG)
    good = abs(eff_fs - C.FS_FEATURE) < 0.1
    all_ok &= good
    print(f"  [{_ok(good)}] effective feature rate = {eff_fs:.2f} Hz "
          f"(expected {C.FS_FEATURE})")
    good = np.isfinite(X).all() and np.isfinite(y).all()
    all_ok &= good
    print(f"  [{_ok(good)}] all finite;  extraction took {dt:.1f}s")

    # --- 4. downsample method agreement -------------------------------------
    print("\n[4] Downsample method agreement")
    variants = {m: downsample_target(glove, method=m)
                for m in ("first", "mean", "last", "center")}
    base = variants["first"]
    for m, v in variants.items():
        if m == "first":
            continue
        d = np.abs(v - base).max()
        print(f"      max|{m:6s} - first| = {d:.6g}")

    # --- 5. scale / conditioning warning ------------------------------------
    print("\n[5] Feature scale (what Stage 5 must survive)")
    print(f"      X  min={X.min():.3e}  median={np.median(X):.3e}  max={X.max():.3e}")
    print(f"      implied X^T X magnitude ~ {X.max() ** 2 * X.shape[0]:.1e}")
    if X.max() ** 2 * X.shape[0] > 1e18:
        print("      -> confirms the normal equations are unusable; Stage 5 will")
        print("         solve by SVD / pinv(X) @ d instead of forming X^T X.")

    # --- 6. naming round-trip ------------------------------------------------
    print("\n[6] Feature naming vs the paper's (channel, band) notation")
    good = (paper_feature_id(("sub", 0)) == (1, 1)
            and paper_feature_id(("gamma", 0)) == (1, 2)
            and paper_feature_id(("fastgamma", 0)) == (1, 3))
    all_ok &= good
    print(f"  [{_ok(good)}] ('fastgamma', ch0) -> {paper_feature_id(('fastgamma', 0))} "
          f"  [paper Fig.3 lists (1,3) as subject 1's top index-finger feature]")
    print(f"      first three feature names: {names[:3]}")
    print(f"      formatted: {format_features(names, [0, 1, 2])}")

    print(f"\n{'=' * 62}")
    print(f"STAGE 3: {'ALL CHECKS PASSED' if all_ok else 'PROBLEMS FOUND'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
