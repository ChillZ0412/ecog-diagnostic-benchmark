"""
Stage 6 verification.

Run:
    python test_selection.py                # unit tests, no real data needed
    python test_selection.py --subject 1    # real forward selection, ~2 min

Checks:
  1. PSD PINV — the eigh-based solve must match numpy's pinv.
  2. LAG BLOCK — contiguous-slice construction must match the Stage 4 stack.
  3. PLANTED FEATURES — when only a few features genuinely drive the target,
     forward selection must find exactly those, and find them first.
  4. STOPPING RULE — behaves as the paper describes on hand-made curves.
  5. NO LEAKAGE — selection must only ever look at the inner split, never at
     data reserved for final testing.
"""
import argparse
import time

import numpy as np

import config as C
from selection import (_psd_pinv_solve, _lag_block, _apply_stopping_rule,
                       forward_select)
from memory_stack import build_memory_stack


def _ok(f):
    return "OK  " if f else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--seconds", type=int, default=400)
    args = ap.parse_args()
    all_ok = True
    rng = np.random.default_rng(0)

    # ---- 1. psd pinv --------------------------------------------------------
    print("\n[1] eigh-based PSD pseudo-inverse vs numpy pinv")
    A = rng.standard_normal((3000, 120))
    G, b = A.T @ A, rng.standard_normal(120)
    w1, w2 = np.linalg.pinv(G) @ b, _psd_pinv_solve(G, b)
    rel = np.linalg.norm(w1 - w2) / np.linalg.norm(w1)
    good = rel < 1e-10
    all_ok &= good
    print(f"  [{_ok(good)}] relative difference = {rel:.2e}")

    # ---- 2. lag block -------------------------------------------------------
    print("\n[2] Lag block matches the Stage 4 memory stack")
    k = C.MEMORY_K
    Xs = rng.standard_normal((300, 5))
    N = Xs.shape[0] - k
    L = _lag_block(Xs, 2, k, N)
    D = build_memory_stack(Xs, columns=[2], add_intercept=False)
    good = np.allclose(L, D)
    all_ok &= good
    print(f"  [{_ok(good)}] max difference = {np.abs(L - D).max():.2e}")

    # ---- 3. planted features ------------------------------------------------
    print("\n[3] Recovery of planted features")
    T, F = 4000, 40
    Xp = np.abs(rng.standard_normal((T, F))) + 0.1
    planted = [7, 23, 31]
    lags = [3, 0, 11]
    y = np.zeros(T)
    for c, lg in zip(planted, lags):
        y[lg:] += Xp[:T - lg, c] * 2.0
    y += 0.4 * y.std() * rng.standard_normal(T)

    names = [("gamma", i) for i in range(F)]
    t0 = time.time()
    res = forward_select(Xp, y, max_features=5, feature_names=names)
    dt = time.time() - t0
    found = res.order[:3]
    good = set(found) == set(planted)
    all_ok &= good
    print(f"  [{_ok(good)}] planted {planted} -> first three chosen {found}  ({dt:.1f}s)")
    print(f"        val r trajectory: " +
          " ".join(f"{v:.3f}" for v in res.val_r))
    good = all(res.val_r[i] <= res.val_r[i + 1] + 1e-9 for i in range(2))
    all_ok &= good
    print(f"  [{_ok(good)}] validation r is non-decreasing over the planted steps")

    # ---- 4. stopping rule ---------------------------------------------------
    print("\n[4] Stopping rule")
    cases = [([0.3, 0.5, 0.6, 0.55, 0.7], 3),
             ([0.4, 0.3], 1),
             ([0.1, 0.2, 0.3, 0.4], 4),
             ([0.5], 1)]
    for curve, expect in cases:
        got = _apply_stopping_rule(curve)
        good = got == expect
        all_ok &= good
        print(f"  [{_ok(good)}] {curve} -> keep {got} (expect {expect})")

    # ---- 5. validation split is actually doing its job ----------------------
    # The paper fits on the inner 3/5 and SELECTS on the 2/5 validation split.
    # (There is no "unseen" data inside the training set -- validation is used
    # for selection by design. The held-out competition test set is never
    # passed to this function at all, which is a structural guarantee.)
    # The meaningful property to test is that a feature which correlates only
    # within the inner-train portion -- an overfitting trap -- must LOSE to a
    # feature that generalises, even though the trap looks better on train.
    print("\n[5] Validation split rejects an overfitting trap")
    T2 = 3000
    N2 = T2 - k
    n_inner = int(round(N2 * C.FS_TRAIN_FRACTION))
    split = k + n_inner                       # index in original time base

    Xl = np.abs(rng.standard_normal((T2, 12))) + 0.1
    signal = Xl[:, 4].copy()                  # feature 4 = the honest predictor
    yl = 1.0 * signal + 1.2 * rng.standard_normal(T2)

    # feature 9 = the trap: nearly perfect on inner-train, pure noise afterwards
    trap = np.abs(rng.standard_normal(T2)) + 0.1
    trap[:split] = yl[:split] * 0.5 + 0.02 * rng.standard_normal(split)
    Xl[:, 9] = trap

    res5 = forward_select(Xl, yl, max_features=1)
    good = res5.order[0] == 4
    all_ok &= good
    print(f"  [{_ok(good)}] chose feature {res5.order[0]} "
          f"(4 = generalises, 9 = fits inner-train only)")
    print(f"        inner split at row {n_inner} of {N2} "
          f"({C.FS_TRAIN_FRACTION:.1%} train / {1 - C.FS_TRAIN_FRACTION:.1%} validation)")

    # ---- optional: real data ------------------------------------------------
    if args.subject is not None:
        print(f"\n[6] Real forward selection, subject {args.subject}")
        from data_io import load_subject
        from features import extract_am_features, downsample_target, align
        sd = load_subject(args.subject)
        n = min(args.seconds * C.FS_ECOG, sd.train_ecog.shape[0])
        t0 = time.time()
        Xr, names = extract_am_features(sd.train_ecog[:n])
        yr = downsample_target(sd.train_glove[:n])
        Xr, yr = align(Xr, yr)
        print(f"      features {Xr.shape}, extraction {time.time() - t0:.1f}s")
        for f in range(C.N_FINGERS):
            t0 = time.time()
            r = forward_select(Xr, yr[:, f], feature_names=names)
            print(f"      {C.FINGER_NAMES[f]:8s} keep {r.n_selected:2d}  "
                  f"val r={r.val_r[r.n_selected - 1]:.4f}  "
                  f"best over 10 = {max(r.val_r):.4f}  "
                  f"[{r.paper_notation()}]  ({time.time() - t0:.0f}s)")

    print(f"\n{'=' * 62}")
    print(f"STAGE 6: {'ALL CHECKS PASSED' if all_ok else 'PROBLEMS FOUND'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
