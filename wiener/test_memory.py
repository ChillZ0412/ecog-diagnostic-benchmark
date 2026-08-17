"""
Stage 4 verification.

Run:
    python test_memory.py --synthetic
    python test_memory.py --subject 1

Checks:
  1. EXACT INDEXING — every entry of the design matrix must equal the feature
     value at the right (time, lag). Verified against a brute-force loop.
     A silent off-by-one here would shift the ECoG history against the finger
     trace and quietly cap the achievable correlation.
  2. TARGET ALIGNMENT — design row i must predict y[i+k], with no drift.
  3. SHAPES — (T-k) x (F*(k+1) [+1]).
  4. SUBSET EQUIVALENCE — stacking a subset must equal stacking everything and
     then slicing the corresponding columns. This is what makes Stage 6 fast
     AND correct.
  5. MEMORY — report the cost of the full stack vs a 10-feature subset.
  6. RECOVERABILITY — build a signal that genuinely depends on a known lag and
     confirm least squares recovers that lag. This proves the stack carries
     usable temporal information, not just extra columns.
"""
import argparse

import numpy as np

import config as C
from memory_stack import (build_memory_stack, stack_targets, build_xy,
                          estimate_memory, column_labels)


def _ok(f):
    return "OK  " if f else "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seconds", type=int, default=200)
    args = ap.parse_args()

    k = C.MEMORY_K
    all_ok = True

    # ---- 1. exact indexing (small, fully checkable) -------------------------
    print(f"\n[1] Exact indexing (k = {k})")
    rng = np.random.default_rng(0)
    T, F = 200, 4
    X = rng.standard_normal((T, F))
    D = build_memory_stack(X, columns=None, add_intercept=False)

    bad = 0
    for i in range(0, D.shape[0], 7):          # sample rows for speed
        for f in range(F):
            for j in range(k + 1):
                if D[i, f * (k + 1) + j] != X[i + k - j, f]:
                    bad += 1
    good = bad == 0
    all_ok &= good
    print(f"  [{_ok(good)}] design[i, f*(k+1)+j] == X[i+k-j, f]   mismatches: {bad}")
    print(f"        row i holds [x(t), x(t-1), ..., x(t-{k})] for t = i+{k}")

    # ---- 2. target alignment -----------------------------------------------
    print("\n[2] Target alignment")
    y = np.arange(T, dtype=float)[:, None]
    Dx, ty = build_xy(X, y, add_intercept=False)
    good = (Dx.shape[0] == ty.shape[0]
            and ty[0, 0] == k
            and ty[-1, 0] == T - 1)
    all_ok &= good
    print(f"  [{_ok(good)}] rows {Dx.shape[0]} == targets {ty.shape[0]}; "
          f"first target = {ty[0, 0]:.0f} (expect {k}), "
          f"last = {ty[-1, 0]:.0f} (expect {T - 1})")

    # ---- 3. shapes ----------------------------------------------------------
    print("\n[3] Shapes")
    D1 = build_memory_stack(X, add_intercept=False)
    D2 = build_memory_stack(X, add_intercept=True)
    good = (D1.shape == (T - k, F * (k + 1))
            and D2.shape == (T - k, F * (k + 1) + 1)
            and np.all(D2[:, -1] == 1.0))
    all_ok &= good
    print(f"  [{_ok(good)}] no intercept {D1.shape}, with intercept {D2.shape}, "
          f"last column all ones")

    # ---- 4. subset equivalence ---------------------------------------------
    print("\n[4] Subset stacking == full stack sliced")
    cols = [3, 0]
    Dsub = build_memory_stack(X, columns=cols, add_intercept=False)
    take = [c * (k + 1) + j for c in cols for j in range(k + 1)]
    good = np.array_equal(Dsub, D1[:, take])
    all_ok &= good
    print(f"  [{_ok(good)}] subset {cols} matches sliced full stack "
          f"{Dsub.shape}  (order preserved)")

    # ---- 5. memory ----------------------------------------------------------
    print("\n[5] Memory cost")
    if args.synthetic:
        from data_io import make_synthetic
        sd = make_synthetic(args.subject)
    else:
        from data_io import load_subject
        sd = load_subject(args.subject)
    n_ch = sd.n_channels
    n_feat = n_ch * len(C.BANDS)
    rows25 = (args.seconds * C.FS_ECOG) // C.AM_WINDOW_SAMPLES

    full = estimate_memory(rows25, n_feat)
    sel = estimate_memory(rows25, C.FS_MAX_FEATURES)
    print(f"      all {n_feat} features : {full['rows']} x {full['cols']} "
          f"= {full['megabytes']:.0f} MB")
    print(f"      {C.FS_MAX_FEATURES} selected features: {sel['rows']} x {sel['cols']} "
          f"= {sel['megabytes']:.1f} MB")
    print(f"      -> Stage 6 selects first, then stacks: "
          f"{full['megabytes'] / max(sel['megabytes'], 1e-9):.0f}x smaller")

    # ---- 6. can least squares recover a known lag? --------------------------
    print("\n[6] Recoverability of a known lag")
    n = 4000
    src = rng.standard_normal((n, 1))
    true_lag = 7
    target = np.zeros(n)
    target[true_lag:] = src[:n - true_lag, 0] * 2.5
    target += 0.01 * rng.standard_normal(n)

    Dl, tl = build_xy(src, target[:, None], add_intercept=True)
    w, *_ = np.linalg.lstsq(Dl, tl[:, 0], rcond=None)
    found = int(np.argmax(np.abs(w[:k + 1])))
    good = found == true_lag
    all_ok &= good
    print(f"  [{_ok(good)}] injected lag {true_lag} -> recovered lag {found} "
          f"(weight {w[found]:+.3f}, expected ~+2.5)")

    labels = column_labels([("sub", 0)], k=k, add_intercept=True)
    print(f"      column label example: {labels[true_lag]} (band, channel, lag)")

    print(f"\n{'=' * 62}")
    print(f"STAGE 4: {'ALL CHECKS PASSED' if all_ok else 'PROBLEMS FOUND'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
