"""
Stage 5 verification.

Run:
    python test_regressor.py

Checks:
  1. EXACT RECOVERY — on a clean, well-conditioned problem the solver must
     recover the true weights essentially exactly.
  2. PEARSON METRIC — correctness against numpy, plus the degenerate case.
  3. INVARIANCE — a fitted model's predictions must not depend on whether the
     features were standardised first (they should agree to solver tolerance).
  4. THE PAPER'S CLAIM — inv() vs pinv() at this dataset's actual scale and
     SNR. This is the experiment that justifies the paper's stated
     contribution, and the numbers belong in the report.
  5. PREDICT/SCORE ROUND-TRIP — a refitted model scores identically.
"""
import numpy as np

import config as C
from regressor import fit_wiener, pearson_r


def _ok(f):
    return "OK  " if f else "FAIL"


def make_scaled_design(n, nfeat, k, seed):
    """Design matrix with the heavy-tailed magnitude of the real AM features."""
    r = np.random.default_rng(seed)
    base = np.exp(r.normal(np.log(1.5e6), 2.2, size=(n + k, nfeat)))
    cols = [base[k - j:k - j + n, f] for f in range(nfeat) for j in range(k + 1)]
    return np.column_stack(cols + [np.ones(n)])


def main():
    all_ok = True

    # ---- 1. exact recovery on a clean problem ------------------------------
    print("\n[1] Exact recovery (well-conditioned, noiseless)")
    rng = np.random.default_rng(0)
    D = np.column_stack([rng.standard_normal((2000, 12)), np.ones(2000)])
    w_true = rng.standard_normal(13)
    d = D @ w_true
    for solver in ("pinv_normal", "svd"):
        fit = fit_wiener(D, d, solver=solver, standardize=False)
        err = np.linalg.norm(fit.weights - w_true) / np.linalg.norm(w_true)
        good = err < 1e-8 and fit.diagnostics["train_r"] > 1 - 1e-10
        all_ok &= good
        print(f"  [{_ok(good)}] {solver:12s} rel weight error = {err:.2e}, "
              f"train r = {fit.diagnostics['train_r']:.10f}")

    # ---- 2. pearson metric --------------------------------------------------
    print("\n[2] Pearson correlation")
    a, b = rng.standard_normal(500), rng.standard_normal(500)
    good = abs(pearson_r(a, b) - np.corrcoef(a, b)[0, 1]) < 1e-12
    all_ok &= good
    print(f"  [{_ok(good)}] matches numpy.corrcoef")
    good = pearson_r(a, a) > 1 - 1e-12 and abs(pearson_r(a, -a) + 1) < 1e-12
    all_ok &= good
    print(f"  [{_ok(good)}] r(a,a)=+1 and r(a,-a)=-1")
    good = np.isnan(pearson_r(np.ones(100), rng.standard_normal(100)))
    all_ok &= good
    print(f"  [{_ok(good)}] constant input -> nan (not a silent 0)")

    # ---- 3. standardisation invariance -------------------------------------
    print("\n[3] Standardisation must not change predictions")
    Dz = make_scaled_design(3000, 6, C.MEMORY_K, seed=3)
    wt = rng.standard_normal(Dz.shape[1]) * 1e-7
    dz = Dz @ wt + 0.5 * np.std(Dz @ wt) * rng.standard_normal(3000)
    f_raw = fit_wiener(Dz, dz, solver="svd", standardize=False)
    f_std = fit_wiener(Dz, dz, solver="svd", standardize=True)
    r_raw, r_std = f_raw.score(Dz, dz), f_std.score(Dz, dz)
    good = abs(r_raw - r_std) < 1e-3
    all_ok &= good
    print(f"  [{_ok(good)}] train r raw = {r_raw:.6f}, standardised = {r_std:.6f}, "
          f"diff = {abs(r_raw - r_std):.2e}")

    # ---- 4. the paper's claim ----------------------------------------------
    print("\n[4] The paper's contribution: inv() vs pinv() at real data scale")
    k, nfeat = C.MEMORY_K, C.FS_MAX_FEATURES
    Xtr = make_scaled_design(9975, nfeat, k, seed=0)
    Xte = make_scaled_design(4975, nfeat, k, seed=99)
    rr = np.random.default_rng(7)
    w_true = rr.standard_normal(Xtr.shape[1]) * 1e-7
    sig_tr, sig_te = Xtr @ w_true, Xte @ w_true
    nz = 1.7 * np.std(sig_tr)            # noise tuned so achievable r ~ 0.5
    dtr = sig_tr + nz * rr.standard_normal(len(sig_tr))
    dte = sig_te + nz * rr.standard_normal(len(sig_te))

    print(f"      cond(X) = {np.linalg.cond(Xtr):.2e}   "
          f"cond(X^T X) = {np.linalg.cond(Xtr.T @ Xtr):.2e}")
    print(f"      ceiling (true model) test r = {pearson_r(sig_te, dte):.4f}")
    print(f"      {'solver':14s} {'train r':>9s} {'TEST r':>9s} {'||w||':>12s} {'eff.rank':>9s}")
    print("      " + "-" * 56)

    norms = {}
    for solver in ("inv", "pinv_normal", "svd"):
        fit = fit_wiener(Xtr, dtr, solver=solver, standardize=False)
        te = fit.score(Xte, dte)
        norms[solver] = fit.diagnostics["weight_norm"]
        print(f"      {solver:14s} {fit.diagnostics['train_r']:9.4f} {te:9.4f} "
              f"{fit.diagnostics['weight_norm']:12.3e} "
              f"{fit.diagnostics['effective_rank']:9d}")

    ratio = norms["inv"] / max(norms["pinv_normal"], 1e-300)
    good = ratio > 100.0
    all_ok &= good
    print(f"  [{_ok(good)}] ||w|| ratio inv/pinv = {ratio:.2e}")
    print("        -> the pseudo-inverse discards near-null-space directions")
    print("           instead of amplifying them. This is the stability the")
    print("           paper claims, and it is why 'pinv_normal' is the default.")

    # ---- 5. predict / score round-trip -------------------------------------
    print("\n[5] Predict / score round-trip")
    fit = fit_wiener(Xtr, dtr, solver="pinv_normal")
    manual = pearson_r(fit.predict(Xte), dte)
    good = abs(manual - fit.score(Xte, dte)) < 1e-12
    all_ok &= good
    print(f"  [{_ok(good)}] predict()+pearson_r == score()  ({manual:.6f})")

    print(f"\n{'=' * 62}")
    print(f"STAGE 5: {'ALL CHECKS PASSED' if all_ok else 'PROBLEMS FOUND'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
