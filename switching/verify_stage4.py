"""
Local verification script for Stage 4 (state_classifier.py).
Run in the same folder as the other pipeline files:

    python verify_stage4.py

Checks, in order:
  1. lambda_s=0 must exactly match ordinary least squares (closed form).
  2. K=1 (single state) must exactly match standard Lasso soft-thresholding.
  3. Group sparsity: on a synthetic problem with a KNOWN sparse true support,
     the solver must recover that support exactly at a large-enough lambda_s.
  4. Full pipeline smoke test on synthetic block data: channel selection ->
     AR features -> classifier. Reports accuracy AND per-state recall, since
     raw accuracy is known to be misleading here (see Stage 4 discussion --
     rest dominates ~68% of samples, so watch the per-state recall line,
     not just the top-line accuracy).
"""
import numpy as np

import config as C
from synthetic_blocks import make_synthetic_blocks
from state_labels import make_state_labels
from channel_selection import rank_channels, select_top_k
from ar_features import extract_ar_features
from state_classifier import fit_group_lasso, build_target_matrix, accuracy, StateClassifier

print("=== 1. lambda_s=0 matches closed-form OLS ===")
rng = np.random.default_rng(0)
T, d, K = 2000, 8, 3
X = rng.standard_normal((T, d))
Y = rng.standard_normal((T, K))
C_bcd, n_iter, conv = fit_group_lasso(X, Y, lambda_s=0.0, max_iter=500, tol=1e-10)
C_ols, *_ = np.linalg.lstsq(X, Y, rcond=None)
err = np.max(np.abs(C_bcd - C_ols))
print(f"  iters={n_iter} converged={conv}  max|C_bcd - C_ols|={err:.2e}  "
      f"{'OK' if err < 1e-4 else 'FAIL'}")

print()
print("=== 2. K=1 matches standard Lasso ===")
Y1 = rng.standard_normal((T, 1))
lam = 5.0
C_bcd1, *_ = fit_group_lasso(X, Y1, lambda_s=lam, max_iter=500, tol=1e-10)


def lasso_cd(X, y, lam, max_iter=500, tol=1e-10):
    T, d = X.shape
    beta = np.zeros(d)
    col_sq = np.sum(X**2, axis=0)
    r = y - X @ beta
    for _ in range(max_iter):
        beta_prev = beta.copy()
        for i in range(d):
            r += X[:, i] * beta[i]
            rho = X[:, i] @ r
            thresh = lam / 2.0
            if rho > thresh:
                beta[i] = (rho - thresh) / col_sq[i]
            elif rho < -thresh:
                beta[i] = (rho + thresh) / col_sq[i]
            else:
                beta[i] = 0.0
            r -= X[:, i] * beta[i]
        if np.max(np.abs(beta - beta_prev)) < tol:
            break
    return beta


beta_ref = lasso_cd(X, Y1[:, 0], lam)
err1 = np.max(np.abs(C_bcd1[:, 0] - beta_ref))
print(f"  max|group_lasso(K=1) - standard_lasso| = {err1:.2e}  {'OK' if err1 < 1e-6 else 'FAIL'}")

print()
print("=== 3. exact support recovery on a known-sparse synthetic problem ===")
rng2 = np.random.default_rng(1)
T2, d2, K2 = 3000, 12, 6
X2 = rng2.standard_normal((T2, d2))
C_true = np.zeros((d2, K2))
true_support = [0, 3, 7]
C_true[true_support, :] = rng2.standard_normal((3, K2)) * 2
Y2 = X2 @ C_true + 0.1 * rng2.standard_normal((T2, K2))
C_est, *_ = fit_group_lasso(X2, Y2, lambda_s=50.0, max_iter=300, tol=1e-8)
recovered = set(np.where(np.linalg.norm(C_est, axis=1) > 1e-8)[0].tolist())
print(f"  true support={set(true_support)}  recovered={recovered}  "
      f"{'OK' if recovered == set(true_support) else 'FAIL'}")

print()
print("=== 4. full pipeline smoke test (synthetic_blocks, subject 1) ===")
sd = make_synthetic_blocks(1, seed=0)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=1)
Ttot = len(sl.train_state)
split = int(Ttot * C.FS_TRAIN_FRACTION)
tr_ecog, va_ecog = sd.train_ecog[:split], sd.train_ecog[split:]
tr_state, va_state = sl.train_state[:split], sl.train_state[split:]

ranking = rank_channels(tr_ecog, tr_state, n_states=C.N_STATES)
top_channels = select_top_k(ranking, k=15)

afs_tr = extract_ar_features(tr_ecog, channels=top_channels, shifts_ms=[0], n_coeffs_keep=2)
afs_va = extract_ar_features(va_ecog, channels=top_channels, shifts_ms=[0], n_coeffs_keep=2)
Y_tr = build_target_matrix(tr_state, C.N_STATES)

clf = StateClassifier(lambda_s=0.0, max_iter=150, tol=1e-6).fit(afs_tr.features, Y_tr)
pred_va = clf.predict(afs_va.features)

maj_acc = accuracy(va_state, np.full_like(va_state, C.REST_STATE))
acc = accuracy(va_state, pred_va)
print(f"  majority-baseline val_acc={maj_acc:.3f}   classifier val_acc={acc:.3f}  "
      f"(top-line accuracy CAN look unremarkable here -- see per-state recall below)")
print("  per-state recall (states 1-5, then 6=rest):")
for k in range(1, C.N_STATES + 1):
    mask = va_state == k
    r = accuracy(va_state[mask], pred_va[mask]) if mask.sum() > 0 else float("nan")
    print(f"    state {k}: n={mask.sum():6d}  recall={r:.2f}")

print()
print("ALL CHECKS RAN (read FAIL markers above if any; per-state recall has no pass/fail --")
print("just confirm states 1-5 recall is high while state 6 recall being weaker matches")
print("the documented no-intercept discussion, not a bug)")
