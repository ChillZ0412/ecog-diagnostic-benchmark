"""
Local verification script for Stage 6 (finger_regressor.py).
Run in the same folder as the other pipeline files:

    python verify_stage6.py

Note: step 5's correlation numbers are IN-SAMPLE (fit and evaluated on the
same data) -- they're a plumbing/sanity check, not a real result. Held-out
evaluation happens in Stage 7.
"""
import numpy as np

import config as C
from synthetic_blocks import make_synthetic_blocks
from state_labels import make_state_labels
from regression_features import savgol_smooth_all_channels, build_regression_features
from finger_regressor import (
    extract_state_segment, FingerRegressor, fit_ridge_svd, prune_features, pearson_r,
)

print("=== 1. SVD ridge matches closed-form normal-equation ridge ===")
rng = np.random.default_rng(0)
n, d, m = 500, 10, 3
X = rng.standard_normal((n, d))
Y = rng.standard_normal((n, m))
lam = 2.0
H_svd = fit_ridge_svd(X, Y, lam)
H_ne = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)
err = np.max(np.abs(H_svd - H_ne))
print(f"  max|H_svd - H_normal_eq| = {err:.2e}  {'OK' if err < 1e-8 else 'FAIL'}")

print()
print("=== 2. lambda -> infinity shrinks H towards 0 ===")
H_big = fit_ridge_svd(X, Y, 1e12)
mx = np.max(np.abs(H_big))
print(f"  max|H| at lambda=1e12: {mx:.2e}  {'OK' if mx < 1e-6 else 'FAIL'}")

print()
print("=== 3. feature pruning recovers a known sparse informative set ===")
d2 = 20
H_true = np.zeros((d2, 3))
true_idx = [2, 9, 15]
H_true[true_idx] = rng.standard_normal((3, 3)) * 2
X2 = rng.standard_normal((2000, d2))
Y2 = X2 @ H_true + 0.05 * rng.standard_normal((2000, 3))
H_fit = fit_ridge_svd(X2, Y2, lambda_k=0.1)
top3 = set(prune_features(H_fit, M=3).tolist())
print(f"  true={set(true_idx)}  recovered={top3}  {'OK' if top3 == set(true_idx) else 'FAIL'}")

print()
print("=== 4. pearson_r edge cases ===")
x = np.array([1., 2., 3., 4., 5.])
r1, r2, r3 = pearson_r(x, x * 2 + 1), pearson_r(x, -x), pearson_r(x, np.ones(5))
print(f"  perfect+ ={r1:.3f}  perfect- ={r2:.3f}  zero-var={r3} (nan expected, no crash)")

print()
print("=== 5. full pipeline, in-sample own-finger correlation (subject 1) ===")
sd = make_synthetic_blocks(1, seed=0)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=1)
smoothed = savgol_smooth_all_channels(sd.train_ecog)
tau_samples = 200  # placeholder, not the tuned tau
X_full, (lo, hi) = build_regression_features(smoothed, tau_samples=tau_samples)
y_aligned = sd.train_glove[lo:hi]
state_aligned = sl.train_state[lo:hi]

correlations = []
for k in range(1, C.N_FINGERS + 1):
    X_k, Y_k = extract_state_segment(X_full, y_aligned, state_aligned, k)
    reg = FingerRegressor(state=k, lambda_k=1.0).fit(X_k, Y_k)
    pred_k = reg.predict(X_k)
    r = pearson_r(Y_k[:, k - 1], pred_k[:, k - 1])
    correlations.append(r)
    print(f"  state {k} ({C.FINGER_NAMES[k-1]:6s}, n={X_k.shape[0]:6d}): r={r:.3f}")
print(f"  average = {np.mean(correlations):.3f}  "
      f"(paper's real-data reference for this step is ~0.4653; synthetic being a bit "
      f"higher is expected -- IN-SAMPLE, not a real result, see module docstring)")

print()
print("ALL CHECKS RAN (read FAIL markers above if any)")
