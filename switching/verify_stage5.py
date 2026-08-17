"""
Local verification script for Stage 5 (regression_features.py).
Run in the same folder as the other pipeline files:

    python verify_stage5.py
"""
import time

import numpy as np

import config as C
from synthetic_blocks import make_synthetic_blocks
from regression_features import (
    savgol_smooth_all_channels, build_regression_features, _sg_window_samples,
)

print("=== 1. SG window length is odd ===")
w = _sg_window_samples(C.FS_ECOG, C.SG_WINDOW_SEC)
print(f"  window_sec={C.SG_WINDOW_SEC} fs={C.FS_ECOG} -> window_samples={w} "
      f"{'OK (odd)' if w % 2 == 1 else 'FAIL (even!)'}")

print()
print("=== 2. smoothing reduces high-frequency noise ===")
rng = np.random.default_rng(0)
T = 5000
t = np.arange(T) / C.FS_ECOG
clean = np.sin(2 * np.pi * 1.0 * t)
noisy = clean + 0.5 * rng.standard_normal(T)
ecog = np.stack([noisy, noisy], axis=1)
smoothed = savgol_smooth_all_channels(ecog)
err_noisy = np.mean((noisy - clean) ** 2)
err_smoothed = np.mean((smoothed[:, 0] - clean) ** 2)
print(f"  MSE vs clean: noisy={err_noisy:.4f} smoothed={err_smoothed:.4f}  "
      f"{'OK' if err_smoothed < err_noisy else 'FAIL'}")

print()
print("=== 3. exact t-tau/t/t+tau indexing (known ramp signal) ===")
T2, n_ch = 100, 2
ramp = np.arange(T2, dtype=float)
sig = np.stack([ramp, ramp * 10], axis=1)
tau = 5
X, (lo, hi) = build_regression_features(sig, tau_samples=tau)
ok_shape = X.shape == (T2 - 2 * tau, n_ch * 3)
ok_vals = True
for i in [0, 20]:
    t_actual = lo + i
    expect = np.array([t_actual - tau, (t_actual - tau) * 10,
                        t_actual, t_actual * 10,
                        t_actual + tau, (t_actual + tau) * 10])
    ok_vals = ok_vals and np.allclose(X[i], expect)
print(f"  shape={X.shape} range=({lo},{hi})  {'OK' if ok_shape and ok_vals else 'FAIL'}")

print()
print("=== 4. full-scale pipeline (subject 1, all channels, full train set) ===")
sd = make_synthetic_blocks(1, seed=0)
t0 = time.time()
smoothed_full = savgol_smooth_all_channels(sd.train_ecog)
dt_sg = time.time() - t0
tau_samples = 37  # placeholder value for this smoke test only, not the tuned tau
t0 = time.time()
Xf, (lo, hi) = build_regression_features(smoothed_full, tau_samples=tau_samples)
dt_feat = time.time() - t0
y_aligned = sd.train_glove[lo:hi]
ok = (Xf.shape == (sd.train_ecog.shape[0] - 2 * tau_samples, sd.n_channels * 3)
      and np.isfinite(Xf).all() and y_aligned.shape[0] == Xf.shape[0])
print(f"  savgol {dt_sg:.1f}s + features {dt_feat:.1f}s, X.shape={Xf.shape}, "
      f"target aligned: {y_aligned.shape[0]==Xf.shape[0]}  {'OK' if ok else 'FAIL'}")

print()
print("=== 5. tau=0 degenerate case ===")
X0, _ = build_regression_features(smoothed_full[:1000], tau_samples=0)
ok0 = X0.shape == (1000, sd.n_channels * 3) and np.allclose(
    X0[:, :sd.n_channels], X0[:, sd.n_channels:2 * sd.n_channels])
print(f"  shape={X0.shape}  {'OK' if ok0 else 'FAIL'}")

print()
print("ALL CHECKS RAN (read FAIL markers above if any)")
