"""
Measure inference_ms for Method 2 -- the last unmeasured field flagged in
the handoff documents. Matches Wiener's methodology: time how long it
takes to produce ONE prediction from already-extracted signal, not
counting model training.

Run in the same folder as the other files, with data/ populated:

    python measure_inference_ms.py

Two things are timed separately, since the switching decoder has more
moving parts than Wiener's single matmul:
  1. "estimated" full pipeline: AR feature extraction (classifier) + group
     lasso predict (state) + H_k predict (finger positions) -- this is
     the realistic end-to-end deployment path.
  2. Just the H_k regression step alone (given features already computed)
     -- the closest analogue to what Wiener's own inference_ms measures
     (Wiener's pipeline also assumes features are already streaming in).

Uses N=200 repeated single-sample predictions (after a warm-up), reports
mean/median/std, mirroring how these things are usually benchmarked.
"""
import time

import numpy as np

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder
from regression_features import savgol_smooth_all_channels, build_regression_features
from ar_features import extract_ar_features

TUNED = {
    1: dict(tau_samples=75, M=30),
    2: dict(tau_samples=150, M=30),
    3: dict(tau_samples=500, M=30),
}
N_TIMED = 200
N_WARMUP = 20

for n in [1, 2, 3]:
    print(f"{'='*60}")
    print(f"SUBJECT {n}")
    print(f"{'='*60}")

    sd = load_subject_clean(n)
    sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)
    tau_samples, M = TUNED[n]["tau_samples"], TUNED[n]["M"]

    bundle = fit_decoder(
        sd.train_ecog, sd.train_glove, sl.train_state,
        n_select_channels=15, lambda_s=0.0, tau_samples=tau_samples,
        lambda_k=1.0, M=M,
    )

    # Precompute the full test-set regression features + AR features once
    # (this is the "signal has already arrived" starting point, same
    # assumption Wiener's own inference_ms makes)
    smoothed_test = savgol_smooth_all_channels(sd.test_ecog)
    X_reg_test, (lo, hi) = build_regression_features(smoothed_test, bundle.tau_samples)
    afs_test = extract_ar_features(sd.test_ecog, channels=bundle.top_channels,
                                    shifts_ms=bundle.ar_shifts_ms, n_coeffs_keep=2)
    cls_features_full = afs_test.features[lo:hi]  # align to same [lo,hi) as X_reg_test

    T = X_reg_test.shape[0]
    rng = np.random.default_rng(0)
    sample_idx = rng.integers(0, T, size=N_TIMED + N_WARMUP)

    # --- Timer 1: H_k regression predict only (one sample at a time) ---
    times_hk = []
    dummy_model = list(bundle.Hk_models.values())[0]
    for i, idx in enumerate(sample_idx):
        x = X_reg_test[idx:idx + 1]
        t0 = time.perf_counter()
        _ = dummy_model.predict(x)
        dt = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_hk.append(dt)
    times_hk = np.array(times_hk) * 1000  # ms

    # --- Timer 2: full "estimated" pipeline (classify state -> select H_k -> predict) ---
    times_full = []
    for i, idx in enumerate(sample_idx):
        x_cls = cls_features_full[idx:idx + 1]
        x_reg = X_reg_test[idx:idx + 1]
        t0 = time.perf_counter()
        state_pred = bundle.classifier.predict(x_cls)[0]
        model = bundle.Hk_models.get(state_pred, dummy_model)
        _ = model.predict(x_reg)
        dt = time.perf_counter() - t0
        if i >= N_WARMUP:
            times_full.append(dt)
    times_full = np.array(times_full) * 1000  # ms

    print(f"  H_k regression only:  mean={times_hk.mean():.3f}ms  "
          f"median={np.median(times_hk):.3f}ms  std={times_hk.std():.3f}ms")
    print(f"  Full 'estimated' pipeline (classify+select+predict): "
          f"mean={times_full.mean():.3f}ms  median={np.median(times_full):.3f}ms  "
          f"std={times_full.std():.3f}ms")
    print()

print("Done. Compare to Wiener's <1ms. Note this does NOT include the AR feature")
print("extraction / Savitzky-Golay smoothing cost upstream, which happens on a")
print("streaming window and is a separate (also cheap, but not timed here) cost.")
