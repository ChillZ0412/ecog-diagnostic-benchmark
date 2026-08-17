"""
Stage 2 verification.

Run:  python test_filters.py

Checks, in order of how badly a failure would corrupt later stages:

  1. Frequency response of each designed filter (ripple / stopband / DC).
  2. BAND SELECTIVITY: a pure tone placed inside each band must survive in
     that band and be strongly attenuated in the other two.
  3. ZERO PHASE: the filtered tone must be time-aligned with the input
     (lag = 0 samples). A non-zero lag here would silently shift the ECoG
     relative to the finger trace and wreck the k=25 memory stack.
  4. AMPLITUDE PRESERVATION: passband gain ~ 1.0, so AM features are not
     rescaled arbitrarily per band.
  5. SHAPE / SPEED on a realistic 400 s x 62 ch array.
"""
import time

import numpy as np
from scipy import signal

import config as C
from filters import (design_bandpass, describe_filter, apply_fir,
                     decompose_bands)


def _ok(flag: bool) -> str:
    return "OK  " if flag else "FAIL"


def test_responses() -> bool:
    print("\n[1] Filter frequency responses")
    all_ok = True
    for name in C.BANDS:
        taps = design_bandpass(name)
        info = describe_filter(name, taps)
        # sanity thresholds: passband gain near 1, decent DC/stopband rejection
        good = (abs(info["center_gain"] - 1.0) < 0.05
                and info["dc_db"] < -30.0
                and info["passband_ripple_db"] < 3.0)
        all_ok &= good
        print(f"  [{_ok(good)}] {name:10s} {info['range_hz']}  N={info['numtaps']:5d}  "
              f"gain={info['center_gain']:.3f}  DC={info['dc_db']:6.1f}dB  "
              f"ripple={info['passband_ripple_db']:.2f}dB  "
              f"stop={info['stopband_db']:6.1f}dB")
    return all_ok


def test_band_selectivity() -> bool:
    """Put a tone in each band; it must pass there and die elsewhere."""
    print("\n[2] Band selectivity (tone in-band vs out-of-band)")
    fs = C.FS_ECOG
    t = np.arange(30 * fs) / fs           # 30 s
    probes = {"sub": 20.0, "gamma": 80.0, "fastgamma": 150.0}

    taps = {n: design_bandpass(n) for n in C.BANDS}
    all_ok = True

    for probe_band, f0 in probes.items():
        x = np.sin(2 * np.pi * f0 * t).astype(C.FIR_DTYPE)[:, None]
        mid = slice(5 * fs, 25 * fs)      # ignore filter edge transients
        amps = {}
        for name in C.BANDS:
            y = apply_fir(x, taps[name])
            amps[name] = float(np.abs(y[mid, 0]).max())

        in_amp = amps[probe_band]
        out_amp = max(v for k, v in amps.items() if k != probe_band)
        good = in_amp > 0.9 and out_amp < 0.1
        all_ok &= good
        others = ", ".join(f"{k}={v:.3f}" for k, v in amps.items() if k != probe_band)
        print(f"  [{_ok(good)}] {f0:5.0f} Hz -> {probe_band:10s} keeps {in_amp:.3f}   "
              f"leaks: {others}")
    return all_ok


def test_zero_phase() -> bool:
    """The single most dangerous silent failure: a time shift."""
    print("\n[3] Zero-phase alignment (lag must be 0 samples)")
    fs = C.FS_ECOG
    t = np.arange(30 * fs) / fs
    probes = {"sub": 20.0, "gamma": 80.0, "fastgamma": 150.0}
    mid = slice(10 * fs, 20 * fs)
    all_ok = True

    for name, f0 in probes.items():
        x = np.sin(2 * np.pi * f0 * t).astype(np.float64)
        y = apply_fir(x[:, None], design_bandpass(name))[:, 0].astype(np.float64)
        a, b = y[mid], x[mid]
        xc = np.correlate(a - a.mean(), b - b.mean(), "full")
        lag = int(np.argmax(xc) - (len(b) - 1))
        good = lag == 0
        all_ok &= good
        print(f"  [{_ok(good)}] {name:10s} lag = {lag:+d} samples")

    # And the causal variant SHOULD show the expected (N-1)/2 delay.
    # NOTE: the probe must be NON-PERIODIC. A pure 80 Hz tone at fs=1000 has a
    # period of 12.5 samples, and the expected 500-sample delay is exactly 40
    # periods -- cross-correlation cannot distinguish it from zero lag. Use
    # broadband noise instead.
    taps = design_bandpass("gamma")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(len(t))
    y = apply_fir(x[:, None], taps, causal=True)[:, 0].astype(np.float64)
    a, b = y[mid], x[mid]
    xc = np.correlate(a - a.mean(), b - b.mean(), "full")
    lag = int(np.argmax(xc) - (len(b) - 1))
    expected = (len(taps) - 1) // 2
    good = abs(lag - expected) <= 1
    all_ok &= good
    print(f"  [{_ok(good)}] causal mode delay = {lag} samples (expected {expected})")
    return all_ok


def test_amplitude() -> bool:
    print("\n[4] Passband amplitude preservation (gain ~ 1.0)")
    fs = C.FS_ECOG
    t = np.arange(30 * fs) / fs
    mid = slice(5 * fs, 25 * fs)
    probes = {"sub": 20.0, "gamma": 80.0, "fastgamma": 150.0}
    all_ok = True
    for name, f0 in probes.items():
        x = np.sin(2 * np.pi * f0 * t).astype(C.FIR_DTYPE)[:, None]
        y = apply_fir(x, design_bandpass(name))
        gain = float(np.abs(y[mid, 0]).max())
        good = abs(gain - 1.0) < 0.05
        all_ok &= good
        print(f"  [{_ok(good)}] {name:10s} gain = {gain:.4f}")
    return all_ok


def test_realistic_shape_and_speed() -> bool:
    print("\n[5] Realistic array: 400 s x 62 ch")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((C.TRAIN_SECONDS * C.FS_ECOG, 62)).astype(C.FIR_DTYPE)
    all_ok = True
    total = 0.0
    for name, Y in decompose_bands(X):
        t0 = time.time()
        dt = time.time() - t0
        good = Y.shape == X.shape and np.isfinite(Y).all()
        all_ok &= good
        total += dt
        print(f"  [{_ok(good)}] {name:10s} out {Y.shape}  {Y.dtype}  "
              f"{Y.nbytes / 1e6:.0f} MB")
    print(f"  input {X.nbytes / 1e6:.0f} MB; bands are generated one at a time")
    return all_ok


def main():
    t0 = time.time()
    results = [
        test_responses(),
        test_band_selectivity(),
        test_zero_phase(),
        test_amplitude(),
        test_realistic_shape_and_speed(),
    ]
    print(f"\n{'=' * 60}")
    print(f"STAGE 2: {'ALL CHECKS PASSED' if all(results) else 'PROBLEMS FOUND'}"
          f"   ({time.time() - t0:.1f}s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
