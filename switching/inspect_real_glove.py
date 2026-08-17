"""
Diagnostic: inspect real glove (finger flexion) data BEFORE trusting the
state labels derived from it. Run in the same folder as the other files,
with data/ populated:

    python inspect_real_glove.py

For each subject, prints:
  - per-finger percentiles of the TRAIN glove trace (min/5/25/50/75/95/max)
  - the state distribution under the CURRENT threshold rule
    (config.STATE_ON_THRESHOLD_FRAC = min + frac*(max-min))
  - the state distribution under an ALTERNATIVE percentile-based rule
    (threshold = 10th percentile + frac*(90th-10th percentile)), which is
    robust to outliers/artifacts -- real glove sensors can have brief
    spikes or drift that blow out a min/max-based range
  - saves a PNG plot of the first 30s of each finger's train trace, so you
    can eyeball whether flexion actually looks like distinct move/rest
    blocks (paper's paradigm) or something else (continuous drift, offset
    baseline, clipped sensor, etc.)

This does NOT change config.py or any pipeline file -- it's read-only,
just for figuring out what the right STATE_ON_THRESHOLD_FRAC (or a better
rule entirely) should be before rerunning the full decoder.
"""
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

import config as C
from data_io import load_subject

for n in [1, 2, 3]:
    print(f"{'='*60}")
    print(f"SUBJECT {n}")
    print(f"{'='*60}")
    sd = load_subject(n)

    print()
    print("-- per-finger percentiles of TRAIN glove trace --")
    pcts = [0, 5, 10, 25, 50, 75, 90, 95, 100]
    for f in range(C.N_FINGERS):
        vals = np.percentile(sd.train_glove[:, f], pcts)
        vals_str = "  ".join(f"{p}%={v:7.2f}" for p, v in zip(pcts, vals))
        print(f"  {C.FINGER_NAMES[f]:6s}: {vals_str}")

    print()
    print("-- state distribution: ROBUST rule (p1 + frac*(p99-p1)) -- THIS IS NOW THE PIPELINE DEFAULT --")
    p_lo, p_hi = C.STATE_RANGE_ROBUST_PCT, 100.0 - C.STATE_RANGE_ROBUST_PCT
    robust_min = np.percentile(sd.train_glove, p_lo, axis=0)
    robust_max = np.percentile(sd.train_glove, p_hi, axis=0)
    thresh_robust = robust_min + C.STATE_ON_THRESHOLD_FRAC * (robust_max - robust_min)
    above_r = sd.train_glove - thresh_robust[None, :]
    any_above_r = (above_r > 0).any(axis=1)
    winner_r = np.argmax(above_r, axis=1) + 1
    state_robust = np.where(any_above_r, winner_r, C.REST_STATE)
    vals_r, counts_r = np.unique(state_robust, return_counts=True)
    dist_r = {int(v): round(float(c/counts_r.sum()), 3) for v, c in zip(vals_r, counts_r)}
    print(f"  thresholds: {np.round(thresh_robust, 2)}")
    print(f"  distribution: {dist_r}")

    print()
    print("-- state distribution: OLD literal-min/max rule (for comparison only, no longer used) --")
    finger_range = sd.train_glove.max(axis=0) - sd.train_glove.min(axis=0)
    thresh_current = sd.train_glove.min(axis=0) + C.STATE_ON_THRESHOLD_FRAC * finger_range
    above = sd.train_glove - thresh_current[None, :]
    any_above = (above > 0).any(axis=1)
    winner = np.argmax(above, axis=1) + 1
    state_current = np.where(any_above, winner, C.REST_STATE)
    vals, counts = np.unique(state_current, return_counts=True)
    dist = {int(v): round(float(c/counts.sum()), 3) for v, c in zip(vals, counts)}
    print(f"  thresholds: {np.round(thresh_current, 2)}")
    print(f"  distribution: {dist}")

    print()
    if not HAVE_MPL:
        print("-- matplotlib not installed, skipping plot (numbers above are still valid) --")
        print("   run: pip install matplotlib   -- then rerun this script for the plots")
        print()
        continue
    print("-- saving plot of first 30s of train glove trace --")
    fs = C.FS_ECOG
    T_plot = 30 * fs
    fig, axes = plt.subplots(C.N_FINGERS, 1, figsize=(10, 8), sharex=True)
    t = np.arange(T_plot) / fs
    for f in range(C.N_FINGERS):
        axes[f].plot(t, sd.train_glove[:T_plot, f], linewidth=0.8)
        axes[f].axhline(thresh_robust[f], color='g', linewidth=1.2,
                         label='ROBUST (pipeline default)')
        axes[f].axhline(thresh_current[f], color='r', linestyle='--', linewidth=0.8,
                         label='old literal min/max')
        axes[f].set_ylabel(C.FINGER_NAMES[f], fontsize=8)
        if f == 0:
            axes[f].legend(fontsize=7, loc='upper right')
    axes[-1].set_xlabel('time (s)')
    fig.suptitle(f'Subject {n}: first 30s of train glove trace')
    fig.tight_layout()
    outpath = f'glove_inspect_subject{n}.png'
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"  saved {outpath}")
    print()

print("Done. Share the printed percentiles/distributions and the 3 PNG files back.")
