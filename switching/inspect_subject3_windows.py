"""
Compare glove waveform structure at three points in subject 3's recording:
start, middle, and near the end. Checks whether the sustained multi-finger
co-activation seen in the first 30s is a warm-up/practice artifact or
persists throughout. Run in the same folder as the other files:

    python inspect_subject3_windows.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data_io import load_subject
from state_labels import make_state_labels

n = 3
sd = load_subject(n)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=n)

fs = C.FS_ECOG
window_sec = 30
starts_sec = [0, 200, C.TRAIN_SECONDS - window_sec]  # start, middle, near-end

fig, axes = plt.subplots(C.N_FINGERS, len(starts_sec), figsize=(15, 9), sharey='row')

for col, start_sec in enumerate(starts_sec):
    lo = start_sec * fs
    hi = lo + window_sec * fs
    t = np.arange(window_sec * fs) / fs
    for f in range(C.N_FINGERS):
        ax = axes[f, col]
        ax.plot(t, sd.train_glove[lo:hi, f], linewidth=0.8)
        ax.axhline(sl.thresholds[f], color='g', linewidth=1.0)
        if col == 0:
            ax.set_ylabel(C.FINGER_NAMES[f], fontsize=8)
        if f == 0:
            ax.set_title(f'train t={start_sec}-{start_sec+window_sec}s')
    axes[-1, col].set_xlabel('time (s)')

fig.suptitle('Subject 3: glove trace at start / middle / near-end of training recording')
fig.tight_layout()
fig.savefig('subject3_windows_compare.png', dpi=120)
print('saved subject3_windows_compare.png')

print()
print('per-window state distribution (using the ROBUST thresholds already fit on full train set):')
for start_sec in starts_sec:
    lo, hi = start_sec * fs, (start_sec + window_sec) * fs
    seg_state = sl.train_state[lo:hi]
    vals, counts = np.unique(seg_state, return_counts=True)
    dist = {int(v): round(float(c/counts.sum()), 3) for v, c in zip(vals, counts)}
    print(f'  t={start_sec}-{start_sec+window_sec}s: {dist}')
