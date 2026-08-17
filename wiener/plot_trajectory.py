"""
Generate Figure 1 for the Results slide: predicted vs. actual finger-angle
trajectory for the best subject/finger (Subject 1, Index — matches the
paper's r=0.71 exactly).

Run:
    python plot_trajectory.py
    python plot_trajectory.py --subject 1 --finger index --window 60   # customize

Output: trajectory_S1_index.png in the current directory. Insert this
directly into the FIGURE PLACEHOLDER box on the slide.
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data_io import load_subject
from features import extract_am_features, downsample_target, align
from memory_stack import build_xy
from regressor import fit_wiener, pearson_r
from selection import forward_select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--finger", default="index", choices=C.FINGER_NAMES)
    ap.add_argument("--window", type=float, default=40.0,
                    help="seconds of test data to display (default 40s)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="start offset in seconds into the test set")
    args = ap.parse_args()
    f = C.FINGER_NAMES.index(args.finger)

    print(f"Loading subject {args.subject}...")
    sd = load_subject(args.subject)
    Xtr, names = extract_am_features(sd.train_ecog)
    ytr = downsample_target(sd.train_glove)
    Xtr, ytr = align(Xtr, ytr)
    Xte, _ = extract_am_features(sd.test_ecog)
    yte = downsample_target(sd.test_glove)
    Xte, yte = align(Xte, yte)

    print("Running forward selection + fit...")
    sel = forward_select(Xtr, ytr[:, f], feature_names=names)
    Dtr, dtr = build_xy(Xtr, ytr[:, f:f + 1], columns=sel.selected)
    fit = fit_wiener(Dtr, dtr)
    Dte, dte = build_xy(Xte, yte[:, f:f + 1], columns=sel.selected)
    pred = fit.predict(Dte)
    target = dte.ravel()
    r = pearson_r(pred, target)
    print(f"Test r = {r:.3f}  ({len(sel.selected)} features)")

    fs = C.FS_FEATURE  # 25 Hz
    i0 = int(args.start * fs)
    i1 = min(i0 + int(args.window * fs), len(target))
    t = np.arange(i0, i1) / fs

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(t, target[i0:i1], color="#1F3864", linewidth=1.8, label="Actual")
    ax.plot(t, pred[i0:i1], color="#C00000", linewidth=1.3, linestyle="--",
            alpha=0.85, label="Predicted")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Finger flexion (a.u.)", fontsize=11)
    ax.set_title(f"Subject {args.subject} — {args.finger.capitalize()} finger  "
                f"(test r = {r:.2f})", fontsize=13, fontweight="bold", color="#1F3864")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = f"trajectory_S{args.subject}_{args.finger}.png"
    fig.savefig(out, dpi=200)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
