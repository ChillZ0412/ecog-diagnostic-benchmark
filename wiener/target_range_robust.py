"""
Robust (percentile-based) min/max for the target (dataglove) values, per
subject per finger -- contrasted with literal min/max.

Why this matters: literal min/max is a single-sample statistic, extremely
sensitive to one outlier. This is exactly the class of bug already found and
fixed on the switching-linear-models side (S3's literal min/max thresholds
were corrupted by an extreme outlier; the fix was p1/p99 percentiles). This
script checks whether the same issue is present for Wiener's target data,
using ONLY training-segment statistics (consistent with the train-only
philosophy used everywhere else in this pipeline -- e.g. electrode_qc_mask).

Note: target units are raw dataglove sensor readings (arbitrary units, not
calibrated to physical degrees -- confirmed against the official dataset
description, which never specifies a physical unit). This script reports
values in those native units; it does not attempt to convert them.

Run:
    python target_range_robust.py --subject 1
    python target_range_robust.py --subject 1 --pct 1        # p1/p99 (default)
    python target_range_robust.py --subject 1 --pct 0.5      # p0.5/p99.5, less aggressive trim
"""
import argparse

import numpy as np

import config as C
from data_io import load_subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--pct", type=float, default=1.0,
                    help="trim percentile per side (default 1.0 -> p1/p99)")
    ap.add_argument("--split", default="train", choices=["train", "test", "both"],
                    help="which segment to report (default train, matching "
                         "the train-only philosophy used elsewhere in this "
                         "pipeline)")
    args = ap.parse_args()

    sd = load_subject(args.subject)
    segments = {"train": sd.train_glove, "test": sd.test_glove}
    if args.split != "both":
        segments = {args.split: segments[args.split]}

    lo_pct, hi_pct = args.pct, 100.0 - args.pct

    for seg_name, glove in segments.items():
        print(f"\n{'=' * 78}")
        print(f"Subject {args.subject} — {seg_name} segment "
              f"(robust range = p{lo_pct:g}/p{hi_pct:g})")
        print("=" * 78)
        print(f"{'finger':<8}{'literal min':>13}{'literal max':>13}"
              f"{'robust min':>13}{'robust max':>13}{'n_outliers':>12}")
        print("-" * 78)

        for f, name in enumerate(C.FINGER_NAMES):
            col = glove[:, f]
            lit_min, lit_max = col.min(), col.max()
            rob_min, rob_max = np.percentile(col, [lo_pct, hi_pct])
            n_out = int(((col < rob_min) | (col > rob_max)).sum())
            flag = "  <-- literal range inflated by outliers" \
                if (lit_max - lit_min) > 1.5 * (rob_max - rob_min) else ""
            print(f"{name:<8}{lit_min:>13.3f}{lit_max:>13.3f}"
                  f"{rob_min:>13.3f}{rob_max:>13.3f}{n_out:>12d}{flag}")

    print("\nNote: target values are raw dataglove sensor readings (arbitrary")
    print("units), not calibrated physical degrees -- confirmed against the")
    print("official BCI Competition IV dataset 4 description, which never")
    print("specifies a physical unit for 'finger position'.")
    print("\nIf any finger shows 'literal range inflated by outliers' above,")
    print("that finger's literal min/max should NOT be used for any")
    print("downstream normalization/threshold decision -- use the robust")
    print("(percentile) range instead, matching the fix already applied on")
    print("the switching-linear-models side.")


if __name__ == "__main__":
    main()
