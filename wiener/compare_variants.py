"""
Compare saved reproduction runs.

Run:
    python compare_variants.py result.json result_clip.json result_log.json
    python compare_variants.py result.json result_f55.json result_f65.json --labels baseline f=0.55 f=0.65

Produces the two tables the report needs:

  1. A headline comparison (all-finger mean, official metric, per-subject).
  2. A per-finger delta table showing exactly where variants diverge.

WHY THE SECOND TABLE MATTERS
Any variant that changes the FEATURES also re-runs forward selection on those
changed features, so a variant's effect is confounded with greedy-selection
path change. Before attributing a +/-0.05 difference to the transform itself,
compare it against the spread produced by simply perturbing the inner split
fraction, which changes the selection path while leaving the features
untouched. Differences smaller than that spread are path noise, not signal.
"""
import argparse
import json

import numpy as np

FINGERS = ["thumb", "index", "middle", "ring", "little"]
OFFICIAL = [0, 1, 2, 4]          # competition excluded the ring finger


def load(path):
    with open(path) as fh:
        return json.load(fh)


def matrix(res, key="test_r_rule"):
    """(n_subjects, 5) array of the chosen metric."""
    return np.array([[f[key] for f in r["fingers"]] for r in res], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--key", default="test_r_rule")
    args = ap.parse_args()

    runs = [load(f) for f in args.files]
    labels = args.labels or [f.replace(".json", "") for f in args.files]
    if len(labels) != len(runs):
        labels = [f.replace(".json", "") for f in args.files]

    mats = [matrix(r, args.key) for r in runs]
    subjects = [r["subject"] for r in runs[0]]
    paper = matrix(runs[0], "paper_r")

    # ---- headline ----------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("HEADLINE COMPARISON")
    print("=" * 78)
    print(f"{'variant':<16}{'all fingers':>13}{'official':>11}"
          + "".join(f"{'S' + str(s):>9}" for s in subjects))
    print("-" * 78)
    print(f"{'paper':<16}{np.nanmean(paper):13.3f}"
          f"{np.nanmean(paper[:, OFFICIAL]):11.3f}"
          + "".join(f"{np.nanmean(paper[i]):9.3f}" for i in range(len(subjects))))
    for lab, m in zip(labels, mats):
        print(f"{lab:<16}{np.nanmean(m):13.3f}"
              f"{np.nanmean(m[:, OFFICIAL]):11.3f}"
              + "".join(f"{np.nanmean(m[i]):9.3f}" for i in range(len(subjects))))

    # ---- per-finger --------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("PER-FINGER DETAIL  (delta vs the first variant listed)")
    print("=" * 78)
    base = mats[0]
    for si, subj in enumerate(subjects):
        print(f"\nSubject {subj}")
        print(f"  {'source':<16}" + "".join(f"{f.capitalize():>10}" for f in FINGERS))
        print("  " + "-" * 66)
        print(f"  {'paper':<16}" + "".join(f"{v:10.2f}" for v in paper[si]))
        for lab, m in zip(labels, mats):
            print(f"  {lab:<16}" + "".join(f"{v:10.2f}" for v in m[si]))
        for lab, m in zip(labels[1:], mats[1:]):
            d = m[si] - base[si]
            marks = "".join(f"{v:+10.2f}" for v in d)
            print(f"  {'d ' + lab:<16}{marks}")

    # ---- spread ------------------------------------------------------------
    if len(mats) > 1:
        stack = np.stack(mats)                       # (variants, subjects, 5)
        spread = np.nanmax(stack, 0) - np.nanmin(stack, 0)
        print(f"\n{'=' * 78}")
        print("SPREAD ACROSS VARIANTS  (max - min, per finger)")
        print("=" * 78)
        print(f"  {'subject':<10}" + "".join(f"{f.capitalize():>10}" for f in FINGERS))
        print("  " + "-" * 60)
        for si, subj in enumerate(subjects):
            print(f"  {subj:<10}" + "".join(f"{v:10.2f}" for v in spread[si]))
        print(f"\n  median spread = {np.nanmedian(spread):.3f}   "
              f"max spread = {np.nanmax(spread):.3f}")
        print("  Treat any variant difference below the median spread as")
        print("  greedy-selection path noise rather than a real effect.")
    print()


if __name__ == "__main__":
    main()
