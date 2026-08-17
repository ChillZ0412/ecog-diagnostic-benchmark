"""
Train-only artifact detector for AM-power features.

Motivation: every fix we tried for Subject 3's ring finger (drop channel,
log1p, percentile clip) was found AFTER observing the test-set failure --
that is retrospective, not a generalizable QC step. This script asks a
stricter question: can a heavy-tailedness check computed ONLY on the
training segment flag `sub/ch49` as suspicious, with zero access to the
test set?

If yes: we have a principled, leakage-free preprocessing step that would
have caught this automatically, on any subject, without ever having to
"know" which channel is bad in advance.
If no: the artifact is genuinely only visible in the test segment (a true
train/test distribution shift with no training-time signature), and the
correct framing is "detectable only post-hoc", which is also a real,
reportable finding -- just a different one.

Method: for every channel x band feature, compute a heavy-tailedness score
using ONLY training data:
  - kurtosis of the (log-compressed) feature distribution
  - ratio of max to 99th percentile within the training segment itself
    (a channel whose own training data has a few extreme spikes relative
    to its typical values is inherently less trustworthy, independent of
    what happens in test)

Run:
    python detect_bad_channels.py --subject 3
    python detect_bad_channels.py --subject 3 --top 15
"""
import argparse

import numpy as np

import config as C
from data_io import load_subject
from features import extract_am_features


def channel_quality_scores(X: np.ndarray, names) -> list:
    """
    Train-only heavy-tailedness score per feature column.

    Both scores use log1p first: AM power is intrinsically heavy-tailed
    (that's expected), so we score how much MORE extreme a channel's own
    training tail is relative to its own typical scale, not raw magnitude.
    """
    Xl = np.log1p(np.maximum(X, 0.0))
    out = []
    for j in range(X.shape[1]):
        col = Xl[:, j]
        mean, std = col.mean(), col.std()
        if std == 0:
            kurt = 0.0
        else:
            kurt = float(((col - mean) ** 4).mean() / std ** 4 - 3.0)  # excess kurtosis
        p50, p99 = np.percentile(col, [50, 99])
        spread = max(p99 - p50, 1e-9)
        max_ratio = float((col.max() - p50) / spread)  # how far the single max sits
                                                        # beyond the typical (p50-p99) spread,
                                                        # in units of that spread
        band, ch = names[j]
        out.append({"band": band, "channel": ch, "kurtosis": kurt,
                    "max_ratio": max_ratio, "col_index": j})
    return out


def aggregate_by_electrode(scores: list) -> dict:
    """
    Pool the per-band scores up to the PHYSICAL ELECTRODE level: if a
    channel's electrode looks suspicious in ANY of its band-filtered
    versions, the whole electrode is suspect, even if the specific
    (band, channel) feature that a downstream selector might pick looks
    clean on its own. This is the aggregation that matters for catching
    an artifact that surfaces in one band but not another on training data.
    """
    by_ch = {}
    for s in scores:
        ch = s["channel"]
        by_ch.setdefault(ch, {"channel": ch, "kurtosis_max": -np.inf,
                              "max_ratio_max": -np.inf, "bands": []})
        entry = by_ch[ch]
        entry["kurtosis_max"] = max(entry["kurtosis_max"], s["kurtosis"])
        entry["max_ratio_max"] = max(entry["max_ratio_max"], s["max_ratio"])
        entry["bands"].append((s["band"], s["kurtosis"], s["max_ratio"]))
    return by_ch


def flagged_channels(by_ch: dict, pct: float = 95.0) -> set:
    """Electrodes whose worst-band score exceeds the given percentile
    across all electrodes, using training data only."""
    kurt_vals = [e["kurtosis_max"] for e in by_ch.values()]
    ratio_vals = [e["max_ratio_max"] for e in by_ch.values()]
    kurt_thr = np.percentile(kurt_vals, pct)
    ratio_thr = np.percentile(ratio_vals, pct)
    flagged = {ch for ch, e in by_ch.items()
              if e["kurtosis_max"] >= kurt_thr or e["max_ratio_max"] >= ratio_thr}
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--top", type=int, default=10,
                    help="how many top-flagged channels to show")
    args = ap.parse_args()

    print(f"Loading subject {args.subject} (TRAINING data only)...")
    sd = load_subject(args.subject)
    Xtr, names = extract_am_features(sd.train_ecog)

    scores = channel_quality_scores(Xtr, names)

    print(f"\n{'=' * 70}")
    print(f"Top {args.top} channels by excess kurtosis (train-only)")
    print("=" * 70)
    by_kurt = sorted(scores, key=lambda d: -d["kurtosis"])[:args.top]
    print(f"{'band/ch':<14}{'kurtosis':>12}{'max_ratio':>12}")
    for s in by_kurt:
        flag = "  <-- ch49" if s["channel"] == 49 else ""
        print(f"{s['band'] + '/ch' + str(s['channel']):<14}"
              f"{s['kurtosis']:>12.1f}{s['max_ratio']:>12.1f}{flag}")

    print(f"\n{'=' * 70}")
    print(f"Top {args.top} channels by max_ratio (train-only)")
    print("=" * 70)
    by_ratio = sorted(scores, key=lambda d: -d["max_ratio"])[:args.top]
    print(f"{'band/ch':<14}{'kurtosis':>12}{'max_ratio':>12}")
    for s in by_ratio:
        flag = "  <-- ch49" if s["channel"] == 49 else ""
        print(f"{s['band'] + '/ch' + str(s['channel']):<14}"
              f"{s['kurtosis']:>12.1f}{s['max_ratio']:>12.1f}{flag}")

    # explicit lookup: where does ch49 (all bands) rank among all channels?
    print(f"\n{'=' * 70}")
    print("ch49 specifically, all bands, and its rank among all channels")
    print("=" * 70)
    all_by_kurt = sorted(scores, key=lambda d: -d["kurtosis"])
    all_by_ratio = sorted(scores, key=lambda d: -d["max_ratio"])
    n = len(scores)
    for s in scores:
        if s["channel"] == 49:
            rk = all_by_kurt.index(s) + 1
            rr = all_by_ratio.index(s) + 1
            print(f"  {s['band']}/ch49   kurtosis={s['kurtosis']:.1f} "
                  f"(rank {rk}/{n})   max_ratio={s['max_ratio']:.1f} (rank {rr}/{n})")

    # ---- electrode-level aggregation ---------------------------------------
    print(f"\n{'=' * 70}")
    print("ELECTRODE-level aggregation (worst band per physical channel)")
    print("=" * 70)
    by_ch = aggregate_by_electrode(scores)
    n_ch = len(by_ch)
    ranked = sorted(by_ch.values(), key=lambda e: -e["kurtosis_max"])
    print(f"{'channel':<10}{'worst kurtosis':>16}{'worst max_ratio':>18}")
    for e in ranked[:args.top]:
        flag = "  <-- ch49" if e["channel"] == 49 else ""
        print(f"{e['channel']:<10}{e['kurtosis_max']:>16.1f}{e['max_ratio_max']:>18.1f}{flag}")

    ch49_rank = [e["channel"] for e in ranked].index(49) + 1
    print(f"\n  ch49 electrode-level rank: {ch49_rank}/{n_ch} "
          f"(pooling across sub/gamma/fastgamma)")

    flagged = flagged_channels(by_ch, pct=95.0)
    print(f"\n  Electrodes flagged at the 95th-percentile threshold "
          f"(train-only): {sorted(flagged)}")
    print(f"  ch49 {'IS' if 49 in flagged else 'is NOT'} flagged by this rule.")

    # ---- leakage-free end-to-end test, ALL FIVE FINGERS -------------------
    # A train-only exclusion rule must be judged on its net effect across
    # every finger, not just the one we already knew was broken -- otherwise
    # we would just be cherry-picking a fix for a known problem again.
    if 49 in flagged:
        print(f"\n{'=' * 70}")
        print("End-to-end test: exclude flagged electrodes' features BEFORE")
        print("forward selection (train-only decision), then fit + evaluate")
        print("ON ALL FIVE FINGERS -- checking for side effects, not just ring")
        print("=" * 70)
        from data_io import load_subject as _load
        from features import downsample_target, align
        from memory_stack import build_xy
        from regressor import fit_wiener
        from selection import forward_select

        sd2 = load_subject(args.subject)
        Xtr, names2 = extract_am_features(sd2.train_ecog)
        ytr = downsample_target(sd2.train_glove)
        Xtr, ytr = align(Xtr, ytr)
        Xte, _ = extract_am_features(sd2.test_ecog)
        yte = downsample_target(sd2.test_glove)
        Xte, yte = align(Xte, yte)

        keep_cols = [j for j, (b, c) in enumerate(names2) if c not in flagged]
        print(f"  candidate pool: {len(names2)} -> {len(keep_cols)} features "
              f"after excluding {len(flagged)} flagged electrode(s): "
              f"{sorted(flagged)}\n")

        Xtr_f = Xtr[:, keep_cols]
        Xte_f = Xte[:, keep_cols]
        names_f = [names2[j] for j in keep_cols]

        paper_ref = {"thumb": 0.69, "index": 0.46, "middle": 0.58,
                    "ring": 0.58, "little": 0.63}
        print(f"  {'finger':<10}{'baseline r':>12}{'excluded r':>13}{'delta':>9}"
              f"{'paper':>8}")
        print("  " + "-" * 54)
        for f_idx, finger in enumerate(C.FINGER_NAMES):
            # baseline: full candidate pool, no exclusion
            sel_base = forward_select(Xtr, ytr[:, f_idx], feature_names=names2)
            Dtr_b, dtr_b = build_xy(Xtr, ytr[:, f_idx:f_idx + 1], columns=sel_base.selected)
            fit_b = fit_wiener(Dtr_b, dtr_b)
            Dte_b, dte_b = build_xy(Xte, yte[:, f_idx:f_idx + 1], columns=sel_base.selected)
            r_base = fit_b.score(Dte_b, dte_b)

            # excluded: flagged electrodes removed from the candidate pool
            sel_ex = forward_select(Xtr_f, ytr[:, f_idx], feature_names=names_f)
            Dtr_e, dtr_e = build_xy(Xtr_f, ytr[:, f_idx:f_idx + 1], columns=sel_ex.selected)
            fit_e = fit_wiener(Dtr_e, dtr_e)
            Dte_e, dte_e = build_xy(Xte_f, yte[:, f_idx:f_idx + 1], columns=sel_ex.selected)
            r_ex = fit_e.score(Dte_e, dte_e)

            print(f"  {finger:<10}{r_base:>12.3f}{r_ex:>13.3f}"
                  f"{r_ex - r_base:>+9.3f}{paper_ref[finger]:>8.2f}")

    print("\nInterpretation:")
    print("  If ch49 ranks near the TOP at the ELECTRODE level using TRAINING")
    print("  data alone -- even if the SPECIFIC (sub, ch49) feature that got")
    print("  selected looked clean in training -- a per-electrode QC rule")
    print("  (pool all bands, flag if ANY band is extreme) would catch it")
    print("  automatically, with zero access to the test set.")
    print("  Check the table above: the exclusion rule is only worth adopting")
    print("  if ring improves substantially WITHOUT dragging other fingers down.")


if __name__ == "__main__":
    main()
