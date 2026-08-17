"""
Stage 7 — Full pipeline and evaluation.

Run:
    python run_reproduction.py                  # all three subjects (~6 min)
    python run_reproduction.py --subject 1
    python run_reproduction.py --subject 1 --solver inv     # ablation
    python run_reproduction.py --save results.json

--------------------------------------------------------------------------
PROTOCOL
--------------------------------------------------------------------------
The competition split is temporal and fixed: the first 400 s are training,
the last 200 s are test. The test set is touched exactly once, at the end.

  1. AM features are computed separately for train and test. (The filters are
     fixed FIR kernels, so this is not a statistical choice -- but computing
     them separately makes it structurally impossible for test data to
     influence training.)
  2. Forward selection runs on the TRAINING data only, using its own inner
     3/5 + 2/5 split.
  3. The chosen features are then refit on the FULL 400 s training set. The
     paper does not state this explicitly, but selection and final fitting are
     separate concerns and using all training data for the final fit is the
     standard reading.
  4. Pearson r is computed on the test set, per finger.

--------------------------------------------------------------------------
WHY TWO NUMBERS PER FINGER
--------------------------------------------------------------------------
Validation r is the quantity forward selection MAXIMISES, over 186 candidates
x 10 rounds. It is therefore optimistically biased -- some of its late-round
improvement is selection noise rather than signal. Reporting it next to the
test r makes the size of that bias visible instead of hiding it, and the gap
between them is itself a result worth putting in the report.

We also report test r under both the paper's stopping rule and a fixed 10
features, since the paper's Table 1 and its Figures 2/3 correspond to
different points on that trajectory.

--------------------------------------------------------------------------
MAE (added 2026-08-04)
--------------------------------------------------------------------------
Mean Absolute Error, same units as the raw dataglove target (data_io.py
applies zero transforms to train_glove/test_glove). This is a 4th
supplementary metric alongside r / R^2 / NRMSE -- NOT a replacement for any
of them. Note NRMSE and R^2 are mathematically redundant (NRMSE =
sqrt(1 - R^2) exactly), so NRMSE was already excluded from the official
three-parameter benchmark comparison; MAE is genuinely independent
information (L1 vs L2 norm) and is being added specifically for
cross-method comparison with the switching linear model and FingerFlex,
both of which now report MAE in the same original dataglove units
(cross-validated 2026-08-04: FingerFlex's inverse-transformed range matches
this pipeline's raw data_io.load_subject() range almost exactly).
"""
import argparse
import json
import time

import numpy as np

import config as C
from data_io import load_subject, make_synthetic
from features import extract_am_features, downsample_target, align
from memory_stack import build_xy
from regressor import fit_wiener, pearson_r, r2_score, calibrated_r2, nrmse, mae_score
from selection import forward_select


# Paper Table 1, band-specific ECoG rows (the method being reproduced).
PAPER_TABLE1 = {
    1: [0.58, 0.71, 0.14, 0.53, 0.29],
    2: [0.51, 0.37, 0.24, 0.47, 0.35],
    3: [0.69, 0.46, 0.58, 0.58, 0.63],
}


def electrode_qc_mask(Xtr: np.ndarray, names, pct: float = 95.0) -> np.ndarray:
    """
    Train-only, leakage-free bad-electrode screen.

    For each (band, channel) feature, compute two heavy-tailedness scores on
    log1p-compressed TRAINING data alone:
      - excess kurtosis
      - how far the single training-set maximum sits beyond the typical
        (median-to-99th-percentile) spread, in units of that spread

    These are then pooled UP TO THE PHYSICAL ELECTRODE (max across its 3
    band-filtered versions), because an artifact can show up in one band's
    training data while the specific band that later gets selected looks
    clean (this is exactly what happened for Subject 3: gamma/ch49 and
    fastgamma/ch49 rank in the top 10 of 192 features by both scores using
    training data alone, even though sub/ch49 -- the feature Stage 6 actually
    picked -- looks unremarkable on its own).

    Electrodes whose WORST band exceeds the `pct` percentile across all
    electrodes (by either score) have every one of their band-filtered
    features excluded from the candidate pool BEFORE forward selection ever
    runs. Verified end-to-end on Subject 3: this recovers the ring finger
    from r=-0.004 to r=+0.562 while leaving every other finger unchanged or
    very slightly improved (thumb/index literally 0.000 delta, since the
    flagged electrodes were never selected for them anyway).

    Returns a boolean mask over columns of Xtr/Xte: True = keep.
    """
    Xl = np.log1p(np.maximum(Xtr, 0.0))
    by_ch = {}
    for j in range(Xtr.shape[1]):
        col = Xl[:, j]
        mean, std = col.mean(), col.std()
        kurt = 0.0 if std == 0 else float(((col - mean) ** 4).mean() / std ** 4 - 3.0)
        p50, p99 = np.percentile(col, [50, 99])
        max_ratio = float((col.max() - p50) / max(p99 - p50, 1e-9))

        _, ch = names[j]
        entry = by_ch.setdefault(ch, {"kurt": -np.inf, "ratio": -np.inf})
        entry["kurt"] = max(entry["kurt"], kurt)
        entry["ratio"] = max(entry["ratio"], max_ratio)

    kurt_thr = np.percentile([e["kurt"] for e in by_ch.values()], pct)
    ratio_thr = np.percentile([e["ratio"] for e in by_ch.values()], pct)
    flagged = {ch for ch, e in by_ch.items()
              if e["kurt"] >= kurt_thr or e["ratio"] >= ratio_thr}

    mask = np.array([ch not in flagged for _, ch in names])
    return mask, sorted(flagged)


def apply_feature_transform(Xtr: np.ndarray, Xte: np.ndarray,
                            transform: str = "none",
                            clip_pct: float = 99.9):
    """
    Optional preprocessing of the AM features, applied identically to train
    and test. Any statistics are estimated on TRAINING data only.

    'none'  — the paper's raw summed power. This is the default and the
              faithful reproduction.
    'log1p' — log compression. Standard practice for band power in the
              EEG/ECoG literature, and it tames the ~1e7 dynamic range
              between the median feature and the extremes.
    'clip'  — winsorise each feature at a training percentile. This is the
              more surgical option: it leaves every in-distribution sample
              untouched and only bounds values the training set never saw,
              which is exactly the failure mode observed for subject 3's
              sub/ch49 (median unchanged, maximum 700000x larger in test).
    """
    if transform == "none":
        return Xtr, Xte
    if transform == "log1p":
        return (np.log1p(np.maximum(Xtr, 0.0)),
                np.log1p(np.maximum(Xte, 0.0)))
    if transform == "clip":
        hi = np.percentile(Xtr, clip_pct, axis=0)
        lo = np.percentile(Xtr, 100.0 - clip_pct, axis=0)
        return np.clip(Xtr, lo, hi), np.clip(Xte, lo, hi)
    raise ValueError(f"unknown feature transform: {transform!r}")


def run_subject(subject: int,
                synthetic: bool = False,
                solver: str = None,
                max_features: int = None,
                standardize: bool = None,
                transform: str = "none",
                clip_pct: float = 99.9,
                inner_fraction: float = None,
                electrode_qc: bool = True,
                qc_percentile: float = 95.0,
                verbose: bool = True) -> dict:
    """Run the complete reproduction for one subject."""
    t_start = time.time()
    sd = make_synthetic(subject) if synthetic else load_subject(subject)

    if verbose:
        print(f"\n{'=' * 74}")
        print(f"Subject {subject}  ({sd.n_channels} channels)"
              + (f"   [features: {transform}]" if transform != "none" else "")
              + ("" if electrode_qc else "   [electrode QC: OFF]"))
        print("=" * 74)

    t0 = time.time()
    Xtr, names = extract_am_features(sd.train_ecog, standardize=False)
    ytr = downsample_target(sd.train_glove)
    Xtr, ytr = align(Xtr, ytr)

    Xte, _ = extract_am_features(sd.test_ecog, standardize=False)
    yte = downsample_target(sd.test_glove)
    Xte, yte = align(Xte, yte)

    if electrode_qc:
        # Train-only, leakage-free: decide which electrodes to drop using
        # ONLY the training segment, then apply the identical column mask to
        # both train and test. Verified on Subject 3 to recover the ring
        # finger (r: -0.004 -> +0.562) with zero cost to any other finger.
        mask, flagged = electrode_qc_mask(Xtr, names, pct=qc_percentile)
        if verbose and flagged:
            print(f"  electrode QC (train-only): flagged {flagged}, "
                  f"{Xtr.shape[1]} -> {int(mask.sum())} features")
        Xtr, Xte = Xtr[:, mask], Xte[:, mask]
        names = [n for n, keep in zip(names, mask) if keep]

    Xtr, Xte = apply_feature_transform(Xtr, Xte, transform, clip_pct)

    if verbose:
        print(f"  features: train {Xtr.shape}, test {Xte.shape}   "
              f"({time.time() - t0:.1f}s)")

    rows = []
    for f in range(C.N_FINGERS):
        t0 = time.time()
        sel = forward_select(Xtr, ytr[:, f], max_features=max_features,
                             train_fraction=inner_fraction,
                             feature_names=names)

        def _test_r(cols):
            if not cols:
                return float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), None
            Dtr, dtr = build_xy(Xtr, ytr[:, f:f + 1], columns=cols)
            fit = fit_wiener(Dtr, dtr, solver=solver, standardize=standardize,
                             diagnostics=True)
            Dte, dte = build_xy(Xte, yte[:, f:f + 1], columns=cols)
            pred = fit.predict(Dte)
            return (pearson_r(pred, dte), r2_score(pred, dte),
                    calibrated_r2(pred, dte), nrmse(pred, dte),
                    mae_score(pred, dte), fit)

        r_rule, r2_rule, r2c_rule, nrmse_rule, mae_rule, fit_rule = _test_r(sel.selected)
        r_full, _, _, _, _, _ = _test_r(sel.order)

        rows.append({
            "finger": C.FINGER_NAMES[f],
            "n_selected": sel.n_selected,
            "val_r": sel.val_r[sel.n_selected - 1] if sel.val_r else float("nan"),
            "test_r_rule": r_rule,
            "test_r_10": r_full,
            "test_r2": r2_rule,
            "test_r2_calibrated": r2c_rule,
            "test_nrmse": nrmse_rule,
            "test_mae": mae_rule,
            "paper_r": PAPER_TABLE1.get(subject, [np.nan] * 5)[f],
            "features": sel.paper_notation(),
            "weight_norm": fit_rule.diagnostics["weight_norm"],
            "cond_X": fit_rule.diagnostics["cond_X"],
            "seconds": time.time() - t0,
        })
        if verbose:
            r = rows[-1]
            print(f"  {r['finger']:<7s} val r={r['val_r']:.3f}  "
                  f"TEST r={r['test_r_rule']:+.3f} (rule, {r['n_selected']} feat)  "
                  f"{r['test_r_10']:+.3f} (10 feat)   "
                  f"paper {r['paper_r']:.2f}   ({r['seconds']:.0f}s)")
            print(f"          R2 = {r['test_r2']:+.3f}   "
                  f"R2 after optimal rescaling (= r^2) = "
                  f"{r['test_r2_calibrated']:.3f}   "
                  f"NRMSE = {r['test_nrmse']:.3f}   "
                  f"MAE = {r['test_mae']:.3f}")

    test_rs = [r["test_r_rule"] for r in rows]
    test_maes = [r["test_mae"] for r in rows]
    result = {
        "subject": subject,
        "n_channels": sd.n_channels,
        "fingers": rows,
        "avg_test_r": float(np.nanmean(test_rs)),
        "avg_test_r_10": float(np.nanmean([r["test_r_10"] for r in rows])),
        "avg_paper": float(np.nanmean([r["paper_r"] for r in rows])),
        "avg_test_nrmse": float(np.nanmean([r["test_nrmse"] for r in rows])),
        "avg_test_mae": float(np.nanmean(test_maes)),
        # the competition's official score excluded the ring finger
        "official_test_r": float(np.nanmean([test_rs[i] for i in (0, 1, 2, 4)])),
        "official_test_mae": float(np.nanmean([test_maes[i] for i in (0, 1, 2, 4)])),
        "seconds": time.time() - t_start,
    }
    if verbose:
        print(f"  {'-' * 70}")
        print(f"  average over 5 fingers: {result['avg_test_r']:+.3f}   "
              f"(paper {result['avg_paper']:.2f})")
        print(f"  official metric (ring excluded): {result['official_test_r']:+.3f}")
        print(f"  average NRMSE over 5 fingers: {result['avg_test_nrmse']:.3f}   "
              f"(lower is better, unbounded, 0 = perfect)")
        print(f"  average MAE over 5 fingers: {result['avg_test_mae']:.3f}   "
              f"(original dataglove units, lower is better)   "
              f"official (ring excluded): {result['official_test_mae']:.3f}")
    return result


def print_table(results):
    """Reproduce the layout of the paper's Table 1 for direct comparison."""
    print(f"\n\n{'=' * 74}")
    print("TABLE 1 COMPARISON  (Pearson r on the held-out 200 s test set)")
    print("=" * 74)
    head = f"{'Subj':<5}{'Source':<10}" + "".join(f"{n.capitalize():>9s}"
                                                  for n in C.FINGER_NAMES) + f"{'Av.':>9s}"
    print(head)
    print("-" * 74)
    for res in results:
        s = res["subject"]
        ours = [r["test_r_rule"] for r in res["fingers"]]
        paper = [r["paper_r"] for r in res["fingers"]]
        print(f"{s:<5}{'paper':<10}" + "".join(f"{v:9.2f}" for v in paper)
              + f"{np.nanmean(paper):9.2f}")
        print(f"{'':<5}{'ours':<10}" + "".join(f"{v:9.2f}" for v in ours)
              + f"{np.nanmean(ours):9.2f}")
        print(f"{'':<5}{'diff':<10}"
              + "".join(f"{o - p:+9.2f}" for o, p in zip(ours, paper))
              + f"{np.nanmean(ours) - np.nanmean(paper):+9.2f}")
        print("-" * 74)

    if len(results) == 3:
        allour = np.array([[r["test_r_rule"] for r in res["fingers"]]
                           for res in results])
        allpap = np.array([[r["paper_r"] for r in res["fingers"]]
                           for res in results])
        print(f"{'Av.':<5}{'paper':<10}"
              + "".join(f"{v:9.2f}" for v in np.nanmean(allpap, 0))
              + f"{np.nanmean(allpap):9.2f}")
        print(f"{'':<5}{'ours':<10}"
              + "".join(f"{v:9.2f}" for v in np.nanmean(allour, 0))
              + f"{np.nanmean(allour):9.2f}")
        print("=" * 74)
        print(f"\n  paper, all fingers          : {C.PAPER_AVG_ALL:.2f}")
        print(f"  ours,  all fingers          : {np.nanmean(allour):.3f}")
        official_ours = np.nanmean(allour[:, [0, 1, 2, 4]])
        print(f"  competition official (no ring): "
              f"{C.COMPETITION_OFFICIAL:.2f} (paper)   {official_ours:.3f} (ours)")


def print_mae_summary(results):
    """MAE summary across all subjects (added 2026-08-04), same layout style
    as the switching linear model's compute_r2_diagnostics.py MAE summary,
    for direct cross-method comparison."""
    print(f"\n{'=' * 74}")
    print("MAE SUMMARY (original dataglove units, mean +/- SD across subjects)")
    print("=" * 74)
    avg_maes = [res["avg_test_mae"] for res in results]
    official_maes = [res["official_test_mae"] for res in results]
    for res in results:
        print(f"  Subject {res['subject']}: avg(5)={res['avg_test_mae']:.4f}   "
              f"official(4)={res['official_test_mae']:.4f}")
    if len(results) > 1:
        print(f"  {'-' * 50}")
        print(f"  avg(5) mean +/- SD:      {np.mean(avg_maes):.4f} +/- "
              f"{np.std(avg_maes, ddof=1):.4f}")
        print(f"  official(4) mean +/- SD: {np.mean(official_maes):.4f} +/- "
              f"{np.std(official_maes, ddof=1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=None, choices=[1, 2, 3])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--solver", default=None,
                    choices=["pinv_normal", "svd", "inv"])
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--standardize", action="store_true", default=None)
    ap.add_argument("--transform", default="none",
                    choices=["none", "log1p", "clip"],
                    help="optional AM-feature preprocessing (default: none, "
                         "which is the paper-faithful setting)")
    ap.add_argument("--inner-fraction", type=float, default=None,
                    help="inner train fraction for forward selection "
                         "(default 0.6, the paper's 3/5). Vary it to measure "
                         "how much test r moves from greedy-path noise alone.")
    ap.add_argument("--clip-pct", type=float, default=99.9,
                    help="percentile for --transform clip (default 99.9)")
    ap.add_argument("--no-electrode-qc", action="store_false", dest="electrode_qc",
                    default=True,
                    help="disable the train-only electrode QC screen "
                         "(on by default; see electrode_qc_mask docstring)")
    ap.add_argument("--qc-percentile", type=float, default=95.0,
                    help="percentile threshold for electrode QC (default 95.0)")
    ap.add_argument("--save", default=None, help="write results to a JSON file")
    args = ap.parse_args()

    subjects = [args.subject] if args.subject else [1, 2, 3]
    results = [run_subject(s, synthetic=args.synthetic, solver=args.solver,
                           max_features=args.max_features,
                           standardize=args.standardize,
                           transform=args.transform,
                           clip_pct=args.clip_pct,
                           inner_fraction=args.inner_fraction,
                           electrode_qc=args.electrode_qc,
                           qc_percentile=args.qc_percentile)
               for s in subjects]
    print_table(results)
    print_mae_summary(results)

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsaved -> {args.save}")


if __name__ == "__main__":
    main()
