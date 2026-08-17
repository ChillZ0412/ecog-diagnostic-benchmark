"""
Diagnose a single subject/finger, and run the ablations the earlier stages set up.

Run:
    python diagnose.py --subject 3 --finger ring
    python diagnose.py --subject 2 --finger index

Motivation: subject 3's ring finger reproduces at test r ~ 0.00 despite a
validation r of ~0.68, while every other finger tracks the paper closely.
A val->test gap of that size is not ordinary selection overfitting (which
costs ~0.05-0.15 elsewhere in this reproduction); it points at something
structural. The two candidate explanations this script separates are:

  (a) OUTLIER DOMINATION. AM features here are extremely heavy-tailed
      (median ~1e6, max ~1e13). If a selected channel carries a large
      transient in the test segment, a handful of samples can dominate the
      prediction and drive Pearson r to zero even when the ranking is fine.
      Signature: Spearman r stays high while Pearson r collapses, and
      trimming a few extreme predictions restores the correlation.

  (b) DISTRIBUTION SHIFT. A selected channel simply behaves differently in
      the test segment than in training (electrode drift, artifact, a seizure
      episode). Signature: the feature's train and test distributions differ
      by orders of magnitude, and dropping that feature repairs the fit.

These are distinguishable, and the distinction matters for the report: (a) is
a preprocessing issue with a principled fix, (b) is a property of the data
that the original paper would also have faced.
"""
import argparse

import numpy as np

import config as C
from data_io import load_subject
from features import extract_am_features, downsample_target, align
from memory_stack import build_xy
from regressor import fit_wiener, pearson_r
from selection import forward_select


def spearman(a, b):
    """Rank correlation — insensitive to monotone outliers."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return pearson_r(ra, rb)


def trimmed_r(pred, target, frac=0.01):
    """Pearson r after removing the most extreme |prediction| samples."""
    n_drop = max(int(len(pred) * frac), 1)
    keep = np.argsort(np.abs(pred - np.median(pred)))[:-n_drop]
    return pearson_r(pred[keep], target[keep])


def evaluate(Xtr, ytr, Xte, yte, f, transform=None, label="", **fit_kw):
    """Full select -> refit -> test for one finger, optionally on transformed features."""
    A, B = (Xtr, Xte) if transform is None else (transform(Xtr), transform(Xte))
    sel = forward_select(A, ytr[:, f])
    cols = sel.selected
    Dtr, dtr = build_xy(A, ytr[:, f:f + 1], columns=cols)
    fit = fit_wiener(Dtr, dtr, **fit_kw)
    Dte, dte = build_xy(B, yte[:, f:f + 1], columns=cols)
    pred = fit.predict(Dte)
    return {
        "label": label,
        "n": len(cols),
        "val_r": sel.val_r[sel.n_selected - 1],
        "test_r": pearson_r(pred, dte),
        "spearman": spearman(pred, dte.ravel()),
        "trimmed_r": trimmed_r(pred, dte.ravel()),
        "cols": cols,
        "pred": pred,
        "target": dte.ravel(),
        "weight_norm": fit.diagnostics["weight_norm"],
    }


def target_distribution_shift(train_glove: np.ndarray, test_glove: np.ndarray,
                              finger_idx: int, rest_threshold_pct: float = 10.0) -> dict:
    """
    Check whether the FINGER'S OWN MOVEMENT PATTERN differs between train and
    test segments -- the regression-side analogue of a classification
    "label distribution shift" (e.g. rest-vs-movement time fraction changing
    between train and test).

    This is a different mechanism from the channel-artifact check above:
    that one asks whether an ECoG FEATURE looks different in test; this one
    asks whether the TARGET BEHAVIOR itself looks different. A subject can be
    clean on one axis and shifted on the other -- they must be checked
    separately.

    'Rest' here is defined as |flexion - training baseline| below
    rest_threshold_pct% of that finger's training range, matching the spirit
    of a common threshold-based rest/movement label rule.
    """
    tr = train_glove[:, finger_idx]
    te = test_glove[:, finger_idx]

    baseline = np.median(tr)
    rng = tr.max() - tr.min()
    thr = rng * rest_threshold_pct / 100.0

    tr_rest_frac = float((np.abs(tr - baseline) < thr).mean())
    te_rest_frac = float((np.abs(te - baseline) < thr).mean())

    return {
        "train_mean": float(tr.mean()), "test_mean": float(te.mean()),
        "train_std": float(tr.std()), "test_std": float(te.std()),
        "train_range": (float(tr.min()), float(tr.max())),
        "test_range": (float(te.min()), float(te.max())),
        "train_rest_frac": tr_rest_frac, "test_rest_frac": te_rest_frac,
        "rest_frac_shift": te_rest_frac - tr_rest_frac,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--finger", default="ring", choices=C.FINGER_NAMES)
    args = ap.parse_args()
    f = C.FINGER_NAMES.index(args.finger)

    sd = load_subject(args.subject)
    Xtr, names = extract_am_features(sd.train_ecog)
    ytr = downsample_target(sd.train_glove)
    Xtr, ytr = align(Xtr, ytr)
    Xte, _ = extract_am_features(sd.test_ecog)
    yte = downsample_target(sd.test_glove)
    Xte, yte = align(Xte, yte)

    print(f"\n{'=' * 70}")
    print(f"Subject {args.subject}, {args.finger} finger")
    print("=" * 70)

    print(f"\n[0] Target (behavioral) distribution shift — train vs test")
    tshift = target_distribution_shift(ytr, yte, f)
    print(f"      train mean={tshift['train_mean']:.3f}  std={tshift['train_std']:.3f}  "
          f"range={tshift['train_range']}")
    print(f"      test  mean={tshift['test_mean']:.3f}  std={tshift['test_std']:.3f}  "
          f"range={tshift['test_range']}")
    print(f"      'rest' time fraction (within 10% of train baseline):")
    print(f"        train {tshift['train_rest_frac']:.1%}  ->  test {tshift['test_rest_frac']:.1%}"
          f"   (shift {tshift['rest_frac_shift']:+.1%})")
    if abs(tshift["rest_frac_shift"]) > 0.15:
        print("      -> LARGE behavioral shift: this finger moves in a genuinely different")
        print("         pattern during test than during training, independent of any ECoG")
        print("         channel artifact. This affects EVERY method trained on this subject,")
        print("         not just this one — worth flagging to the classification teammate too.")

    base = evaluate(Xtr, ytr, Xte, yte, f, label="baseline")
    print(f"\n[1] Baseline")
    print(f"      features selected : {base['n']}")
    print(f"      validation r      : {base['val_r']:+.4f}")
    print(f"      TEST r (Pearson)  : {base['test_r']:+.4f}")
    print(f"      TEST r (Spearman) : {base['spearman']:+.4f}")
    print(f"      TEST r (1% trimmed): {base['trimmed_r']:+.4f}")
    print(f"      ||w||             : {base['weight_norm']:.3e}")

    print(f"\n[2] Is the prediction outlier-dominated?")
    pred, tgt = base["pred"], base["target"]
    dev = np.abs(pred - pred.mean())
    order = np.argsort(dev)[::-1]
    top1 = max(int(len(pred) * 0.01), 1)
    share = (dev[order[:top1]] ** 2).sum() / (dev ** 2).sum()
    print(f"      prediction range  : [{pred.min():.3g}, {pred.max():.3g}]")
    print(f"      target range      : [{tgt.min():.3g}, {tgt.max():.3g}]")
    print(f"      top 1% of samples carry {share:.1%} of prediction variance")
    if share > 0.5:
        print("      -> OUTLIER DOMINATED. A handful of samples control the fit.")
    elif base["spearman"] - base["test_r"] > 0.2:
        print("      -> Spearman >> Pearson: ranking is fine, scale is not.")
    else:
        print("      -> not obviously outlier driven.")

    print(f"\n[3] Train vs test distribution of the SELECTED features")
    print(f"      {'feature':<16}{'train med':>12}{'test med':>12}"
          f"{'train max':>12}{'test max':>12}{'shift':>9}")
    shifts = []
    for c in base["cols"]:
        a, b = Xtr[:, c], Xte[:, c]
        shift = np.median(b) / max(np.median(a), 1e-30)
        shifts.append((abs(np.log10(max(shift, 1e-30))), c, shift))
        band, ch = names[c]
        print(f"      {band + '/ch' + str(ch):<16}{np.median(a):12.3g}"
              f"{np.median(b):12.3g}{a.max():12.3g}{b.max():12.3g}{shift:9.2f}x")
    worst = max(shifts)
    if worst[0] > np.log10(3):
        band, ch = names[worst[1]]
        print(f"      -> largest median shift: {band}/ch{ch} at {worst[2]:.2f}x")

    print(f"\n[4] Drop-one-feature: which selected feature is responsible?")
    for c in base["cols"]:
        cols = [x for x in base["cols"] if x != c]
        if not cols:
            continue
        Dtr, dtr = build_xy(Xtr, ytr[:, f:f + 1], columns=cols)
        fit = fit_wiener(Dtr, dtr)
        Dte, dte = build_xy(Xte, yte[:, f:f + 1], columns=cols)
        r = fit.score(Dte, dte)
        band, ch = names[c]
        flag = "   <-- removing this helps a lot" if r - base["test_r"] > 0.15 else ""
        print(f"      without {band + '/ch' + str(ch):<16} test r = {r:+.4f}"
              f"  ({r - base['test_r']:+.4f}){flag}")

    print(f"\n[5] Ablations")
    variants = [
        ("log1p features", lambda X: np.log1p(np.maximum(X, 0)), {}),
        ("standardised fit", None, {"standardize": True}),
        ("rcond = 1e-8", None, {"rcond": 1e-8}),
        ("rcond = 1e-6", None, {"rcond": 1e-6}),
        ("solver = svd", None, {"solver": "svd"}),
    ]
    print(f"      {'variant':<20}{'n':>4}{'val r':>9}{'TEST r':>9}{'vs base':>10}")
    print(f"      {'baseline':<20}{base['n']:4d}{base['val_r']:9.4f}"
          f"{base['test_r']:9.4f}{0.0:10.4f}")
    for label, tf, kw in variants:
        try:
            v = evaluate(Xtr, ytr, Xte, yte, f, transform=tf, label=label, **kw)
            print(f"      {label:<20}{v['n']:4d}{v['val_r']:9.4f}"
                  f"{v['test_r']:9.4f}{v['test_r'] - base['test_r']:+10.4f}")
        except Exception as e:
            print(f"      {label:<20} failed: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
