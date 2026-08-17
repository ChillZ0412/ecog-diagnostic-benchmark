"""
Ablation experiments for ECoG finger movement classification.
═══════════════════════════════════════════════════════════════════════════

Three controlled experiments, each varying ONE factor while holding all others
at their main-benchmark configuration:

  CSP frequency band:  (8-200) / (30-200) / (60-200) / (65-175) Hz
  Window length:       250ms / 500ms / 1000ms
  Classifier choice:   Spectral+LGB  vs  Spectral+SVM

(DL-side ablations (Mixup augmentation and SimCLR pretraining) are
implemented in run_dl_ablations.py. The sliding-vs-trial segmentation comparison
is reported in the manuscript as a method-selection rationale rather than a
standalone ablation, since the dense sliding-window DL reference
was superseded by the trial-based pipeline used throughout.)

Usage:
    python run_ablations.py --ablation csp_band
    python run_ablations.py --ablation window
    python run_ablations.py --ablation classifier
    python run_ablations.py --ablation all         # run all traditional-method ablations
"""
import dataclasses
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traditional.config import BenchmarkConfig
from traditional.data_loader import (
    download_dataset,
    generate_labels,
    load_subject,
    window_majority_labels,
)
from traditional.features import extract_spectral_features, extract_csp_features
from traditional.models import CSPLDA, LightGBMClassifier, SpectralSVM
from traditional.evaluation import evaluate_classification, save_results

SEP = '─' * 70
SUBJECTS = ['sub1', 'sub2', 'sub3']

# ── CSP frequency band candidates ──
CSP_BANDS = {
    'allgamma':  (8,   200),   # All gamma + alpha/beta
    'widegamma': (30,  200),   # Gamma only
    'highgamma': (60,  200),   # High gamma
    'optimal':   (65,  175),   # Our grid-search optimum
}

# ── Window length candidates ──
WINDOW_LENGTHS_MS = [250, 500, 1000]  # ms


# ══════════════════════════════════════════════════════════════════════
# RUNNER HELPERS
# ══════════════════════════════════════════════════════════════════════

def _run_one_trial(feat_cfg, csp_cfg, lgb_cfg, svm_cfg, subject_id, methods):
    """Run one trial for a given config set. Returns {method: metrics, ...}."""
    d = load_subject(subject_id)
    ecog_t = d['train_ecog']
    ecog_v = d['test_ecog']
    dg_t = d['train_dg']
    dg_v = d['test_dg']
    labels_t = generate_labels(dg_t)                              # train maxima (default)
    labels_v = generate_labels(dg_v, max_angles=dg_t.max(axis=0))  # shared threshold

    results = {}

    # Spectral features (shared) — full-window majority-vote labels
    need_spec = 'lgb' in methods or 'ssvm' in methods
    if need_spec:
        Xs_tr, _ = extract_spectral_features(ecog_t, feat_cfg)
        Xs_te, _ = extract_spectral_features(ecog_v, feat_cfg)
        maj_t = window_majority_labels(labels_t, feat_cfg.window_samples, feat_cfg.step_samples)
        maj_v = window_majority_labels(labels_v, feat_cfg.window_samples, feat_cfg.step_samples)
        yc_tr = maj_t[:len(Xs_tr)]
        yc_te = maj_v[:len(Xs_te)]

    # CSP features — full-window majority-vote labels
    if 'csp' in methods:
        csp_win = int(csp_cfg.window_ms * feat_cfg.fs / 1000)
        csp_step = int(csp_cfg.step_ms * feat_cfg.fs / 1000)
        maj_csp_t = window_majority_labels(labels_t, csp_win, csp_step)
        maj_csp_v = window_majority_labels(labels_v, csp_win, csp_step)
        Xc_tr, _, csp_f = extract_csp_features(ecog_t, maj_csp_t, csp_cfg, feat_cfg, csp_filters=None)
        ycsp_tr = maj_csp_t[:len(Xc_tr)]
        Xc_te, _, _ = extract_csp_features(ecog_v, maj_csp_v, csp_cfg, feat_cfg, csp_filters=csp_f)
        ycsp_te = maj_csp_v[:len(Xc_te)]

    # LGB
    if 'lgb' in methods:
        m = LightGBMClassifier(lgb_cfg)
        m.fit(Xs_tr, yc_tr)
        yp = m.predict(Xs_te)
        ypb = m.predict_proba(Xs_te)
        ev = evaluate_classification(yc_te, yp, y_proba=ypb)
        ev['train_time'] = m.train_time
        results['lgb'] = ev

    # CSP+LDA
    if 'csp' in methods:
        m = CSPLDA(csp_cfg)
        m.fit(Xc_tr, ycsp_tr)
        yp = m.predict(Xc_te)
        ypb = m.predict_proba(Xc_te)
        ev = evaluate_classification(ycsp_te, yp, y_proba=ypb)
        ev['train_time'] = m.train_time
        results['csp'] = ev

    # Spectral+SVM
    if 'ssvm' in methods:
        m = SpectralSVM(svm_cfg)
        m.fit(Xs_tr, yc_tr)
        yp = m.predict(Xs_te)
        ypb = m.predict_proba(Xs_te)
        ev = evaluate_classification(yc_te, yp, y_proba=ypb)
        ev['train_time'] = m.train_time
        results['ssvm'] = ev

    return results


def _print_ablation_table(
    title: str,
    variants: List[str],
    subject_results: Dict[str, Dict[str, Dict]],
    metric: str = 'macro_f1',
    fmt: str = '.3f',
):
    """Print a clean ablation comparison table."""
    print(f'\n{SEP}')
    print(f'  {title}')
    print(f'  Metric: {metric}')
    print(f'{SEP}')

    header = f"{'Variant':<14}"
    for s in SUBJECTS:
        header += f' {s:>10}'
    header += f' {"Mean":>10}'
    print(header)
    print('─' * len(header))

    best_val = -1
    best_var = ''

    for var in variants:
        vals = []
        row = f'{var:<14}'
        for s in SUBJECTS:
            v = subject_results.get(var, {}).get(s, {})
            val = v.get(metric, 0)
            vals.append(val)
            row += f' {val:{fmt}}'
        mean_val = np.mean(vals)
        row += f' {mean_val:{fmt}}'
        print(row)
        if mean_val > best_val:
            best_val = mean_val
            best_var = var

    print('─' * len(header))
    print(f'  Best: {best_var}  ({metric}={best_val:{fmt}})')


# ══════════════════════════════════════════════════════════════════════
# ABLATION RUNNERS
# ══════════════════════════════════════════════════════════════════════

def run_ablation_csp_band():
    """Vary the CSP frequency band, keeping n_components=3 and LDA fixed."""
    print(f'\n{"="*70}')
    print('  CSP FREQUENCY BAND ABLATION')
    print(f'{"="*70}')

    base_cfg = BenchmarkConfig()
    results_by_band = {}

    for band_name, band in CSP_BANDS.items():
        print(f'\n  Band: {band_name} {band} Hz')
        csp_cfg = dataclasses.replace(base_cfg.csp, freq_band=band, n_components=3)
        results_by_band[band_name] = {}

        for subj in SUBJECTS:
            t0 = time.time()
            r = _run_one_trial(
                base_cfg.features, csp_cfg,
                base_cfg.lgb, base_cfg.svm,
                subj, methods=['csp'],
            )
            elapsed = time.time() - t0
            csp_r = r['csp']
            results_by_band[band_name][subj] = csp_r
            print(f'    {subj}: F1={csp_r["macro_f1"]:.3f}  '
                  f'BalAcc={csp_r["balanced_accuracy"]:.3f}  κ={csp_r["cohen_kappa"]:.3f}  '
                  f'({elapsed:.1f}s)')

    _print_ablation_table('CSP Frequency Band', list(CSP_BANDS.keys()), results_by_band, 'macro_f1')

    return {
        'ablation': 'csp_band',
        'variants': {k: v for k, v in CSP_BANDS.items()},
        'results': results_by_band,
    }


def run_ablation_window():
    """Vary the window length, testing CSP+LDA and Spectral+LGB."""
    print(f'\n{"="*70}')
    print('  WINDOW LENGTH ABLATION')
    print(f'{"="*70}')

    base_cfg = BenchmarkConfig()
    methods = ['lgb', 'csp']
    all_results = {}

    for win_ms in WINDOW_LENGTHS_MS:
        variant_name = f'{win_ms}ms'
        print(f'\n  Window: {win_ms} ms')
        all_results[variant_name] = {}

        feat_cfg = dataclasses.replace(
            base_cfg.features,
            window_length_ms=win_ms,
            step_ms=40,
        )
        csp_cfg = dataclasses.replace(
            base_cfg.csp,
            window_ms=win_ms,
            step_ms=40,
        )

        for subj in SUBJECTS:
            t0 = time.time()
            r = _run_one_trial(
                feat_cfg, csp_cfg,
                base_cfg.lgb, base_cfg.svm,
                subj, methods=methods,
            )
            elapsed = time.time() - t0
            all_results[variant_name][subj] = {}
            for mn in methods:
                if mn in r:
                    all_results[variant_name][subj][mn] = r[mn]
                    m = r[mn]
                    print(f'    {subj}/{mn}: F1={m["macro_f1"]:.3f}  '
                          f'BalAcc={m["balanced_accuracy"]:.3f}  ({elapsed:.1f}s)')

    # Per-method tables
    for mn in methods:
        per_method = {}
        for var in all_results:
            per_method[var] = {s: all_results[var][s].get(mn, {}) for s in SUBJECTS}
        _print_ablation_table(
            f'Window Length — {mn}',
            [f'{w}ms' for w in WINDOW_LENGTHS_MS],
            per_method, 'macro_f1',
        )

    return {
        'ablation': 'window_length',
        'variants': [{'window_ms': w} for w in WINDOW_LENGTHS_MS],
        'results': all_results,
    }


def run_ablation_classifier():
    """Compare LightGBM vs RBF-SVM on identical spectral features."""
    print(f'\n{"="*70}')
    print('  CLASSIFIER CHOICE ABLATION')
    print('  (Same 6-band spectral features; LightGBM vs RBF-SVM)')
    print(f'{"="*70}')

    base_cfg = BenchmarkConfig()
    results_by_clf = {'LGB': {}, 'SVM': {}}

    for subj in SUBJECTS:
        t0 = time.time()
        r = _run_one_trial(
            base_cfg.features, base_cfg.csp,
            base_cfg.lgb, base_cfg.svm,
            subj, methods=['lgb', 'ssvm'],
        )
        elapsed = time.time() - t0
        for mn in ['lgb', 'ssvm']:
            if mn in r:
                label = 'LGB' if mn == 'lgb' else 'SVM'
                results_by_clf[label][subj] = r[mn]
                m = r[mn]
                print(f'  {subj}/{label}: F1={m["macro_f1"]:.3f}  '
                      f'BalAcc={m["balanced_accuracy"]:.3f}  Train={m["train_time"]:.1f}s  '
                      f'({elapsed:.1f}s)')

    _print_ablation_table('Classifier Choice', ['LGB', 'SVM'], results_by_clf, 'macro_f1')

    return {
        'ablation': 'classifier_choice',
        'variants': ['LGB', 'SVM'],
        'results': results_by_clf,
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ablation experiments')
    parser.add_argument('--ablation', default='all',
                        choices=['all', 'classifier', 'csp_band', 'window'])
    parser.add_argument('--output_dir', default='results/ablations')
    args = parser.parse_args()

    download_dataset()
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    t_start = time.time()

    if args.ablation in ('all', 'classifier'):
        results['classifier'] = run_ablation_classifier()

    if args.ablation in ('all', 'csp_band'):
        results['csp_band'] = run_ablation_csp_band()

    if args.ablation in ('all', 'window'):
        results['window'] = run_ablation_window()

    t_total = time.time() - t_start

    # Save
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = save_results(
        {'ablations': results, 'runtime_s': t_total},
        f'ablation_{args.ablation}_{ts}.json',
        results_dir=args.output_dir,
    )
    print(f'\n{"="*70}')
    print(f'  All ablations complete.  Total: {t_total/60:.1f} min')
    print(f'  Saved → {path}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
