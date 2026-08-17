"""
Unified Benchmark Runner — ECoG Finger Movement Decoding (Classification)
═══════════════════════════════════════════════════════════════════════════

Design: Same data source → method-specific pipelines → same evaluation function.
Three unification levels are enforced across methods:
    1. mandatory — official 400 s / 200 s split, generate_labels(),
       evaluate_classification();
    2. recommended — per-channel ECoG z-score (train-fit → test);
    3. method-specific — frequency bands, feature extraction, model architecture.

Methods (1000 Hz): CSP+LDA, Spectral+LGB  (SpectralSVM optional ablation)

Usage:
    python run_benchmark.py --subjects sub1
    python run_benchmark.py --n_runs 5 --methods lgb csp
    python run_benchmark.py --n_runs 3 --methods lgb csp ssvm   # include ablation
"""
import argparse
import dataclasses
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Dict, List

import numpy as np

warnings.filterwarnings('ignore')
SEP = '=' * 78

# ── Project root ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Traditional pipeline imports (1000 Hz)
from traditional.config import BenchmarkConfig, FINGER_NAMES
from traditional.data_loader import (
    download_dataset,
    generate_labels,
    load_subject,
    window_majority_labels,
)
from traditional.features import extract_spectral_features, extract_csp_features
from traditional.models import CSPLDA, LightGBMClassifier, SpectralSVM
from traditional.evaluation import evaluate_classification, save_results

TRAD_METHODS = ['lgb', 'csp']                   # main classification baselines
ABLATION_METHODS = ['ssvm']                     # optional: Spectral+SVM ablation
ALL_METHODS = TRAD_METHODS + ABLATION_METHODS
MAIN_METHODS = TRAD_METHODS                     # default methods for main benchmark

# Multi-run seeds: spaced by 100 for seed independence
SEEDS_TRAD = [42, 142, 242, 342, 442]   # 5-run seeds


# ══════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_data_for_subject(subject_id: str, verbose: bool = False):
    """Load and return ECoG + glove angles + labels (unified across methods)."""
    data = load_subject(subject_id)
    train_ecog = data['train_ecog']     # (n_ch, 400000) @1000Hz, z-scored
    test_ecog  = data['test_ecog']      # (n_ch, 200000) @1000Hz, z-scored
    train_dg   = data['train_dg']       # (10000, 5) @25Hz, raw angles
    test_dg    = data['test_dg']        # (5000, 5) @25Hz, raw angles
    n_ch       = data['n_channels']

    train_labels = generate_labels(train_dg)                 # train maxima (default)
    train_max = train_dg.max(axis=0)                         # per-finger train maxima
    test_labels = generate_labels(test_dg, max_angles=train_max)  # identical threshold

    if verbose:
        train_dist = get_label_distribution(train_labels)
        test_dist  = get_label_distribution(test_labels)
        print(f'  Train dist: {", ".join(f"{k}:{v[1]:.0f}%" for k,v in train_dist.items())}')
        print(f'  Test  dist: {", ".join(f"{k}:{v[1]:.0f}%" for k,v in test_dist.items())}')

    return {
        'train_ecog': train_ecog,
        'test_ecog': test_ecog,
        'train_dg': train_dg,
        'test_dg': test_dg,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'n_ch': n_ch,
    }


# ══════════════════════════════════════════════════════════════════════
# TRADITIONAL METHODS (1000 Hz)
# ══════════════════════════════════════════════════════════════════════

def run_traditional_subject(subject_id: str, seed: int, methods: List[str],
                            cfg: BenchmarkConfig, verbose: bool) -> Dict:
    """Run traditional methods for one subject with one seed."""
    feat_cfg = deepcopy(cfg.features)
    d = load_data_for_subject(subject_id, verbose=False)

    train_ecog = d['train_ecog']
    test_ecog  = d['test_ecog']
    train_dg   = d['train_dg']
    test_dg    = d['test_dg']
    train_labels = d['train_labels']
    test_labels  = d['test_labels']

    result = {'subject': subject_id, 'seed': seed, 'results': {}}

    # ── Spectral features (shared by LGB + SpectralSVM) ──
    # Labels use a full-window majority vote (identical rule to DL trial
    # extraction), replacing the earlier onset-aligned labelling.
    need_spectral = 'lgb' in methods or 'ssvm' in methods
    if need_spectral:
        X_spec_train, _ = extract_spectral_features(train_ecog, feat_cfg)
        X_spec_test, _  = extract_spectral_features(test_ecog, feat_cfg)
        maj_tr = window_majority_labels(train_labels, feat_cfg.window_samples, feat_cfg.step_samples)
        maj_te = window_majority_labels(test_labels, feat_cfg.window_samples, feat_cfg.step_samples)
        y_cls_tr = maj_tr[:len(X_spec_train)]
        y_cls_te = maj_te[:len(X_spec_test)]

    # ── CSP features ──
    if 'csp' in methods:
        csp_win = int(cfg.csp.window_ms * feat_cfg.fs / 1000)
        csp_step = int(cfg.csp.step_ms * feat_cfg.fs / 1000)
        maj_csp_tr = window_majority_labels(train_labels, csp_win, csp_step)
        maj_csp_te = window_majority_labels(test_labels, csp_win, csp_step)
        X_csp_train, _, csp_filt = extract_csp_features(
            train_ecog, maj_csp_tr, cfg.csp, feat_cfg, csp_filters=None)
        y_csp_tr = maj_csp_tr[:len(X_csp_train)]
        X_csp_test, _, _ = extract_csp_features(
            test_ecog, maj_csp_te, cfg.csp, feat_cfg, csp_filters=csp_filt)
        y_csp_te = maj_csp_te[:len(X_csp_test)]

    # ── LightGBM ──
    if 'lgb' in methods:
        lgb_cfg = dataclasses.replace(cfg.lgb, random_state=seed)
        model = LightGBMClassifier(lgb_cfg)
        model.fit(X_spec_train, y_cls_tr)
        yp = model.predict(X_spec_test)
        ypb = model.predict_proba(X_spec_test)
        m = evaluate_classification(y_cls_te, yp, y_proba=ypb)
        m['train_time'] = model.train_time
        m['infer_time'] = model.infer_time
        m['n_params'] = None  # tree-based: variable
        result['results']['lgb'] = m

    # ── CSP+LDA ──
    if 'csp' in methods:
        model = CSPLDA(cfg.csp)
        model.fit(X_csp_train, y_csp_tr)
        yp = model.predict(X_csp_test)
        ypb = model.predict_proba(X_csp_test)
        m = evaluate_classification(y_csp_te, yp, y_proba=ypb)
        m['train_time'] = model.train_time
        m['infer_time'] = model.infer_time
        m['n_params'] = X_csp_train.shape[1]  # feature count as proxy
        result['results']['csp'] = m

    # ── SpectralSVM ──
    if 'ssvm' in methods:
        svm_cfg = dataclasses.replace(cfg.svm, random_state=seed)
        model = SpectralSVM(svm_cfg)
        model.fit(X_spec_train, y_cls_tr)
        yp = model.predict(X_spec_test)
        ypb = model.predict_proba(X_spec_test)
        m = evaluate_classification(y_cls_te, yp, y_proba=ypb)
        m['train_time'] = model.train_time
        m['infer_time'] = model.infer_time
        m['n_params'] = None
        result['results']['ssvm'] = m

    return result


# ══════════════════════════════════════════════════════════════════════
# STATISTICAL SUMMARY
# ══════════════════════════════════════════════════════════════════════

def compute_statistics(all_results):
    """Compute mean±std per method×subject and Friedman+Wilcoxon tests.

    Focus on the three primary metrics: Balanced Accuracy, Macro F1, Cohen's κ.
    """
    from collections import defaultdict

    PRIMARY_METRICS = ['balanced_accuracy', 'macro_f1', 'cohen_kappa']

    # Group by (subject, method) → list of metric values across runs
    by_subj_method = defaultdict(lambda: defaultdict(list))
    # Per-finger F1 (movement fingers only): (subject, method) → {finger: [f1 across runs]}
    MOVEMENT_FINGERS = FINGER_NAMES[1:]  # Thumb..Little (exclude Rest)
    by_subj_method_finger = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        subj = r['subject']
        for mname, metrics in r['results'].items():
            for k in PRIMARY_METRICS:
                if k in metrics:
                    by_subj_method[(subj, mname)][k].append(metrics[k])
            pc = metrics.get('per_class')
            if pc:
                for finger in MOVEMENT_FINGERS:
                    f1 = (pc.get(finger) or {}).get('f1')
                    if f1 is not None:
                        by_subj_method_finger[(subj, mname)][finger].append(f1)

    subjects = sorted(set(k[0] for k in by_subj_method.keys()))
    method_names = sorted(set(k[1] for k in by_subj_method.keys()))
    n_runs = max(len(v['balanced_accuracy']) for v in by_subj_method.values())

    # Aggregate: per-subject mean (across runs) → mean ± std ACROSS SUBJECTS.
    # (A run is a repeated measurement of the same subject; the meaningful
    # variability for reporting is inter-subject, not inter-run.)
    summary = {}
    for mn in method_names:
        summary[mn] = {}
        for metric in PRIMARY_METRICS:
            subj_means = []
            for subj in subjects:
                vals = by_subj_method.get((subj, mn), {}).get(metric, [])
                subj_means.append(np.mean(vals) if vals else np.nan)
            arr = np.array(subj_means)  # (n_subjects,)
            summary[mn][metric] = {
                'mean': float(np.nanmean(arr)),
                'std': float(np.nanstd(arr)),
                'per_subject': [float(x) for x in arr],
            }

    # Friedman test: blocks = SUBJECTS (N=3), one column per method.
    # Each cell = subject's mean Macro F1 across runs. (A run is a repeated
    # measurement of the same subject, NOT an independent block.)
    friedman_p = None
    friedman_stat = None
    try:
        from scipy.stats import friedmanchisquare
        method_arrays = []   # one array per method, length = n_subjects
        for mn in method_names:
            arr = [np.mean(by_subj_method[(subj, mn)]['macro_f1'])
                   for subj in subjects
                   if by_subj_method.get((subj, mn), {}).get('macro_f1')]
            method_arrays.append(np.array(arr))
        if (len(method_arrays) >= 3
                and all(len(a) == len(subjects) and len(a) >= 3 for a in method_arrays)):
            friedman_stat, friedman_p = friedmanchisquare(*method_arrays)
    except Exception:
        pass

    # Wilcoxon signed-rank post-hoc: per-finger × subject pairing (n = 15).
    # Flattened movement-finger F1 (5 fingers × 3 subjects), averaged across runs.
    wilcoxon_pairs = {}
    try:
        from scipy.stats import wilcoxon as wilcoxon_test
        per_method_flat = {}
        for mn in method_names:
            flat = []
            for subj in subjects:
                for finger in MOVEMENT_FINGERS:
                    vals = by_subj_method_finger.get((subj, mn), {}).get(finger, [])
                    flat.append(np.mean(vals) if vals else np.nan)
            per_method_flat[mn] = np.array(flat)
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                v1 = per_method_flat.get(m1)
                v2 = per_method_flat.get(m2)
                if v1 is not None and v2 is not None and len(v1) == len(v2):
                    valid = ~(np.isnan(v1) | np.isnan(v2))
                    if valid.sum() >= 5:
                        _, p = wilcoxon_test(v1[valid], v2[valid])
                        wilcoxon_pairs[f'{m1}_vs_{m2}'] = float(p)
    except Exception:
        pass

    # Cohen's d effect sizes (pairwise) — using per-subject means across runs
    # Rationale: with 3 subjects and deterministic methods (std_runs=0),
    # cross-run pooled variance is unsuitable. Subject-level means capture
    # genuine inter-subject variability.
    cohens_d = {}
    try:
        per_subj_means = defaultdict(list)
        for (subj, mn), metrics in by_subj_method.items():
            f1_vals = metrics.get('macro_f1', [])
            if f1_vals:
                per_subj_means[mn].append(np.mean(f1_vals))

        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                v1 = np.array(per_subj_means.get(m1, []))
                v2 = np.array(per_subj_means.get(m2, []))
                if len(v1) >= 2 and len(v2) >= 2:
                    mu1, mu2 = np.mean(v1), np.mean(v2)
                    s1, s2 = np.std(v1, ddof=1), np.std(v2, ddof=1)
                    n1, n2 = len(v1), len(v2)
                    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
                    d = (mu1 - mu2) / pooled if pooled > 1e-8 else 0.0
                    cohens_d[f'{m1}_vs_{m2}'] = float(d)
    except Exception:
        pass

    stats = {
        'summary': summary,
        'friedman_p': friedman_p,
        'wilcoxon': wilcoxon_pairs,
        'cohens_d': cohens_d,
        'n_runs': n_runs,
        'n_subjects': len(subjects),
        'primary_metrics': PRIMARY_METRICS,
    }
    return stats


def print_summary(stats):
    """Pretty-print statistical summary focused on primary metrics."""
    print(f"\n{SEP}")
    print(f"  CROSS-RUN SUMMARY — Primary classification metrics")
    print(f"  (Balanced Accuracy, Macro F1, Cohen's κ; mean ± std across runs)")
    print(f"{SEP}")
    methods = sorted(stats['summary'].keys())
    header = f"{'Method':<16} {'Bal.Acc':>14} {'Macro F1':>14} {'Cohen κ':>14}"
    print(header)
    print('-' * len(header))
    for mn in methods:
        s = stats['summary'][mn]
        bac = f"{s['balanced_accuracy']['mean']:.3f}±{s['balanced_accuracy']['std']:.3f}"
        f1  = f"{s['macro_f1']['mean']:.3f}±{s['macro_f1']['std']:.3f}"
        kap = f"{s['cohen_kappa']['mean']:.3f}±{s['cohen_kappa']['std']:.3f}"
        print(f"{mn:<16} {bac:>14} {f1:>14} {kap:>14}")

    if stats['friedman_p'] is not None:
        print(f"\n  Friedman test on Macro F1: p = {stats['friedman_p']:.4f}")
        if stats['friedman_p'] < 0.05:
            print("  → Significant difference among methods (p < 0.05)")
        else:
            print("  → No significant difference among methods (p ≥ 0.05)")

    if stats['wilcoxon']:
        print(f"\n  Wilcoxon post-hoc (Macro F1):")
        for pair, p in sorted(stats['wilcoxon'].items()):
            sig = '*' if p < 0.05 else 'ns'
            print(f"    {pair:<24s} p = {p:.4f}  ({sig})")

    if stats.get('cohens_d'):
        print(f"\n  Cohen's d effect sizes (Macro F1, pairwise):")
        for pair, d in sorted(stats['cohens_d'].items()):
            mag = 'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'
            print(f"    {pair:<24s} d = {d:+.3f}  ({mag})")


# ══════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════

METHOD_LABELS = {
    'lgb': 'Spectral+LGB',
    'csp': 'CSP+LDA',
    'ssvm': 'Spectral+SVM (ablation)',
}


def _mean_std(values):
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def generate_report(all_runs, stats, config, out_path):
    """Generate publication-ready Markdown tables (Table 1 + Table 2)."""
    from collections import defaultdict

    # Group by (method, subject) across runs
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_runs:
        subj = r['subject']
        for mn, m in r['results'].items():
            for k in ['balanced_accuracy', 'macro_f1', 'cohen_kappa',
                      'train_time', 'infer_time', 'n_params']:
                if k in m and m[k] is not None:
                    grouped[(mn, subj)][k].append(m[k])

    methods = sorted(set(k[0] for k in grouped.keys()),
                     key=lambda x: list(METHOD_LABELS).index(x) if x in METHOD_LABELS else 99)
    subjects = sorted(set(k[1] for k in grouped.keys()))

    lines = []
    lines.append('# ECoG Finger Movement Classification Benchmark Report')
    lines.append('')
    lines.append(f'**Generated:** {time.strftime("%Y-%m-%d %H:%M:%S")}  ')
    lines.append(f'**Dataset:** BCI Competition IV Dataset 4 (3 subjects, 6-class finger classification)')
    lines.append('')
    lines.append('## Experimental Configuration')
    lines.append('')
    lines.append('| Setting | Value |')
    lines.append('|:---|:---|')
    lines.append(f"| Methods sampling rate | {config.get('trad_hz', 1000)} Hz |")
    lines.append(f"| Runs / seeds | {config.get('n_runs_trad', 5)} / {config.get('seeds_trad', [])} |")
    lines.append('')

    # ── Table 1: Classification performance ──
    lines.append('## Table 1: Classification Performance')
    lines.append('')
    lines.append('Primary metrics: Balanced Accuracy (BalAcc), Macro F1, Cohen\'s κ.')
    lines.append('Values shown as mean ± std across independent runs.')
    lines.append('')

    header = f"| Method | {' | '.join(f'{s.upper()} BalAcc | {s.upper()} F1 | {s.upper()} κ' for s in subjects)} | Avg BalAcc | Avg F1 | Avg κ |"
    lines.append(header)
    lines.append('|' + '|'.join(['---'] * (1 + 3 * len(subjects) + 3)) + '|')

    for mn in methods:
        label = METHOD_LABELS.get(mn, mn)
        cells = [label]
        avg_vals = defaultdict(list)
        for subj in subjects:
            g = grouped.get((mn, subj), {})
            for metric, key in [('balanced_accuracy', 'BalAcc'),
                                ('macro_f1', 'F1'),
                                ('cohen_kappa', 'κ')]:
                vals = g.get(metric, [])
                if vals:
                    mu, sd = _mean_std(vals)
                    cells.append(f'{mu:.3f}±{sd:.3f}')
                    avg_vals[metric].append(mu)
                else:
                    cells.append('—')
        for metric in ['balanced_accuracy', 'macro_f1', 'cohen_kappa']:
            if avg_vals[metric]:
                mu, sd = _mean_std(avg_vals[metric])
                cells.append(f'{mu:.3f}±{sd:.3f}')
            else:
                cells.append('—')
        lines.append('| ' + ' | '.join(cells) + ' |')

    lines.append('')
    if stats and stats.get('friedman_p') is not None:
        lines.append(f"**Friedman test** on Macro F1: $p = {stats['friedman_p']:.4f}$ "
                     f"({'significant' if stats['friedman_p'] < 0.05 else 'not significant'} at $\\alpha=0.05$).")
        lines.append('')
        if stats.get('wilcoxon'):
            lines.append('**Wilcoxon post-hoc** pairwise comparisons:')
            for pair, p in sorted(stats['wilcoxon'].items()):
                sig = 'significant' if p < 0.05 else 'not significant'
                lines.append(f"- {pair}: $p = {p:.4f}$ ({sig})")
            lines.append('')
        if stats.get('cohens_d'):
            lines.append('**Cohen\'s d effect sizes** (Macro F1, pairwise):')
            lines.append('')
            lines.append('| Comparison | d | Magnitude |')
            lines.append('|:---|---:|:---|')
            for pair, d in sorted(stats['cohens_d'].items()):
                mag = 'large' if abs(d) >= 0.8 else 'medium' if abs(d) >= 0.5 else 'small'
                lines.append(f'| {pair} | {d:+.3f} | {mag} |')
            lines.append('')

    # ── Table 2: Computational efficiency ──
    lines.append('## Table 2: Computational Efficiency')
    lines.append('')
    lines.append('| Method | #Parameters | Train time (s) | Inference time (ms) |')
    lines.append('|:---|---:|---:|---:|')

    for mn in methods:
        label = METHOD_LABELS.get(mn, mn)
        # Average across subjects and runs
        params_list, train_list, infer_list = [], [], []
        for subj in subjects:
            g = grouped.get((mn, subj), {})
            if g.get('n_params'):
                params_list.extend(g['n_params'])
            if g.get('train_time'):
                train_list.extend(g['train_time'])
            if g.get('infer_time'):
                infer_list.extend(g['infer_time'])

        npar = int(np.mean(params_list)) if params_list else '—'
        npar_str = f'{npar:,}' if isinstance(npar, int) else str(npar)

        def _fmt_time(vals):
            if not vals:
                return '—'
            mu, sd = np.mean(vals), np.std(vals)
            if mu < 0.1:
                return '<0.1'
            return f'{mu:.1f}±{sd:.1f}'

        ttrain = _fmt_time(train_list)
        tinfer = _fmt_time(infer_list)
        lines.append(f'| {label} | {npar_str} | {ttrain} | {tinfer} |')

    lines.append('')
    lines.append('---')
    lines.append('*Note: DL methods run at 500 Hz to preserve the HighG2 band (125–175 Hz) while reducing compute. '
                 'Traditional methods run at the native 1000 Hz. All methods use the same official 400 s / 200 s '
                 'temporal train/test split and the same evaluation function.*')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    import json as _json_loader
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description='Unified ECoG Decoding Benchmark')
    parser.add_argument('--subjects', nargs='+', default=['sub1', 'sub2', 'sub3'])
    parser.add_argument('--methods', nargs='+', default=MAIN_METHODS,
                        help=f'Methods to run (default: {MAIN_METHODS}); use "ssvm" for ablation')
    parser.add_argument('--n_runs', type=int, default=None,
                        help='Number of independent runs (default: 5 trad, 3 DL)')
    parser.add_argument('--skip_trad', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_dir', default='results/checkpoints',
                        help='Checkpoint directory for incremental saving')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint (skip already-completed runs)')
    parser.add_argument('--log_file', default=None,
                        help='Redirect stdout+stderr to log file (auto-flush)')
    args = parser.parse_args()

    # ── Determine methods ──
    valid_trad = set(TRAD_METHODS + ABLATION_METHODS)
    trad_methods = [m for m in args.methods if m in valid_trad] if not args.skip_trad else []

    if not trad_methods:
        print("ERROR: No methods selected. Use --methods or remove --skip_trad.")
        sys.exit(1)

    # Seeds
    n_runs_trad = args.n_runs if args.n_runs else (5 if trad_methods else 0)
    seeds_trad = SEEDS_TRAD[:n_runs_trad]

    verbose = not args.quiet

    # ── Log file ──
    if args.log_file:
        log_fh = open(args.log_file, 'a', buffering=1)  # line-buffered
        sys.stdout = log_fh
        sys.stderr = log_fh

    # ── Checkpoint path ──
    ckpt_dir = _Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'checkpoint.json'

    # ── Resume ──
    completed_set = set()  # (subject, seed, method)
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt_data = _json_loader.load(f)
        for r in ckpt_data.get('completed', []):
            completed_set.add(tuple(r))
        print(f'[RESUME] Found {len(completed_set)} completed runs. Skipping...')
        sys.stdout.flush()

    download_dataset()

    print(f'\n{SEP}')
    print(f'  UNIFIED ECoG DECODING BENCHMARK')
    print(f'  Methods (1000Hz): {trad_methods or ["—"]} x {n_runs_trad} runs')
    print(f'  Subjects: {args.subjects}')
    print(f'  Checkpoint: {ckpt_path}  |  Resume: {args.resume}')
    print(f'{SEP}')

    def _save_ckpt(all_runs_list):
        """Atomically save checkpoint with Windows-safe retry."""
        completed = []
        for r in all_runs_list:
            subj = r['subject']
            seed = r['seed']
            for mn in r.get('results', {}).keys():
                completed.append((subj, seed, mn))
        tmp_path = str(ckpt_path) + '.tmp'
        for attempt in range(3):
            try:
                with open(tmp_path, 'w') as f:
                    _json_loader.dump({'completed': completed}, f)
                os.replace(tmp_path, str(ckpt_path))
                return
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
        # Final fallback: direct write without tmp
        try:
            with open(str(ckpt_path), 'w') as f:
                _json_loader.dump({'completed': completed}, f)
        except Exception:
            pass  # non-critical; results are saved independently

    t_total = time.time()
    all_runs = []

    # ── Traditional ──
    if trad_methods:
        cfg = BenchmarkConfig()
        for seed in seeds_trad:
            for subj in args.subjects:
                # Check if all trad methods for this (subj, seed) are done
                remaining = [mn for mn in trad_methods
                             if (subj, seed, mn) not in completed_set]
                if not remaining:
                    if verbose:
                        print(f'  [SKIP] {subj} trad seed={seed} — already completed')
                    continue
                if verbose:
                    print(f'\n── Trad methods | {subj} seed={seed} ──')
                try:
                    r = run_traditional_subject(subj, seed, trad_methods, cfg, verbose)
                    all_runs.append(r)
                    _save_ckpt(all_runs)
                    if verbose:
                        for mn in trad_methods:
                            m = r['results'].get(mn, {})
                            if m:
                                print(f'  {subj}/{mn}: BalAcc={m["balanced_accuracy"]:.3f}  '
                                      f'F1={m["macro_f1"]:.3f}  κ={m["cohen_kappa"]:.3f}')
                    sys.stdout.flush()
                except Exception as e:
                    print(f'  [ERROR] {subj} trad seed={seed}: {e}')
                    import traceback; traceback.print_exc()
                    sys.stdout.flush()

    t_total = time.time() - t_total

    # ── Summary ──
    if len(all_runs) > 1:
        stats = compute_statistics(all_runs)
        print_summary(stats)
    else:
        stats = None

    # ── Save ──
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    methods_str = '_'.join(sorted(set(m for r in all_runs for m in r['results'].keys()))[:4])
    filename = f'benchmark_{methods_str}_{timestamp}.json'
    config_dict = {
        'trad_hz': 1000,
        'n_runs_trad': n_runs_trad,
        'seeds_trad': seeds_trad,
    }
    filepath = save_results({
        'config': config_dict,
        'results': all_runs,
        'statistics': stats,
        'runtime_s': t_total,
    }, filename)
    print(f'\n  Results saved → {filepath}')

    # Generate Markdown report
    report_path = filepath.replace('.json', '.md')
    try:
        generate_report(all_runs, stats, config_dict, report_path)
        print(f'  Report saved  → {report_path}')
    except Exception as e:
        print(f'  [WARN] Could not generate Markdown report: {e}')

    print(f'  Total runtime: {t_total/60:.1f} min')


if __name__ == '__main__':
    main()
