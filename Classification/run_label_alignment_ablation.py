"""Label-alignment ablation for dense sliding-window classification.

Compares three label-alignment conventions for the dense sliding-window
pipeline used by the traditional methods (CSP+LDA, Spectral+LGB):

  onset    — window label = label at the window's first sample (the earlier,
             now-superseded convention);
  center   — window label = label at the window's midpoint;
  majority — window label = majority vote over the full window (the convention
             adopted in the main benchmark, matching DL trial extraction).

This ablation quantifies how strongly the alignment convention affects
traditional-method performance and motivates the full-window majority choice.

Usage:
    python run_label_alignment_ablation.py --subject sub1
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime

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
from traditional.models import CSPLDA, LightGBMClassifier
from traditional.evaluation import evaluate_classification


def _align_onset(labels, n):
    return labels[:n]


def _align_center(labels, n, offset):
    return labels[offset:offset + n]


def run_one(method, align_name, align_y_tr, align_y_te, feat_cfg, csp_cfg, lgb_cfg,
            train_ecog, test_ecog, tr_lab, te_lab, win_samples, step_samples, offset):
    """Run one (method, alignment) combination; return macro-F1 dict."""
    if method == 'lgb':
        X_tr, _ = extract_spectral_features(train_ecog, feat_cfg)
        X_te, _ = extract_spectral_features(test_ecog, feat_cfg)
        y_tr = align_y_tr(tr_lab, len(X_tr), win_samples, step_samples, offset)
        y_te = align_y_te(te_lab, len(X_te), win_samples, step_samples, offset)
        model = LightGBMClassifier(lgb_cfg)
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        return evaluate_classification(y_te, yp)
    else:  # csp
        csp_win = int(csp_cfg.window_ms * feat_cfg.fs / 1000)
        csp_step = int(csp_cfg.step_ms * feat_cfg.fs / 1000)
        # CSP fitting consumes per-window labels; build the full-length aligned
        # label series first (train: 10000, test: 5000 at 25 Hz).
        y_tr_full = align_y_tr(tr_lab, 10000, csp_win, csp_step, offset)
        y_te_full = align_y_te(te_lab, 5000, csp_win, csp_step, offset)
        X_tr, _, csp_f = extract_csp_features(train_ecog, y_tr_full, csp_cfg, feat_cfg)
        X_te, _, _ = extract_csp_features(test_ecog, y_te_full, csp_cfg, feat_cfg, csp_filters=csp_f)
        y_tr = y_tr_full[:len(X_tr)]
        y_te = y_te_full[:len(X_te)]
        model = CSPLDA(csp_cfg)
        model.fit(X_tr, y_tr)
        yp = model.predict(X_te)
        return evaluate_classification(y_te, yp)


def main():
    parser = argparse.ArgumentParser(description='Label-alignment ablation')
    parser.add_argument('--subject', default='sub1')
    parser.add_argument('--methods', nargs='+', default=['lgb', 'csp'])
    args = parser.parse_args()

    download_dataset()
    cfg = BenchmarkConfig()
    d = load_subject(args.subject)

    tr_lab = generate_labels(d['train_dg'])
    te_lab = generate_labels(d['test_dg'], max_angles=d['train_dg'].max(axis=0))

    feat_cfg = cfg.features
    csp_cfg = cfg.csp
    win_samples = feat_cfg.window_samples      # 500
    step_samples = feat_cfg.step_samples        # 40
    offset = win_samples // (2 * step_samples)  # 6 (250 ms centre)

    def onset(lab, n, w, s, o):
        return lab[:n]

    def center(lab, n, w, s, o):
        return lab[o:o + n]

    def majority(lab, n, w, s, o):
        return window_majority_labels(lab, w, s)[:n]

    alignments = {'onset': onset, 'center': center, 'majority': majority}

    print('=' * 70)
    print(f'  LABEL-ALIGNMENT ABLATION — {args.subject}')
    print(f'  window={win_samples} samples, step={step_samples}, centre offset={offset}')
    print('=' * 70)

    results = {}
    for align_name, align_fn in alignments.items():
        results[align_name] = {}
        for method in args.methods:
            ev = run_one(method, align_name, align_fn, align_fn,
                         feat_cfg, csp_cfg, cfg.lgb,
                         d['train_ecog'], d['test_ecog'], tr_lab, te_lab,
                         win_samples, step_samples, offset)
            results[align_name][method] = ev
            print(f'  {align_name:<10} {method:<5} F1={ev["macro_f1"]:.3f}  '
                  f'BalAcc={ev["balanced_accuracy"]:.3f}  κ={ev["cohen_kappa"]:.3f}')

    print('-' * 70)
    print(f'  {"alignment":<12}' + ''.join(f'{m:>14}' for m in args.methods))
    for align_name in alignments:
        row = f'  {align_name:<12}'
        for method in args.methods:
            row += f'{results[align_name][method]["macro_f1"]:>14.3f}'
        print(row)
    print('-' * 70)

    # Persist results for reproducibility
    os.makedirs('results/ablations', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f'results/ablations/label_alignment_{args.subject}_{ts}.json'
    payload = {
        'ablation': 'label_alignment',
        'subject': args.subject,
        'window_samples': win_samples,
        'step_samples': step_samples,
        'methods': args.methods,
        'results': results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
