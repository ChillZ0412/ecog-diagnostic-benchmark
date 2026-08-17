"""
Mixup and SSL ablation runners for trial-based DL methods.

    Mixup augmentation:  trial-based DL with vs without Mixup (alpha = 0.4 vs 0).
    SSL pretraining:     SimCLR pretrain + fine-tune vs supervised-only baseline.

Both ablations hold every other factor at the main-benchmark configuration, so
the reported difference is attributable solely to the varied factor. Results are
printed as a per-subject comparison table and saved under ``results/ablations/``.

Usage:
    python run_dl_ablations.py --ablation mixup
    python run_dl_ablations.py --ablation ssl
    python run_dl_ablations.py --ablation all      # run Mixup + SSL
"""
import argparse
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traditional.data_loader import download_dataset
from traditional.evaluation import save_results
from run_trial_dl import run_subject_trial, SUBJECTS, SEEDS
from run_ssl_dl import run_subject_ssl

METRIC = 'macro_f1'
METHODS = ['eegnet', 'eegconformer']


def _aggregate(runs, metric=METRIC):
    """Aggregate per-subject×method metrics across runs → {(subj, method): (mean, std)}."""
    per_key = defaultdict(list)
    for r in runs:
        for mk, m in r['results'].items():
            if metric in m:
                per_key[(r['subject'], mk)].append(m[metric])
    return {
        key: (float(np.mean(vals)), float(np.std(vals)))
        for key, vals in sorted(per_key.items())
    }


def _print_comparison(title, variant_results, metric=METRIC):
    """Print a with/without comparison table (per subject × method, then mean)."""
    sep = '=' * 88
    print(f'\n{sep}\n  {title}\n{sep}')

    names = list(variant_results.keys())
    agg = {name: _aggregate(runs, metric) for name, runs in variant_results.items()}
    keys = sorted(agg[names[0]].keys())

    header = (f"{'Subject':<8}{'Method':<13}"
              + ''.join(f' {name.replace("_", " "):>22}' for name in names)
              + f' {"Δ (with−without)":>18}')
    print(header)
    print('-' * len(header))

    for (subj, mk) in keys:
        row = f'{subj:<8}{mk:<13}'
        vals = []
        for name in names:
            mean, std = agg[name].get((subj, mk), (np.nan, np.nan))
            row += f' {mean:>10.3f}±{std:.3f}'
            vals.append(mean)
        row += f' {vals[0] - vals[-1]:>+16.3f}'
        print(row)

    print('-' * len(header))
    for mk in METHODS:
        row = f'{"Mean":<8}{mk:<13}'
        means = []
        for name in names:
            subj_means = [agg[name][(s, mk)][0] for s in SUBJECTS if (s, mk) in agg[name]]
            mean = float(np.mean(subj_means))
            row += f' {mean:>10.3f}     '
            means.append(mean)
        row += f' {means[0] - means[-1]:>+16.3f}'
        print(row)


def run_ablation_mixup(subjects, seeds, n_epochs, patience):
    """Trial-based DL with Mixup (α=0.4) vs without (α=0)."""
    results = {'with_mixup': [], 'without_mixup': []}
    for subj in subjects:
        for seed in seeds:
            results['with_mixup'].append(
                run_subject_trial(subj, seed, n_epochs=n_epochs,
                                  patience=patience, mixup_alpha=0.4))
            results['without_mixup'].append(
                run_subject_trial(subj, seed, n_epochs=n_epochs,
                                  patience=patience, mixup_alpha=0.0))
    return results


def run_ablation_ssl(subjects, seeds, n_epochs_ssl, n_epochs_sup, patience):
    """SimCLR pretraining vs a supervised-only baseline (same fine-tune budget)."""
    results = {'with_ssl': [], 'without_ssl': []}
    for subj in subjects:
        for seed in seeds:
            results['with_ssl'].append(
                run_subject_ssl(subj, seed, n_epochs_ssl=n_epochs_ssl,
                                n_epochs_sup=n_epochs_sup, patience=patience))
            results['without_ssl'].append(
                run_subject_trial(subj, seed, n_epochs=n_epochs_sup,
                                  patience=patience, mixup_alpha=0.0))
    return results


def main():
    parser = argparse.ArgumentParser(description='Mixup and SSL ablation runners')
    parser.add_argument('--ablation', default='all', choices=['all', 'mixup', 'ssl'])
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Supervised fine-tune epochs (Mixup: both arms; SSL: supervised arm).')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--ssl_epochs', type=int, default=50,
                        help='SimCLR pretraining epochs (SSL with-pretraining arm).')
    parser.add_argument('--output_dir', default='results/ablations')
    args = parser.parse_args()

    download_dataset()
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    t_start = time.time()

    if args.ablation in ('all', 'mixup'):
        print('\n' + '=' * 88)
        print('  MIXUP AUGMENTATION ABLATION')
        print('=' * 88)
        r = run_ablation_mixup(args.subjects, args.seeds, args.epochs, args.patience)
        _print_comparison('Mixup — with (α=0.4) vs without (α=0)', r)
        results['mixup'] = r

    if args.ablation in ('all', 'ssl'):
        print('\n' + '=' * 88)
        print('  SSL PRETRAINING ABLATION')
        print('=' * 88)
        r = run_ablation_ssl(args.subjects, args.seeds, args.ssl_epochs, args.epochs, args.patience)
        _print_comparison('SSL — SimCLR pretrain vs supervised-only', r)
        results['ssl'] = r

    ts = time.strftime('%Y%m%d_%H%M%S')
    path = save_results(
        {'ablations': results, 'runtime_s': time.time() - t_start},
        f'dl_ablation_{args.ablation}_{ts}.json',
        results_dir=args.output_dir,
    )
    print(f'\nSaved → {path}')


if __name__ == '__main__':
    main()
