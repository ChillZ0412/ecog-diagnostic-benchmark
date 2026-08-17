"""
preprocess_raw.py — end-to-end data preparation for the ECoG benchmark.

Usage
-----
    python preprocess_raw.py

This script downloads BCI Competition IV Dataset 4 (if not already cached),
applies z-score normalization per channel (training statistics only), and
exports both sliding-window (traditional methods) and trial-based (DL methods)
representations for all three subjects.

Output
------
    results/preprocessed/<subject>/train_ecog.npy  — (n_ch, 400000) @ 1000 Hz
    results/preprocessed/<subject>/test_ecog.npy   — (n_ch, 200000) @ 1000 Hz
    results/preprocessed/<subject>/train_dg.npy    — (5, 10000)   @ 25 Hz
    results/preprocessed/<subject>/test_dg.npy     — (5, 5000)    @ 25 Hz
    results/preprocessed/<subject>/train_labels.npy — (n_windows,) sliding-window
    results/preprocessed/<subject>/test_labels.npy  — (n_trials,)  trial-based
    results/preprocessed/<subject>/train_trials.npy — (n_trials, n_ch, 500)
    results/preprocessed/<subject>/test_trials.npy  — (n_trials, n_ch, 500)

Requirements
------------
    pip install numpy scipy scikit-learn

Reference
---------
    Miller, K. J. & Schalk, G. (2008). BCI Competition IV, Dataset 4.
"""
import os, sys, argparse, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traditional.config import SUBJECT_CHANNELS
from traditional.data_loader import download_dataset, load_subject, generate_labels
from traditional.dl_utils import downsample_ecog, extract_trials, extract_test_trials


def save_if_not_exists(path: str, arr: np.ndarray) -> bool:
    """Save numpy array, skip if already exists. Returns True if saved."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return False
    np.save(path, arr.astype(np.float32))
    return True


def process_subject(subject_id: str, output_root: str = "results/preprocessed") -> dict:
    """Extract and save all preprocessed representations for one subject.

    Parameters
    ----------
    subject_id : str
        One of 'sub1', 'sub2', 'sub3'.
    output_root : str
        Base directory for saved .npy files.

    Returns
    -------
    dict with keys: n_channels, n_train_windows, n_test_windows, n_train_trials,
    n_test_trials, n_classes, files_saved, files_skipped.
    """
    subject_dir = os.path.join(output_root, subject_id)
    d = load_subject(subject_id)
    n_ch = d['train_ecog'].shape[0]
    stats = {'n_channels': n_ch}

    # ── 1. ECoG (z-scored) + data glove (raw angles), prepared by load_subject ──
    saved = 0; skipped = 0
    for key, arr in [('train_ecog', d['train_ecog']),
                      ('test_ecog', d['test_ecog']),
                      ('train_dg', d['train_dg']),
                      ('test_dg', d['test_dg'])]:
        path = os.path.join(subject_dir, key + '.npy')
        if save_if_not_exists(path, arr):
            saved += 1
        else:
            skipped += 1

    # ── 2. Sliding-window labels (for traditional methods) ──
    train_max = d['train_dg'].max(axis=0)   # per-finger train maxima (shared threshold)
    tr_labels = generate_labels(d['train_dg'])
    te_labels = generate_labels(d['test_dg'], max_angles=train_max)
    stats['n_train_windows'] = len(tr_labels)
    stats['n_test_windows'] = len(te_labels)
    stats['n_classes'] = len(np.unique(tr_labels))
    for key, arr in [('train_labels', tr_labels),
                      ('test_labels', te_labels)]:
        path = os.path.join(subject_dir, key + '.npy')
        if save_if_not_exists(path, arr):
            saved += 1
        else:
            skipped += 1

    # ── 3. Trial-based ECoG (for DL methods) ──
    ecog_train_500 = downsample_ecog(d['train_ecog'])
    ecog_test_500 = downsample_ecog(d['test_ecog'])
    tr_trials, tr_trial_labels = extract_trials(ecog_train_500, d['train_dg'], seed=42)
    te_trials, te_trial_labels = extract_test_trials(ecog_test_500, d['test_dg'], train_max=train_max)
    stats['n_train_trials'] = len(tr_trial_labels)
    stats['n_test_trials'] = len(te_trial_labels)

    for key, arr in [('train_trials', tr_trials),
                      ('test_trials', te_trials),
                      ('train_trial_labels', tr_trial_labels),
                      ('test_trial_labels', te_trial_labels)]:
        path = os.path.join(subject_dir, key + '.npy')
        if save_if_not_exists(path, arr):
            saved += 1
        else:
            skipped += 1
    stats['files_saved'] = saved
    stats['files_skipped'] = skipped

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BCI Competition IV Dataset 4 for the ECoG benchmark.")
    parser.add_argument('--subjects', nargs='+', default=['sub1', 'sub2', 'sub3'],
                        help='Subjects to process (default: all three).')
    parser.add_argument('--output', default='results/preprocessed',
                        help='Output directory for .npy files.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if cached files exist.')
    args = parser.parse_args()

    print("=" * 60)
    print("  ECoG FINGER MOVEMENT — PREPROCESSING PIPELINE")
    print("  Dataset: BCI Competition IV, Dataset 4")
    print("=" * 60)

    # Ensure raw data is available
    download_dataset()
    print()

    # Process each subject
    all_stats = {}
    for subj in args.subjects:
        print(subj + "...")
        stats = process_subject(subj, args.output)
        all_stats[subj] = stats
        print(f"  Channels: {stats.get('n_channels', '?')} (after electrode QC; {SUBJECT_CHANNELS.get(subj, '?')} raw)")
        print(f"  Sliding-window samples: train={stats['n_train_windows']}, test={stats['n_test_windows']}")
        print(f"  Trial-based samples:    train={stats['n_train_trials']}, test={stats['n_test_trials']}")
        print(f"  Classes: {stats['n_classes']} (Rest, Thumb, Index, Middle, Ring, Little)")
        print(f"  Files: {stats['files_saved']} saved, {stats['files_skipped']} skipped")
        print()

    print("=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("  Target: " + os.path.abspath(args.output))
    print("""
  Next steps:
    # Run full benchmark
    python run_benchmark.py --n_runs 3

    # DL only with trial-based segmentation
    python run_trial_dl.py --epochs 100 --patience 15

    # Ablation experiments
    python run_ablations.py --ablation all
""")
    print("=" * 60)


if __name__ == '__main__':
    main()
