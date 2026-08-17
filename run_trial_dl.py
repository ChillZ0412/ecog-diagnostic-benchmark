"""
Trial-based DL experiment: segment ECoG into discrete movement trials
instead of dense sliding windows. This matches the trial-based paradigm
used in original EEGNet / EEGConformer papers (BCI IV 2a).
"""

import argparse, json, os, sys, time, warnings
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from traditional.data_loader import download_dataset, load_subject
from traditional.evaluation import evaluate_classification
from traditional.dl_utils import (downsample_ecog, extract_trials, extract_test_trials,
                                   TRIAL_SAMPLES, TRIAL_MS)

SUBJECTS = ['sub1', 'sub2', 'sub3']
SEEDS = [42, 142, 242]


def train_dl(model, Xtr_t, ytr_t, Xte_t, yte_t, lr, betas, batch_size, n_epochs, patience, mixup_alpha=0.0):
    """Train one DL model on trial data with Mixup augmentation.

    Mixup (Zhang et al., 2018): linearly interpolates pairs of training samples
    and their labels, acting as a strong regularizer especially effective for
    small datasets.  Early stopping monitors training loss; given the limited
    data (153–510 trials), a separate validation set is not reserved (see Methods).

    Reference protocols: EEGNet uses 2:1:1 train/val/test (Lawhern 2018);
    we adapt by using all training data with Mixup as the primary regularizer.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    n_train = len(Xtr_t)
    indices = np.arange(n_train)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        model.train()
        train_loss = 0.0
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            Xb = Xtr_t[batch_idx].to(device)
            yb = ytr_t[batch_idx].to(device)

            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                perm = torch.randperm(len(Xb))
                X_mix = lam * Xb + (1 - lam) * Xb[perm]
                optimizer.zero_grad()
                logits = model(X_mix)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb[perm])
            else:
                optimizer.zero_grad()
                logits = model(Xb)
                loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        n_batches = max(1, (n_train + batch_size - 1) // batch_size)
        avg_loss = train_loss / n_batches
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred = torch.argmax(logits, dim=1).cpu().numpy()

    return pred, proba, train_time, 0.0, n_params


def run_subject_trial(subject_id, seed, n_epochs=100, patience=15, mixup_alpha=0.0, verbose=False):
    """Run trial-based DL on one subject."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    d = load_subject(subject_id)

    ecog_train_500 = downsample_ecog(d['train_ecog'])
    ecog_test_500 = downsample_ecog(d['test_ecog'])

    train_max = d['train_dg'].max(axis=0)   # per-finger train maxima (shared threshold)
    Xtr, ytr = extract_trials(ecog_train_500, d['train_dg'], seed=seed)
    Xte, yte = extract_test_trials(ecog_test_500, d['test_dg'], train_max=train_max)

    if verbose:
        unique_tr, cnt_tr = np.unique(ytr, return_counts=True)
        unique_te, cnt_te = np.unique(yte, return_counts=True)
        print(f"  {subject_id} trials: train={len(ytr)}, test={len(yte)}")
        print(f"    Train dist: {dict(zip(unique_tr, cnt_tr))}")
        print(f"    Test dist:  {dict(zip(unique_te, cnt_te))}")

    Xtr_t = torch.FloatTensor(Xtr)
    ytr_t = torch.LongTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    yte_t = torch.LongTensor(yte)

    n_ch = Xtr.shape[1]
    results = {}

    # ── EEGNet ──
    from braindecode.models import EEGNet
    model = EEGNet(n_outputs=6, n_chans=n_ch, n_times=TRIAL_SAMPLES,
                   F1=8, D=2, F2=16, kernel_length=250,
                   pool_mode='mean', drop_prob=0.5, final_conv_length='auto')
    yp, ypb, ttrain, tinfer, npar = train_dl(
        model, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=1e-3, betas=(0.9, 0.999), batch_size=64,
        n_epochs=n_epochs, patience=patience, mixup_alpha=mixup_alpha)

    ev = evaluate_classification(yte, yp)
    ev['train_time'] = ttrain
    ev['infer_time'] = tinfer
    ev['n_params'] = npar
    results['eegnet'] = ev

    if verbose:
        print(f"    EEGNet: F1={ev['macro_f1']:.3f}, BalAcc={ev['balanced_accuracy']:.3f}, Train={ttrain:.0f}s")

    # ── EEGConformer ──
    from braindecode.models import EEGConformer
    model = EEGConformer(n_outputs=6, n_chans=n_ch,
                         n_filters_time=40, filter_time_length=50,
                         pool_time_length=150, pool_time_stride=30,
                         drop_prob=0.5, num_layers=6, num_heads=10,
                         att_drop_prob=0.5, n_times=TRIAL_SAMPLES,
                         final_fc_length='auto')
    yp, ypb, ttrain, tinfer, npar = train_dl(
        model, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=2e-4, betas=(0.5, 0.999), batch_size=72,
        n_epochs=n_epochs, patience=patience, mixup_alpha=mixup_alpha)

    ev2 = evaluate_classification(yte, yp)
    ev2['train_time'] = ttrain
    ev2['infer_time'] = tinfer
    ev2['n_params'] = npar
    results['eegconformer'] = ev2

    if verbose:
        print(f"    Conformer: F1={ev2['macro_f1']:.3f}, BalAcc={ev2['balanced_accuracy']:.3f}, Train={ttrain:.0f}s")

    return {
        'subject': subject_id,
        'seed': seed,
        'results': results,
        'n_trials': {'train': len(ytr), 'test': len(yte)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', nargs='+', default=SUBJECTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    download_dataset()
    
    all_results = []
    t0 = time.time()
    
    for seed in args.seeds:
        for subj in args.subjects:
            print(f"[{time.strftime('%H:%M:%S')}] {subj} seed={seed}")
            r = run_subject_trial(subj, seed, n_epochs=args.epochs,
                                  patience=args.patience, mixup_alpha=args.mixup_alpha,
                                  verbose=args.verbose)
            all_results.append(r)
    
    runtime_s = time.time() - t0
    
    # ── Summary ──
    SEP = '=' * 78
    print(f"\n{SEP}")
    print("  TRIAL-BASED DL RESULTS")
    print(f"{SEP}")
    
    by_ms = defaultdict(list)
    for r in all_results:
        for mn, m in r['results'].items():
            if 'macro_f1' in m:
                by_ms[(mn, r['subject'])].append((r['seed'], m['macro_f1'], 
                    m.get('balanced_accuracy', 0), m.get('cohen_kappa', 0)))
    
    print(f"\n{'Method':<16} {'sub1':>8} {'sub2':>8} {'sub3':>8} {'Mean':>8}")
    print('-' * 55)
    for mn in ['eegnet', 'eegconformer']:
        vals = []
        for subj in ['sub1', 'sub2', 'sub3']:
            entries = [e[1] for e in by_ms.get((mn, subj), [])]
            vals.append(np.mean(entries) if entries else 0)
        print(f'{mn:<16} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {np.mean(vals):>8.3f}')
    
    # ── Save ──
    os.makedirs('results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = {
        'experiment': 'trial_based_dl',
        'timestamp': ts,
        'config': {'trial_ms': TRIAL_MS, 'epochs': args.epochs, 'patience': args.patience},
        'results': all_results,
        'runtime_s': runtime_s,
    }
    
    path = f'results/trial_dl_eegconformer_eegnet_{ts}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved → {path}')
    print(f'Runtime: {runtime_s/60:.1f} min')


if __name__ == '__main__':
    main()
