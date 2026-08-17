"""
Self-Supervised Pretraining + Fine-tuning for DL models.

Strategy: SimCLR-style contrastive pretraining on unlabeled ECoG windows,
then supervised fine-tuning on trial-based labels.  Within-subject only
(fair comparison with traditional methods).

Reference: Chen et al. (2020) — SimCLR: a simple framework for contrastive
learning of visual representations (ICML 2020). We adapt this framework for
ECoG finger movement classification.
"""

import argparse, json, os, sys, time, warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

from traditional.data_loader import download_dataset, load_subject
from traditional.evaluation import evaluate_classification
from traditional.dl_utils import (downsample_ecog, extract_trials, extract_test_trials,
                                   TRIAL_SAMPLES, ECOG_FS_ORIG, DG_FS, DL_TARGET_FS,
                                   TRIAL_MS, DG_SAMPLES_PER_TRIAL, DS_RATIO)

SUBJECTS = ['sub1', 'sub2', 'sub3']
SEEDS = [42, 142, 242]


def extract_unlabeled_windows(ecog_500, n_per_subject=800):
    """Extract non-overlapping windows from ECoG for self-supervised pretraining."""
    n_total = ecog_500.shape[1]
    step = TRIAL_SAMPLES  # non-overlapping
    windows = []
    for start in range(0, n_total - TRIAL_SAMPLES + 1, step):
        windows.append(ecog_500[:, start:start + TRIAL_SAMPLES])
    X = np.stack(windows)
    # Sub-sample if too many
    if len(X) > n_per_subject:
        idx = np.random.RandomState(42).choice(len(X), n_per_subject, replace=False)
        X = X[idx]
    return X


def augment(x, noise_std=0.05, shift_max=10):
    """Create an augmented view of x for contrastive learning.
    
    Two augmentations: (1) additive Gaussian noise, (2) random time shift.
    Both are standard in EEG/ECoG self-supervised learning.
    """
    B, C, T = x.shape
    x_aug = x.clone()
    # Gaussian noise
    x_aug += torch.randn_like(x_aug) * noise_std * x_aug.std(dim=-1, keepdim=True)
    # Random circular time shift
    shift = torch.randint(-shift_max, shift_max + 1, (B,)).to(x.device)
    for i in range(B):
        if shift[i] != 0:
            x_aug[i] = torch.roll(x_aug[i], shifts=shift[i].item(), dims=-1)
    return x_aug


def simclr_pretrain(encoder, X_unlabeled_t, n_ch, model_type, n_epochs=50, lr=5e-4, batch_size=64):
    """Pretrain encoder using SimCLR on unlabeled ECoG windows.
    
    Returns pretrained encoder state dict.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    n_samples = len(X_unlabeled_t)
    
    # Projection head: encoder output → 128-dim embedding
    if model_type == 'eegnet':
        # Get encoder output dim: run a dummy forward pass
        with torch.no_grad():
            dummy = encoder(X_unlabeled_t[:2].to(device))
            enc_dim = dummy.shape[1]
    else:
        with torch.no_grad():
            dummy = encoder(X_unlabeled_t[:2].to(device))
            enc_dim = dummy.shape[1]
    
    proj_head = nn.Sequential(
        nn.Linear(enc_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()), lr=lr)
    
    indices = np.arange(n_samples)
    t0 = time.time()
    
    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            Xb = X_unlabeled_t[batch_idx].to(device)
            
            # Create two augmented views
            v1 = augment(Xb)
            v2 = augment(Xb)
            
            # Encode
            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))
            
            # Normalize
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            # NT-Xent (contrastive) loss
            z = torch.cat([z1, z2], dim=0)  # (2B, 64)
            sim = torch.mm(z, z.T)  # (2B, 2B) cosine similarity
            
            # Temperature scaling
            sim = sim / 0.1
            
            # Positive pairs: (i, i+B) for each i in [0, B)
            labels = torch.arange(len(batch_idx)).to(sim.device)
            labels = torch.cat([labels + len(batch_idx), labels], dim=0)
            
            # Remove self-similarity for numerical stability
            sim = sim - torch.eye(len(z)).to(sim.device) * 1e9
            
            loss = F.cross_entropy(sim, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"    SSL epoch {epoch+1}/{n_epochs}, loss={epoch_loss/n_batches:.4f}")
    
    pretrain_time = time.time() - t0
    return encoder.state_dict(), pretrain_time


def train_supervised(model, Xtr_t, ytr_t, Xte_t, yte_t, lr, betas, batch_size,
                     n_epochs, patience, mixup_alpha=0.0):
    """Supervised training with Mixup augmentation."""
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
        
        # Early stopping on training loss
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


def run_subject_ssl(subject_id, seed, n_epochs_ssl=50, n_epochs_sup=100, 
                    patience=15, mixup_alpha=0.0, verbose=False):
    """SSL pretrain + supervised fine-tune on one subject."""
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

    d = load_subject(subject_id)
    ecog_train_500 = downsample_ecog(d['train_ecog'])
    ecog_test_500 = downsample_ecog(d['test_ecog'])
    n_ch = ecog_train_500.shape[0]
    
    # Unlabeled windows for SSL
    X_unlabeled = extract_unlabeled_windows(ecog_train_500)
    X_unlabeled_t = torch.FloatTensor(X_unlabeled)
    
    # Labeled trials for supervised fine-tuning
    train_max = d['train_dg'].max(axis=0)   # per-finger train maxima (shared threshold)
    Xtr, ytr = extract_trials(ecog_train_500, d['train_dg'], seed=seed)
    Xte, yte = extract_test_trials(ecog_test_500, d['test_dg'], train_max=train_max)
    
    if verbose:
        print(f"  {subject_id}: SSL windows={len(X_unlabeled)}, labeled trials={len(ytr)}")
    
    Xtr_t = torch.FloatTensor(Xtr)
    ytr_t = torch.LongTensor(ytr)
    Xte_t = torch.FloatTensor(Xte)
    yte_t = torch.LongTensor(yte)
    
    results = {}
    
    # ── EEGNet + SSL ──
    from braindecode.models import EEGNet
    model = EEGNet(n_outputs=6, n_chans=n_ch, n_times=TRIAL_SAMPLES,
                   F1=8, D=2, F2=16, kernel_length=250,
                   pool_mode='mean', drop_prob=0.5, final_conv_length='auto')
    
    if verbose:
        print(f"    EEGNet SSL pretraining...")
    encoder_state, pretrain_time = simclr_pretrain(
        model, X_unlabeled_t, n_ch, 'eegnet', 
        n_epochs=n_epochs_ssl, lr=5e-4, batch_size=64)
    model.load_state_dict(encoder_state)
    
    if verbose:
        print(f"    EEGNet supervised fine-tuning...")
    yp, ypb, ttrain, _, npar = train_supervised(
        model, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=1e-3, betas=(0.9, 0.999), batch_size=64,
        n_epochs=n_epochs_sup, patience=patience, mixup_alpha=mixup_alpha)
    
    ev = evaluate_classification(yte, yp)
    ev['train_time'] = ttrain + pretrain_time
    ev['pretrain_time'] = pretrain_time
    ev['n_params'] = npar
    results['eegnet'] = ev
    
    if verbose:
        print(f"    EEGNet: F1={ev['macro_f1']:.3f}, BalAcc={ev['balanced_accuracy']:.3f}, Pretrain={pretrain_time:.0f}s, FT={ttrain:.0f}s")
    
    # ── EEGConformer + SSL ──
    from braindecode.models import EEGConformer
    model = EEGConformer(n_outputs=6, n_chans=n_ch,
                         n_filters_time=40, filter_time_length=50,
                         pool_time_length=150, pool_time_stride=30,
                         drop_prob=0.5, num_layers=6, num_heads=10,
                         att_drop_prob=0.5, n_times=TRIAL_SAMPLES,
                         final_fc_length='auto')
    
    if verbose:
        print(f"    Conformer SSL pretraining...")
    encoder_state, pretrain_time2 = simclr_pretrain(
        model, X_unlabeled_t, n_ch, 'eegconformer',
        n_epochs=n_epochs_ssl, lr=5e-4, batch_size=64)
    model.load_state_dict(encoder_state)
    
    if verbose:
        print(f"    Conformer supervised fine-tuning...")
    yp, ypb, ttrain, _, npar = train_supervised(
        model, Xtr_t, ytr_t, Xte_t, yte_t,
        lr=2e-4, betas=(0.5, 0.999), batch_size=72,
        n_epochs=n_epochs_sup, patience=patience, mixup_alpha=mixup_alpha)
    
    ev2 = evaluate_classification(yte, yp)
    ev2['train_time'] = ttrain + pretrain_time2
    ev2['pretrain_time'] = pretrain_time2
    ev2['n_params'] = npar
    results['eegconformer'] = ev2
    
    if verbose:
        print(f"    Conformer: F1={ev2['macro_f1']:.3f}, BalAcc={ev2['balanced_accuracy']:.3f}, Pretrain={pretrain_time2:.0f}s, FT={ttrain:.0f}s")
    
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
    parser.add_argument('--ssl_epochs', type=int, default=50)
    parser.add_argument('--sup_epochs', type=int, default=100)
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
            r = run_subject_ssl(subj, seed,
                               n_epochs_ssl=args.ssl_epochs,
                               n_epochs_sup=args.sup_epochs,
                               patience=args.patience,
                               mixup_alpha=args.mixup_alpha,
                               verbose=args.verbose)
            all_results.append(r)
    
    runtime_s = time.time() - t0
    
    # ── Summary ──
    SEP = '=' * 78
    print(f"\n{SEP}")
    print("  SSL + SUPERVISED DL RESULTS")
    print(f"{SEP}")
    
    by_ms = defaultdict(list)
    for r in all_results:
        for mn, m in r['results'].items():
            if 'macro_f1' in m:
                by_ms[(mn, r['subject'])].append(m['macro_f1'])
    
    print(f"\n{'Method':<16} {'sub1':>8} {'sub2':>8} {'sub3':>8} {'Mean':>8}")
    print('-' * 55)
    for mn in ['eegnet', 'eegconformer']:
        vals = []
        for subj in ['sub1', 'sub2', 'sub3']:
            entries = by_ms.get((mn, subj), [])
            vals.append(np.mean(entries) if entries else 0)
        print(f'{mn:<16} {vals[0]:>8.3f} {vals[1]:>8.3f} {vals[2]:>8.3f} {np.mean(vals):>8.3f}')
    
    # ── Save ──
    os.makedirs('results', exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out = {
        'experiment': 'ssl_trial_dl',
        'timestamp': ts,
        'config': {'ssl_epochs': args.ssl_epochs, 'sup_epochs': args.sup_epochs,
                   'mixup_alpha': args.mixup_alpha},
        'results': all_results,
        'runtime_s': runtime_s,
    }
    path = f'results/ssl_dl_eegconformer_eegnet_{ts}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {path}')
    print(f'Runtime: {runtime_s/60:.1f} min')


if __name__ == '__main__':
    main()
