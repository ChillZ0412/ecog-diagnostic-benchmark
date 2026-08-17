"""Supplementary analysis: per-finger correlation + training curves
Pure analysis, zero training. Based on final main-experiment results (trajectory output + per-channel).

Outputs:
  analysis/finger_correlation.csv  — 5-finger true-movement correlation matrix (physiological coupling)
  analysis/training_curves.png     — 3-subject training loss curves
"""
import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'C:/Users/75060/WorkBuddy/2026-07-20-12-42-08/dtcnet_regression'
DATA = 'C:/Users/75060/WorkBuddy/data_dtcnet'
OUT = os.path.join(BASE, 'analysis')
os.makedirs(OUT, exist_ok=True)

FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

# ============ 1. Per-finger true-movement correlation (physiological coupling) ============
print('=' * 60)
print('Per-finger true-movement correlation (physiological coupling, test finger data)')
print('=' * 60)

corr_all = []
for sid in [1, 2, 3]:
    f = np.load(f'{DATA}/sub{sid}_test_finger.npy')  # (5, 20000)
    corr = np.corrcoef(f)  # 5x5
    corr_all.append(corr)
    print(f'\nSubject {sid} true finger-movement correlation matrix:')
    print('      ' + ' '.join(f'{x:>8}' for x in FINGERS))
    for i, name in enumerate(FINGERS):
        print(f'{name:>6} ' + ' '.join(f'{corr[i,j]:8.3f}' for j in range(5)))

corr_mean = np.mean(corr_all, axis=0)
print('\nMean correlation matrix (3 subjects):')
print('      ' + ' '.join(f'{x:>8}' for x in FINGERS))
for i, name in enumerate(FINGERS):
    print(f'{name:>6} ' + ' '.join(f'{corr_mean[i,j]:8.3f}' for j in range(5)))

# save CSV
import csv
with open(os.path.join(OUT, 'finger_correlation.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([''] + FINGERS)
    for i, name in enumerate(FINGERS):
        w.writerow([name] + [f'{corr_mean[i,j]:.3f}' for j in range(5)])

# key finding: Middle/Ring/Little coupling strength
print('\nKey finding (mean matrix):')
for i in range(5):
    for j in range(i+1, 5):
        v = corr_mean[i, j]
        flag = ' <- strong coupling' if abs(v) > 0.5 else ''
        print(f'  {FINGERS[i]}-{FINGERS[j]}: {v:.3f}{flag}')

# ============ 2. Training curves ============
print('\n' + '=' * 60)
print('Training curves')
print('=' * 60)

fig, ax = plt.subplots(figsize=(10, 5))
for sid in [1, 2, 3]:
    lp = os.path.join(BASE, f'results_final/sub{sid}_loss.npy')
    if os.path.exists(lp):
        loss = np.load(lp)
        ax.plot(loss, label=f'Subject {sid} (final={loss[-1]:.4f})', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Train Loss (MSE + Cosine)')
ax.set_title('DTCNet Training Curves (final configuration)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'training_curves.png'), dpi=150)
print(f'Training curves saved: {os.path.join(OUT, "training_curves.png")}')
print(f'Correlation matrix saved: {os.path.join(OUT, "finger_correlation.csv")}')
