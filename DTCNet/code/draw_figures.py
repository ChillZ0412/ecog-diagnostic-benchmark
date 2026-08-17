"""Draw figures for the DTC paper: architecture + per-finger correlation heatmap + frequency-resolution ablation trend"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

BASE = 'C:/Users/75060/WorkBuddy/2026-07-20-12-42-08/dtcnet_regression'
FIG = os.path.join(BASE, 'analysis')
os.makedirs(FIG, exist_ok=True)
FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

# ============ 1. Architecture ============
def draw_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis('off')

    def block(x, y, w, h, text, fc='#e8f0fe', fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                                    fc=fc, ec='#1a73e8', lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

    # input
    block(0.2, 2.3, 1.8, 1.4, 'Input\n3D Spectrogram\n(C × 40 × 256)', fc='#fce8e6')
    # feature reduction
    block(2.6, 2.3, 1.8, 1.4, 'Feature Reduction\n1×1 Conv\nC×40 → 48', fc='#e8f0fe')
    # 5 encoder blocks
    enc_labels = ['Enc 1\n48→64\ndil=1', 'Enc 2\n64→96\ndil=2', 'Enc 3\n96→128\ndil=3',
                  'Enc 4\n128→128\ndil=1', 'Enc 5\n128→128\ndil=2']
    for i, lab in enumerate(enc_labels):
        block(5.0 + i*1.55, 3.4, 1.4, 1.2, lab, fc='#e6f4ea', fs=7)
    # 5 decoder blocks
    dec_labels = ['Dec 1\n128→96', 'Dec 2\n96→64', 'Dec 3\n64→48', 'Dec 4\n48→32', 'Dec 5\n32→32']
    for i, lab in enumerate(dec_labels):
        block(5.0 + i*1.55, 1.2, 1.4, 1.2, lab, fc='#fef7e0', fs=7)
    # output
    block(12.0, 2.3, 1.8, 1.4, 'Output\n1×1 Conv\n5 × 256\ntrajectory', fc='#f3e8fd')

    # arrows
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                                     mutation_scale=12, color='#555', lw=1.2))
    arrow(2.0, 3.0, 2.6, 3.0)
    arrow(4.4, 3.0, 5.0, 3.7)
    for i in range(4):
        arrow(6.4 + i*1.55, 3.4, 6.4 + i*1.55 + 0.15, 3.4)
    # encoder -> decoder (down)
    arrow(8.5, 3.4, 8.5, 2.4)
    # decoder -> output
    arrow(11.0, 1.8, 12.0, 2.9)
    # skip connections (dashed)
    for i in range(5):
        ax.plot([5.7 + i*1.55, 5.7 + (4-i)*1.55], [3.4, 2.4], '--', color='#999', lw=0.8)

    ax.set_title('DTCNet Architecture', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ fig_architecture.png')

# ============ 2. Per-finger correlation heatmap ============
def draw_correlation_heatmap():
    corr_mean = np.array([
        [1.000, -0.136, 0.023, -0.102, -0.061],
        [-0.136, 1.000, 0.202, 0.145, 0.246],
        [0.023, 0.202, 1.000, 0.166, 0.035],
        [-0.102, 0.145, 0.166, 1.000, 0.321],
        [-0.061, 0.246, 0.035, 0.321, 1.000],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(FINGERS); ax.set_yticklabels(FINGERS)
    ax.set_title('True Finger Movement Correlation (mean over subjects)', fontsize=11)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{corr_mean[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='black' if abs(corr_mean[i,j])<0.6 else 'white')
    plt.colorbar(im, label='Pearson r')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_finger_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ fig_finger_correlation.png')

# ============ 3. Frequency-resolution ablation trend ============
def draw_freq_ablation():
    freqs = [1, 10, 20, 40]
    r_mean = [0.1568, 0.4916, 0.5369, 0.5213]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(freqs, r_mean, 'o-', color='#1a73e8', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Morlet frequency bands')
    ax.set_ylabel('Mean official_r')
    ax.set_title('Frequency-Resolution Ablation', fontsize=12, fontweight='bold')
    ax.set_xticks(freqs)
    ax.set_xticklabels(['1\n(no Morlet)', '10', '20', '40\n(main)'])
    for x, y in zip(freqs, r_mean):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig_freq_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ fig_freq_ablation.png')

draw_architecture()
draw_correlation_heatmap()
draw_freq_ablation()
print('\nAll figures done')
