"""
All publication figures — unified journal style guide.
Colors, fonts, spacing, and export settings are applied consistently.

Style: IEEE TBME / J. Neural Engineering
Font: Arial 7–10 pt, 300 dpi, PDF+PNG
Palette: blue (CSP), burgundy (LGB), amber (EEGNet), rust (Conformer)
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Style ──
COLORS = {
    'csp':       '#2E86AB',
    'lgb':       '#A23B72',
    'eegnet':    '#F18F01',
    'conformer': '#C73E1D',
    'sub1':      '#4A90D9',
    'sub2':      '#E67E22',
    'sub3':      '#27AE60',
}
CLASSES = ['Rest','Thumb','Index','Middle','Ring','Little']
CLASS_COLORS = ['#95A5A6','#2E86AB','#E74C3C','#F39C12','#27AE60','#8E44AD']

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
    'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'savefig.pad_inches':0.03,'axes.spines.top':False,'axes.spines.right':False,
})
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

# ── Data paths (dynamic: pick the most recently modified matching file) ──
def _latest(pattern):
    """Return the most recently modified file matching a glob pattern.

    Raises FileNotFoundError with an actionable message if no file matches
    (e.g. before the benchmark has been run).
    """
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"No result file matching '{pattern}'. Run the benchmark first.")
    return matches[-1]


TRAD   = _latest('results/benchmark_csp_lgb_*.json')
DL_MIX = _latest('results/trial_dl_eegconformer_eegnet_*.json')
ABL    = _latest('results/ablations/ablation_all_*.json')

def load(p):
    """Load a results JSON file."""
    with open(p) as f: return json.load(f)

def get_f1s(data, mk, subj, metric='macro_f1'):
    """Collect macro-F1 values for a method across a subject's runs."""
    return [r['results'][mk][metric] for r in data['results']
            if r['subject']==subj and mk in r.get('results',{})]

def save(fn):
    """Save current figure to both PNG and PDF."""
    plt.savefig(f'{OUT}/{fn}.png', dpi=300)
    plt.savefig(f'{OUT}/{fn}.pdf')
    plt.close()
    print(f'  {fn}')

# ═══════════════════════════════════════════════════════════════
# Fig1 — Label Distribution
# ═══════════════════════════════════════════════════════════════
def fig1():
    """Fig1 - Label Distribution: train/test class proportions per subject."""
    from traditional.data_loader import load_subject, generate_labels

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.85, bottom=0.14, wspace=0.35)

    for ax, subj in zip(axes, ['sub1','sub2','sub3']):
        d = load_subject(subj)
        train_max = d['train_dg'].max(axis=0)
        tr = generate_labels(d['train_dg'])
        te = generate_labels(d['test_dg'], max_angles=train_max)
        x = np.arange(6)
        w = 0.35
        tr_pct = np.bincount(tr, minlength=6) / len(tr) * 100
        te_pct = np.bincount(te, minlength=6) / len(te) * 100

        ax.bar(x-w/2, tr_pct, w, color=CLASS_COLORS, edgecolor='white', linewidth=0.3, alpha=0.9)
        ax.bar(x+w/2, te_pct, w, color=CLASS_COLORS, edgecolor='black', linewidth=0.6, alpha=0.3)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=40, ha='right')
        ax.set_title(subj.replace('sub', 'Subject '), fontweight='bold')
        if ax == axes[0]: ax.set_ylabel('Percentage (%)')

        # Rest shift: place the annotation in the upper-right of each panel
        # (instead of overlapping the tallest Rest bar). White rounded box keeps
        # it readable without colliding with bars or axis labels.
        shift = te_pct[0] - tr_pct[0]
        ax.text(0.97, 0.96, f'Rest Δ={shift:+.1f}pp',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold', color='#C0392B',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='#C0392B', linewidth=0.6, alpha=0.92))
        ax.grid(axis='y', alpha=0.15)
        ax.set_ylim(0, max(tr_pct[0], te_pct[0]) * 1.18)

    axes[0].legend(['Train (400 s)','Test (200 s)'], fontsize=7, loc='upper right', framealpha=0.9)
    axes[2].legend([],[])
    fig.suptitle('Label Distribution — Train vs Test', fontweight='bold', fontsize=11, y=1.01)
    save('fig1_label_distribution')

# ═══════════════════════════════════════════════════════════════
# Fig2 — Classification Performance
# ═══════════════════════════════════════════════════════════════
def fig2():
    """Fig2 - Classification Performance: macro-F1 across methods and subjects."""
    methods = [
        ('CSP+LDA',       'csp',       TRAD,   COLORS['csp']),
        ('Spectral+LGB',  'lgb',       TRAD,   COLORS['lgb']),
        ('EEGNet',  'eegnet',    DL_MIX, COLORS['eegnet']),
        ('EEG-Conformer','eegconformer', DL_MIX, COLORS['conformer']),
    ]
    metrics = [('balanced_accuracy','Balanced Accuracy'),('macro_f1','Macro F1'),('cohen_kappa',"Cohen's κ")]
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.86, bottom=0.15, wspace=0.35)
    x = np.arange(3); w = 0.18
    
    for ai, (met, lab) in enumerate(metrics):
        ax = axes[ai]
        for mi, (mn, mk, path, col) in enumerate(methods):
            data = load(path)
            means, stds = [], []
            for subj in ['sub1','sub2','sub3']:
                v = get_f1s(data, mk, subj, met)
                means.append(np.mean(v)); stds.append(np.std(v, ddof=1) if len(v)>1 else 0)
            off = (mi-(4-1)/2)*w
            ax.bar(x+off, means, w*0.85, color=col, edgecolor='white', linewidth=0.3,
                   label=mn if ai==2 else None, yerr=stds if any(s>0.002 for s in stds) else None,
                   capsize=2, error_kw={'linewidth':0.6})
        ax.set_xticks(x); ax.set_xticklabels(['Subject 1','Subject 2','Subject 3'])
        ax.set_title(lab, fontweight='bold', pad=6)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel(lab, fontsize=8)
        if met=='cohen_kappa': ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.axhline(y=1/6, color='gray', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.grid(axis='y', alpha=0.15)
    
    axes[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), framealpha=0.9,
                   fontsize=7.5, title='Method', title_fontsize=8)
    fig.suptitle('Classification Performance — Within-Subject Evaluation', fontweight='bold', fontsize=11, y=1.01)
    save('fig2_classification_performance')

# ═══════════════════════════════════════════════════════════════
# Fig3 — Confusion Matrices
# ═══════════════════════════════════════════════════════════════
def fig3():
    """Fig3 - Confusion Matrices: per-subject confusion matrices."""
    methods = [
        ('CSP+LDA','csp',TRAD), ('Spectral+LGB','lgb',TRAD),
        ('EEGNet','eegnet',DL_MIX), ('EEG-Conformer','eegconformer',DL_MIX),
    ]
    fig, axes = plt.subplots(2,2,figsize=(7,7))
    fig.subplots_adjust(left=0.08, right=0.89, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
    axes = axes.flatten()
    
    for ax, (mn, mk, path) in zip(axes, methods):
        data = load(path)
        cms = [np.array(r['results'][mk]['confusion_matrix']) for r in data['results']
               if r['subject']=='sub3' and mk in r.get('results',{})]
        cm_sum = sum(cms)
        cm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
        
        im = ax.imshow(cm, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CLASSES, rotation=40, ha='right', fontsize=6.5)
        ax.set_yticklabels(CLASSES, fontsize=6.5)
        ax.set_title(mn, fontweight='bold', fontsize=10)
        
        for i in range(6):
            for j in range(6):
                v = cm[i,j]
                ax.text(j,i, f'{v:.0%}' if v>0.04 else '', ha='center', va='center',
                       fontsize=7, color='white' if v>0.6 else 'black',
                       fontweight='bold' if i==j else 'normal')
        ax.set_xlabel('Predicted', fontsize=8); ax.set_ylabel('True', fontsize=8)
    
    cbar = fig.colorbar(im, ax=axes, location='right', fraction=0.03, pad=0.02)
    cbar.set_label('Recall', fontsize=8)
    
    fig.suptitle('Normalised Confusion Matrices — Subject 3', fontweight='bold', fontsize=11)
    save('fig3_confusion_matrices')

# ═══════════════════════════════════════════════════════════════
# Fig4 — Cohen's d
# ═══���═══════════════════════════════════════════════════════════
def fig4():
    """Fig4 - Cohen's d: pairwise effect sizes between methods."""
    pairs = [
        ('CSP+LDA', 'csp', 'Spectral+LGB', 'lgb'),
        ('CSP+LDA', 'csp', 'EEGNet', 'eegnet'),
        ('CSP+LDA', 'csp', 'EEG-Conformer', 'eegconformer'),
        ('Spectral+LGB','lgb','EEGNet','eegnet'),
        ('Spectral+LGB','lgb','EEG-Conformer','eegconformer'),
        ('EEGNet','eegnet','EEG-Conformer','eegconformer'),
    ]
    ds, labs = [], []
    for n1, k1, n2, k2 in pairs:
        v1, v2 = [], []
        for subj in ['sub1','sub2','sub3']:
            p1 = TRAD if k1 in ('csp','lgb') else DL_MIX
            p2 = TRAD if k2 in ('csp','lgb') else DL_MIX
            v1.extend(get_f1s(load(p1), k1, subj))
            v2.extend(get_f1s(load(p2), k2, subj))
        v1, v2 = np.array(v1), np.array(v2)
        p = np.sqrt(((len(v1)-1)*np.std(v1,ddof=1)**2+(len(v2)-1)*np.std(v2,ddof=1)**2)/(len(v1)+len(v2)-2))
        ds.append((np.mean(v1)-np.mean(v2))/p); labs.append(f'{n1} vs {n2}')
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cols = ['#E74C3C' if abs(d)>=0.8 else '#F39C12' if abs(d)>=0.5 else '#95A5A6' for d in ds]
    ax.barh(np.arange(6), ds, color=cols, edgecolor='white', linewidth=0.5, height=0.6)
    
    for i, (y,d) in enumerate(zip(np.arange(6), ds)):
        mag = 'v.large' if abs(d)>=1.3 else 'large' if abs(d)>=0.8 else 'med' if abs(d)>=0.5 else 'small'
        ax.text(d+0.03*np.sign(d), y, f'{d:+.2f}', va='center',
               fontsize=7.5, fontweight='bold')
    
    for xr, ls in [(0.2,':'),(0.5,'--'),(0.8,'-')]:
        ax.axvline(x=xr, color='gray', linestyle=ls, linewidth=0.6, alpha=0.4)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("Cohen's d", fontsize=9)
    ax.set_title('Effect Size Between Method Pairs', fontweight='bold', fontsize=10)
    ax.set_xlim(-1.8, 1.8); ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    save('fig4_cohens_d')

# ═══════════════════════════════════════════════════════════════
# Fig5 — Efficiency + Performance
# ═══════════════════════════════════════════════════════════════
def fig5():
    """Fig5 - Efficiency: performance vs. parameter-count trade-off."""
    methods = [
        ('CSP+LDA','csp',TRAD,COLORS['csp']),
        ('Spectral\n+LGB','lgb',TRAD,COLORS['lgb']),
        ('EEGNet','eegnet',DL_MIX,COLORS['eegnet']),
        ('EEG-Conformer','eegconformer',DL_MIX,COLORS['conformer']),
    ]
    f1m, f1s, tm, ts, nms, cls = [],[],[],[],[],[]
    for mn,mk,path,col in methods:
        fv, tv = [], []
        for r in load(path)['results']:
            if mk in r.get('results',{}):
                fv.append(r['results'][mk]['macro_f1'])
                tv.append(max(r['results'][mk].get('train_time',0),0.01))
        if fv:
            f1m.append(np.mean(fv)); f1s.append(np.std(fv,ddof=1))
            tm.append(np.mean(tv)); ts.append(np.std(tv,ddof=1))
            nms.append(mn); cls.append(col)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.15, wspace=0.45)
    y = np.arange(len(nms))
    
    ax1.barh(y, f1m, color=cls, edgecolor='white', height=0.6, xerr=f1s, capsize=3)
    for i,v in enumerate(f1m):
        ax1.text(v+0.03, i, f'{v:.3f}', va='center', fontsize=8.5, fontweight='bold', color='#333')
    ax1.set_yticks(y); ax1.set_yticklabels(nms)
    ax1.set_title('Performance', fontweight='bold', fontsize=10)
    ax1.set_xlim(0, 0.5); ax1.axvline(1/6, color='gray', ls='--', lw=0.6, alpha=0.4)
    ax1.invert_yaxis(); ax1.grid(axis='x', alpha=0.15)
    
    ax2.barh(y, tm, color=cls, edgecolor='white', height=0.6, xerr=ts, capsize=3)
    for i,v in enumerate(tm):
        lbl = f'{v*1000:.0f} ms' if v<1 else f'{v:.1f} s' if v<60 else f'{v/60:.0f} min'
        ax2.text(v*1.1, i, lbl, va='center', fontsize=8, fontweight='bold')
    ax2.set_yticks(y); ax2.set_yticklabels([])
    ax2.set_title('Efficiency', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Training Time (linear scale)')
    ax2.invert_yaxis(); ax2.grid(axis='x', alpha=0.15)
    ax2.set_xlim(0, max(tm)*1.4)
    
    fig.suptitle('Performance vs Computational Cost', fontweight='bold', fontsize=11, y=1.01)
    save('fig5_efficiency')

# ═══════════════════════════════════════════════════════════════
# FigS1 — Per-class F1 Heatmap
# ═══════════════════════════════════════════════════════════════
def figS1():
    """FigS1 - Per-class F1 heatmap across fingers and subjects."""
    methods = [('CSP+LDA','csp',TRAD),('Spectral+LGB','lgb',TRAD),
               ('EEGNet','eegnet',DL_MIX),('EEG-Conformer','eegconformer',DL_MIX)]
    mat, rlabs = [], []
    for mn,mk,path in methods:
        data = load(path)
        for subj in ['sub1','sub2','sub3']:
            cms = [np.array(r['results'][mk]['confusion_matrix']) for r in data['results']
                   if r['subject']==subj and mk in r.get('results',{})]
            cm_sum = sum(cms)
            cm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
            f1s = [2*cm[i,i]/(cm[i,i]+cm[:,i].sum()+cm[i,:].sum()) for i in range(6)]
            mat.append(f1s); rlabs.append(f'{mn} | {subj}')
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(np.array(mat), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(6)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(12)); ax.set_yticklabels(rlabs, fontsize=6.5)
    ax.set_title('Per-Class F1 Score', fontweight='bold')
    
    for i in range(12):
        for j in range(6):
            v = mat[i][j]
            ax.text(j,i, f'{v:.2f}' if v>0.01 else '.', ha='center', va='center',
                   fontsize=6.5, color='white' if v>0.5 else 'black',
                   fontweight='bold' if v>0.4 else 'normal')
    
    for i in range(3):
        ax.axhline(y=i*4-0.5, color='#999', linewidth=1)
    
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label('F1 Score', fontsize=8)
    save('figS1_per_class_f1')

# ═══════════════════════════════════════════════════════════════
# FigS2-S4 — Ablations (simplified)
# ═══════════════════════════════════════════════════════════════

def figS2():
    """CSP frequency band ablation."""
    abl = load(ABL)['ablations']['csp_band']['results']
    names = {'optimal':'(65-175)Hz','highgamma':'(60-200)Hz','widegamma':'(30-200)Hz','allgamma':'(8-200)Hz'}
    order = ['optimal','highgamma','widegamma','allgamma']
    
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(4); w = 0.25
    for si, subj in enumerate(['sub1','sub2','sub3']):
        vals = [abl[bn][subj]['macro_f1'] for bn in order]
        ax.bar(x+(si-1)*w, vals, w, label=subj, color=[COLORS['sub1'],COLORS['sub2'],COLORS['sub3']][si],
               edgecolor='white', linewidth=0.3)
        for bi, v in enumerate(vals):
            ax.text(bi+(si-1)*w, v+0.008, f'{v:.3f}', ha='center', fontsize=6)
    
    ax.set_xticks(x); ax.set_xticklabels([names[bn] for bn in order], rotation=25, ha='right')
    ax.set_ylabel('Macro F1'); ax.set_title('CSP Frequency Band Selection', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylim(0, 0.55)
    ax.axhline(1/6, color='gray', ls='--', lw=0.6, alpha=0.4)
    ax.grid(axis='y', alpha=0.15)
    save('figS2_csp_frequency_band')

def figS3():
    """Window length ablation."""
    abl = load(ABL)['ablations']['window']['results']
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(3); w = 0.35
    for mi, (mk, col) in enumerate([('csp',COLORS['csp']),('lgb',COLORS['lgb'])]):
        vals = [np.mean([abl[wn][subj][mk]['macro_f1'] for subj in ['sub1','sub2','sub3']]) for wn in ['250ms','500ms','1000ms']]
        ax.bar(x+(mi-0.5)*w, vals, w, label=mk.upper(), color=col, edgecolor='white', linewidth=0.3)
        for i, v in enumerate(vals):
            ax.text(i+(mi-0.5)*w, v+0.005, f'{v:.3f}', ha='center', fontsize=7)
    
    ax.set_xticks(x); ax.set_xticklabels(['250 ms','500 ms','1000 ms'])
    ax.set_ylabel('Macro F1'); ax.set_title('Temporal Window Length', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylim(0.2, 0.35)
    ax.grid(axis='y', alpha=0.15)
    save('figS3_window_length')

def figS4():
    """Classifier choice ablation."""
    abl = load(ABL)['ablations']['classifier']['results']
    
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(3); w = 0.35
    for ci, (ck, cn, col) in enumerate([('LGB','LightGBM',COLORS['lgb']),('SVM','RBF-SVM',COLORS['csp'])]):
        vals = [abl[ck][subj]['macro_f1'] for subj in ['sub1','sub2','sub3']]
        ax.bar(x+(ci-0.5)*w, vals, w, label=cn, color=col, edgecolor='white', linewidth=0.3)
        for i, v in enumerate(vals):
            ax.text(i+(ci-0.5)*w, v+0.01, f'{v:.3f}', ha='center', fontsize=7.5)
    
    ax.set_xticks(x); ax.set_xticklabels(['Subject 1','Subject 2','Subject 3'])
    ax.set_ylabel('Macro F1'); ax.set_title('Spectral Features: LGB vs RBF-SVM', fontweight='bold')
    ax.legend(fontsize=7); ax.set_ylim(0, 0.55)
    ax.axhline(1/6, color='gray', ls='--', lw=0.6, alpha=0.4)
    ax.grid(axis='y', alpha=0.15)
    save('figS4_classifier_choice')

def figS5():
    """DL optimization trajectory — supervised baseline → +Mixup → +SSL."""
    # Mixup and SSL ablation results both live in the `--ablation all` output file
    # (dl_ablation_all_*.json holds 'mixup' and 'ssl' keys), which after the
    # ch50 bad-electrode fix contains fresh no-Mixup data for both ablations.
    dl_ab = load(_latest('results/ablations/dl_ablation_all_*.json'))['ablations']
    mixup = dl_ab['mixup']
    ssl = dl_ab['ssl']

    def _mean_std(runs, mk):
        vals = [r['results'][mk]['macro_f1'] for r in runs if mk in r.get('results', {})]
        return np.mean(vals), np.std(vals, ddof=1)

    # Three config points: supervised baseline, +Mixup, +SSL
    configs = [mixup['without_mixup'], mixup['with_mixup'], ssl['with_ssl']]
    labels = ['Supervised\nBaseline', '+Mixup', '+SSL']

    en_mean, en_std, ec_mean, ec_std = [], [], [], []
    for runs in configs:
        em, es = _mean_std(runs, 'eegnet')
        cm, cs = _mean_std(runs, 'eegconformer')
        en_mean.append(em); en_std.append(es)
        ec_mean.append(cm); ec_std.append(cs)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(3)

    ax.plot(x, en_mean, 'o-', color=COLORS['eegnet'], lw=1.8, ms=8, mfc='white', mew=1.5, label='EEGNet')
    ax.fill_between(x, np.array(en_mean)-np.array(en_std), np.array(en_mean)+np.array(en_std),
                    color=COLORS['eegnet'], alpha=0.15)
    ax.plot(x, ec_mean, 's-', color=COLORS['conformer'], lw=1.8, ms=8, mfc='white', mew=1.5, label='EEGConformer')
    ax.fill_between(x, np.array(ec_mean)-np.array(ec_std), np.array(ec_mean)+np.array(ec_std),
                    color=COLORS['conformer'], alpha=0.15)

    for i in range(3):
        ax.annotate(f'{en_mean[i]:.3f}', (x[i], en_mean[i]), xytext=(0,-14),
                   textcoords='offset points', ha='center', fontsize=7.5, color=COLORS['eegnet'], fontweight='bold')
        ax.annotate(f'{ec_mean[i]:.3f}', (x[i], ec_mean[i]), xytext=(0,12),
                   textcoords='offset points', ha='center', fontsize=7.5, color=COLORS['conformer'], fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Macro F1'); ax.set_title('DL Optimization Trajectory (Mixup & SSL)', fontweight='bold', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

    # Best traditional method reference line (read dynamically, not hardcoded).
    # Compact in-axes label on the right edge instead of a floating caption.
    trad_data = load(TRAD)
    lgb_f1s = [r['results']['lgb']['macro_f1'] for r in trad_data['results'] if 'lgb' in r.get('results', {})]
    if lgb_f1s:
        lgb_mean = float(np.mean(lgb_f1s))
        ax.axhline(lgb_mean, color=COLORS['lgb'], ls='--', lw=1.0, alpha=0.7)
        ax.text(2.92, lgb_mean - 0.005, f'LGB baseline ({lgb_mean:.3f})',
                fontsize=7, color=COLORS['lgb'], ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                          edgecolor=COLORS['lgb'], linewidth=0.5, alpha=0.9))

    ax.axhline(1/6, color='gray', ls=':', lw=0.6, alpha=0.4)
    ax.set_ylim(0.05, 0.45); ax.grid(axis='y', alpha=0.15)
    save('figS5_dl_optimization')

# ═══════════════════════════════════════════════════════════════
# Fig7 — Paired Differences (Wilcoxon signed-rank, per-finger × subject)
# ═══════════════════════════════════════════════════════════════
def _per_finger_f1(mk, path):
    """Return per-finger F1 vector (5 fingers × 3 subjects = 15 values),
    averaged across seeds for each (subject, finger) cell — matching
    ``compute_combined_stats.method_per_finger`` (used for the reported
    p-values)."""
    data = load(path)
    out = []
    for subj in ['sub1', 'sub2', 'sub3']:
        for f in ['Thumb', 'Index', 'Middle', 'Ring', 'Little']:
            vals = []
            for r in data['results']:
                if r['subject'] == subj and mk in r.get('results', {}):
                    pc = r['results'][mk].get('per_class', {})
                    if f in pc:
                        vals.append(pc[f]['f1'])
            out.append(float(np.mean(vals)))
    return np.array(out)


def fig7():
    """Fig7 - Paired Differences: Wilcoxon signed-rank (per-finger × subject, n = 15)."""
    from scipy.stats import wilcoxon
    methods = [('CSP+LDA', 'csp', TRAD), ('Spectral+LGB', 'lgb', TRAD),
               ('EEGNet', 'eegnet', DL_MIX), ('EEG-Conformer', 'eegconformer', DL_MIX)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    diffs, plabs = [], []
    for i, j in pairs:
        v1 = _per_finger_f1(methods[i][1], methods[i][2])
        v2 = _per_finger_f1(methods[j][1], methods[j][2])
        d = v1 - v2
        _, p = wilcoxon(v1, v2)
        diffs.append(d)
        plabs.append(f'{methods[i][0]} − {methods[j][0]}   (p = {p:.4f})')

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.14)
    y = np.arange(len(pairs))
    for yi, d in enumerate(diffs):
        ax.scatter(d, np.full_like(d, yi), s=22, alpha=0.45, color='#7F8C8D', zorder=2)
        ax.plot([np.mean(d)], [yi], 'D', color='#C0392B', ms=9, zorder=3)
        ax.axhline(y=yi, color='#BDC3C7', lw=0.5, alpha=0.6, zorder=1)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(plabs, fontsize=7)
    ax.set_xlabel('Macro F1 difference (per finger × subject, n = 15)')
    ax.set_title('Paired Differences — Wilcoxon Signed-Rank', fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    save('fig7_paired_differences')

# ═══════════════════════════════════════════════════════════════
# Fig8 — Critical Difference Diagram (Demšar post-hoc, k = 4, N = 3)
# ═══════════════════════════════════════════════════════════════
def fig8():
    """Fig8 - Critical Difference diagram (Demšar post-hoc, k = 4, N = 3)."""
    methods = [('CSP+LDA', 'csp', TRAD), ('Spectral+LGB', 'lgb', TRAD),
               ('EEGNet', 'eegnet', DL_MIX), ('EEG-Conformer', 'eegconformer', DL_MIX)]
    k = len(methods)
    N = 3

    # Macro F1 matrix (k methods × N subjects)
    mat = np.zeros((k, N))
    for mi, (_, mk, path) in enumerate(methods):
        data = load(path)
        for si, subj in enumerate(['sub1', 'sub2', 'sub3']):
            vals = [r['results'][mk]['macro_f1'] for r in data['results']
                    if r['subject'] == subj and mk in r.get('results', {})]
            mat[mi, si] = np.mean(vals)

    # Average rank per method (1 = best)
    ranks = np.zeros(k)
    for si in range(N):
        order = np.argsort(-mat[:, si])
        for rank, mi in enumerate(order):
            ranks[mi] += rank + 1
    ranks /= N

    # Critical difference (two-sided Nemenyi)
    from scipy.stats import studentized_range
    q = studentized_range.ppf(0.95, k, np.inf)
    CD = q * np.sqrt(k * (k + 1) / (6 * N))

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.20)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.7, 1.1)
    ax.axhline(0, color='black', lw=1.0)
    ax.text(k + 0.35, 0.02, 'best →', fontsize=7, color='#7F8C8D', ha='right')
    ax.text(0.65, 0.02, '← worst', fontsize=7, color='#7F8C8D')

    method_colors = [COLORS['csp'], COLORS['lgb'], COLORS['eegnet'], COLORS['conformer']]
    # Stagger label heights in rank order so adjacent ranks don't overlap.
    sorted_idx = np.argsort(ranks)
    label_y = {}
    for pos, mi in enumerate(sorted_idx):
        # Alternating heights: low (0.20) / high (0.55)
        label_y[mi] = 0.55 if (pos % 2 == 1) else 0.20

    for mi, (mn, _, _) in enumerate(methods):
        rank_val = ranks[mi]
        ax.plot(rank_val, 0, 'o', ms=14, color=method_colors[mi], zorder=4,
                markeredgecolor='white', markeredgewidth=1.2)
        # Thin vertical leader from dot to label
        ax.plot([rank_val, rank_val], [0.04, label_y[mi] - 0.04],
                color=method_colors[mi], lw=0.7, alpha=0.55, zorder=1, ls=':')
        ax.text(rank_val, label_y[mi], mn, ha='center', fontsize=8,
                fontweight='bold', color=method_colors[mi])

    # CD line below the axis
    ax.plot([1, 1 + CD], [-0.42, -0.42], color='#C0392B', lw=2.2)
    ax.text(1 + CD / 2, -0.60, f'CD = {CD:.2f}', ha='center', color='#C0392B',
            fontsize=8, fontweight='bold')
    ax.set_xlabel('Average Rank')
    ax.set_yticks([])
    ax.set_title('Demšar Critical Difference (k = 4, N = 3, α = 0.05)',
                 fontweight='bold', fontsize=10)
    save('fig8_critical_difference')

# ═══════════════════════════════════════════════════════════════
# Fig9 — Bootstrap Confidence Intervals (pairwise differences, 2000 resamples)
# ═══════════════════════════════════════════════════════════════
def fig9():
    """Fig9 - Bootstrap confidence intervals (pairwise differences, 2000 resamples)."""
    methods = [('CSP+LDA', 'csp', TRAD), ('Spectral+LGB', 'lgb', TRAD),
               ('EEGNet', 'eegnet', DL_MIX), ('EEG-Conformer', 'eegconformer', DL_MIX)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    rng = np.random.RandomState(42)
    labels, means, lows, ups = [], [], [], []
    for i, j in pairs:
        v1 = _per_finger_f1(methods[i][1], methods[i][2])
        v2 = _per_finger_f1(methods[j][1], methods[j][2])
        diff = v1 - v2
        boots = []
        for _ in range(2000):
            idx = rng.randint(0, len(diff), len(diff))
            boots.append(np.mean(diff[idx]))
        boots = np.array(boots)
        labels.append(f'{methods[i][0]} − {methods[j][0]}')
        means.append(np.mean(diff))
        lows.append(np.percentile(boots, 2.5))
        ups.append(np.percentile(boots, 97.5))

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.90, bottom=0.14)
    y = np.arange(len(pairs))
    for yi in y:
        ax.plot([lows[yi], ups[yi]], [yi, yi], color='#7F8C8D', lw=1.6)
        ax.plot(means[yi], yi, 'o', color='#C0392B', ms=8)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Macro F1 difference (95% bootstrap CI, 2000 resamples)')
    ax.set_title('Bootstrap Confidence Intervals — Pairwise Differences',
                 fontweight='bold', fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    save('fig9_bootstrap_ci')

# ═══════════════════════════════════════════════════════════════
fig1()
fig2()
fig3()
fig4()
fig5()
figS1()
figS2()
figS3()
figS4()
figS5()
fig7()
fig8()
fig9()
print('Done — 13 figures generated.')
