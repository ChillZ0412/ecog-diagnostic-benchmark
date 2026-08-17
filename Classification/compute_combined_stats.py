"""Combined statistics: traditional + DL methods, after ch50 QC fix.

Loads the latest traditional and DL JSONs, then computes:
  1. 4-method table (per-subject Macro F1, mean across seeds)
  2. Per-finger F1 (movement fingers, averaged across seeds)
  3. Friedman (4 methods, subject-level Macro F1, N=3)
  4. Wilcoxon: traditional-family vs DL-family (per-finger x subject, n=15)
  5. Pairwise Wilcoxon (6 pairs, per-finger n=15) + Cohen's d
"""
import json, glob, os
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))

def latest(pattern):
    fs = sorted(glob.glob(os.path.join(ROOT, 'results', pattern)))
    return fs[-1] if fs else None

TRAD = latest('benchmark_csp_lgb_*.json')
DL   = latest('trial_dl_*.json')
print('TRAD:', os.path.basename(TRAD))
print('DL  :', os.path.basename(DL))

trad = json.load(open(TRAD))
dl = json.load(open(DL))

SUBJECTS = ['sub1', 'sub2', 'sub3']
FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

def method_f1(data, mk):
    """Per-subject mean macro F1 (across seeds)."""
    out = {}
    for s in SUBJECTS:
        vals = [r['results'][mk]['macro_f1'] for r in data['results']
                if r['subject'] == s and mk in r.get('results', {})]
        out[s] = float(np.mean(vals))
    return out

def method_per_finger(data, mk):
    """Per (subject, finger) mean F1, averaged across seeds -> 15-vector."""
    out = []
    for s in SUBJECTS:
        for f in FINGERS:
            vals = []
            for r in data['results']:
                if r['subject'] == s and mk in r.get('results', {}):
                    pc = r['results'][mk].get('per_class', {})
                    if f in pc:
                        vals.append(pc[f]['f1'])
            out.append(float(np.mean(vals)))
    return np.array(out)

methods = [('CSP+LDA', 'csp', trad), ('Spectral+LGB', 'lgb', trad),
           ('EEGNet', 'eegnet', dl), ('EEG-Conformer', 'eegconformer', dl)]

print('\n=== 4-method Macro F1 (mean across seeds) ===')
print(f"{'Method':<16} {'sub1':>8} {'sub2':>8} {'sub3':>8} {'Mean':>8}")
mat = np.zeros((4, 3))
for mi, (mn, mk, data) in enumerate(methods):
    f1 = method_f1(data, mk)
    mat[mi] = [f1[s] for s in SUBJECTS]
    print(f"{mn:<16} {f1['sub1']:>8.3f} {f1['sub2']:>8.3f} {f1['sub3']:>8.3f} {np.mean([f1[s] for s in SUBJECTS]):>8.3f}")

print('\n=== Friedman (4 methods, subject-level Macro F1, N=3) ===')
stat, p = friedmanchisquare(mat[0], mat[1], mat[2], mat[3])
print(f'Friedman chi2={stat:.3f}, p={p:.4f}')

# per-finger vectors
pf = {mn: method_per_finger(data, mk) for mn, mk, data in methods}
print('\n=== per-finger F1 (15-vector) ===')
for mn in pf:
    print(f'{mn:<16} mean={pf[mn].mean():.3f}  {np.round(pf[mn],3)}')

# Traditional-family vs DL-family (pooled by MEAN of the 2 methods per family)
trad_vec = (pf['CSP+LDA'] + pf['Spectral+LGB']) / 2
dl_vec = (pf['EEGNet'] + pf['EEG-Conformer']) / 2
print('\n=== Wilcoxon: traditional-family vs DL-family (n=15) ===')
_, p_td = wilcoxon(trad_vec, dl_vec)
diff = trad_vec - dl_vec
d_td = np.mean(diff) / (np.std(diff, ddof=1) + 1e-9)
print(f'trad mean={trad_vec.mean():.3f}, DL mean={dl_vec.mean():.3f}, p={p_td:.4f}, Cohen d={d_td:+.3f}')

print('\n=== Pairwise Wilcoxon (per-finger n=15) + Cohen d ===')
keys = list(pf.keys())
for i in range(4):
    for j in range(i+1, 4):
        v1, v2 = pf[keys[i]], pf[keys[j]]
        _, p = wilcoxon(v1, v2)
        dd = v1 - v2
        d = np.mean(dd) / (np.std(dd, ddof=1) + 1e-9)
        sig = '*' if p < 0.05 else 'ns'
        print(f'{keys[i]:<16} vs {keys[j]:<16} p={p:.4f} ({sig})  d={d:+.3f}')

# Also: best-traditional vs best-DL (max pooling) as sensitivity
trad_max = np.maximum(pf['CSP+LDA'], pf['Spectral+LGB'])
dl_max = np.maximum(pf['EEGNet'], pf['EEG-Conformer'])
_, p_max = wilcoxon(trad_max, dl_max)
print(f'\n[sensitivity] best-trad vs best-DL (max-pool): p={p_max:.4f}')
