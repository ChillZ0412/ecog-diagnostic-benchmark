"""Diagnostic: per-channel kurtosis & RMS (train vs test) for all subjects.

Purpose: resolve the bad-electrode discrepancy between regression (DTCNet) and
classification QC. Specifically:
  * sub3 ch50 (1-based) is a physical bad channel that classification QC missed.
  * sub3 ch28/39/42/55 (1-based) were excluded by kurtosis threshold — need to
    verify whether these are true outliers or threshold false positives.

Outputs per channel: train kurtosis (fisher), test kurtosis, train RMS, test RMS,
and the current QC decision under (median + 10*MAD) kurtosis and (median/10) RMS rules.
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy import stats

DATA_DIR = os.environ.get("BCI_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
FILES = {
    "sub1": "sub1_comp.mat",
    "sub2": "sub2_comp.mat",
    "sub3": "sub3_comp.mat",
}

def analyze(subject):
    path = os.path.join(DATA_DIR, FILES[subject])
    mat = loadmat(path)
    train = mat["train_data"].T.astype(np.float32)  # (n_ch, 400000)
    test = mat["test_data"].T.astype(np.float32)    # (n_ch, 200000)
    n_ch = train.shape[0]

    tr_kurt = stats.kurtosis(train, axis=1, fisher=True)
    te_kurt = stats.kurtosis(test, axis=1, fisher=True)
    tr_rms = np.sqrt((train ** 2).mean(axis=1))
    te_rms = np.sqrt((test ** 2).mean(axis=1))

    # current QC rules (train-only)
    med_k = np.median(tr_kurt)
    mad_k = np.median(np.abs(tr_kurt - med_k)) + 1e-8
    kurt_thr = med_k + 10 * mad_k
    med_rms = np.median(tr_rms)
    dead_thr = med_rms / 10

    bad_kurt = set(int(c) for c in np.where(tr_kurt > kurt_thr)[0])
    bad_dead = set(int(c) for c in np.where(tr_rms < dead_thr)[0])
    bad = sorted(bad_kurt | bad_dead)

    print(f"\n===== {subject}  (n_ch={n_ch}) =====")
    print(f"kurtosis: median={med_k:.2f} MAD={mad_k:.2f} threshold(median+10MAD)={kurt_thr:.2f}")
    print(f"RMS     : median={med_rms:.4f} dead_threshold(median/10)={dead_thr:.4f}")
    print(f"current QC bad channels (1-based) = {[c+1 for c in bad]}")

    # full table, sorted by train kurtosis descending
    order = np.argsort(-tr_kurt)
    print(f"\n{'ch(1b)':>7} {'tr_kurt':>10} {'te_kurt':>10} {'kurt_flag':>10} {'tr_rms':>10} {'te_rms':>10} {'rms_flag':>10}")
    for c in order:
        kf = "KURT" if c in bad_kurt else ""
        rf = "DEAD" if c in bad_dead else ""
        print(f"{c+1:>7} {tr_kurt[c]:>10.2f} {te_kurt[c]:>10.2f} {kf:>10} {tr_rms[c]:>10.4f} {te_rms[c]:>10.4f} {rf:>10}")

    # test-set-only anomalies (what QC missed because it uses train-only)
    med_k_te = np.median(te_kurt)
    mad_k_te = np.median(np.abs(te_kurt - med_k_te)) + 1e-8
    te_kurt_thr = med_k_te + 10 * mad_k_te
    te_only = [int(c) for c in np.where(te_kurt > te_kurt_thr)[0]]
    print(f"\ntest-only kurtosis outliers (1-based, threshold={te_kurt_thr:.2f}): {[c+1 for c in te_only]}")
    # ratio test/train RMS
    ratio = te_rms / (tr_rms + 1e-8)
    print(f"test/train RMS ratio extremes: max={ratio.max():.1f}@ch{int(np.argmax(ratio))+1}, min={ratio.min():.2f}@ch{int(np.argmin(ratio))+1}")
    big = sorted(int(c) for c in np.where(ratio > 50)[0])
    print(f"channels with test/train RMS ratio > 50 (1-based): {[c+1 for c in big]}")
    return

for s in ["sub1", "sub2", "sub3"]:
    analyze(s)
