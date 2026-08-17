# ECoG Finger-Movement Classification Benchmark

A diagnostic benchmark for ECoG-based finger-movement classification on
BCI Competition IV, Dataset 4 (3 subjects, 62/48/64 channels, 400 s train /
200 s test per subject). Four methods — CSP+LDA, Spectral+LightGBM, EEGNet,
and EEG-Conformer — are evaluated under a single unified protocol.

## Environment

- Python 3.13
- numpy, scipy, scikit-learn, lightgbm, mne, matplotlib
- PyTorch 2.x (for EEGNet / EEG-Conformer)
- braindecode 1.3.2 (optional, DL utilities)

## Data

The competition data are downloaded automatically from the BCI Competition IV
website on first run. Set `BCI_DATA_DIR` to control the cache location
(default: `./data/`). The official test labels
(`true_labels.zip`, sub1/sub2/sub3_testlabels.mat) must be placed under
`BCI_DATA_DIR/test_labels/`.

## Pipeline

1. **Preprocess** (optional — the benchmark re-derives features on the fly):
   ```
   python preprocess_raw.py
   ```

2. **Main benchmark** (traditional + deep-learning methods):
   ```
   python run_benchmark.py            # CSP+LDA, Spectral+LGB (5 seeds)
   python run_trial_dl.py             # EEGNet, EEG-Conformer (3 seeds)
   ```

3. **Ablations**:
   ```
   python run_ablations.py --ablation all              # frequency band, window, classifier
   python run_dl_ablations.py --ablation all           # Mixup, SSL
   python run_label_alignment_ablation.py --subject sub1   # label alignment
   ```

4. **Statistics**:
   ```
   python compute_combined_stats.py    # Friedman, per-finger Wilcoxon, Cohen's d
   ```

5. **Figures** (13 publication figures, PNG + PDF at 300 dpi):
   ```
   python figures_final.py
   ```

## Electrode quality control

Three absolute thresholds are applied on training data only (no test leakage):
spiky channels (kurtosis > 50), dead channels (RMS < median/10), and
test-time breakage (test/train RMS ratio > 10x). This removes
sub1 ch55, sub2 ch21+ch38, and sub3 ch50 (61/46/63 channels remain), matching
the regression task.

## Label alignment

Dense sliding-window labels use a full-window majority vote over the 25 Hz
discrete labels (Rest class included), matching the DL trial-extraction rule.
Onset alignment is shown to underestimate traditional-method macro-F1 by
~0.11 (see `run_label_alignment_ablation.py`).

## Code lock

All 15 core files are checksummed in `.code_lock.sha256` (V18). The commit
hash pins the exact source that produced the reported results.

## Results

| Method | S1 F1 | S2 F1 | S3 F1 | Mean F1 | BalAcc | Cohen's κ |
|---|---|---|---|---|---|---|
| CSP+LDA | 0.432 | 0.271 | 0.601 | 0.435 | 0.435 | 0.343 |
| Spectral+LGB | 0.450 | 0.210 | 0.565 | 0.409 | 0.416 | 0.336 |
| EEGNet | 0.225 | 0.281 | 0.240 | 0.249 | 0.269 | 0.143 |
| EEG-Conformer | 0.258 | 0.199 | 0.244 | 0.234 | 0.281 | 0.140 |

Traditional methods significantly outperform deep learning
(per-finger paired Wilcoxon, n = 15: p = 0.0054; Cohen's d = 1.48,
subject-level pooled). Subject-level Friedman test: p = 0.334 (n = 3,
underpowered).
