# Wiener Filter — ECoG Finger Movement Regression

Reproduction of a Wiener filter decoder for continuous finger-flexion regression from ECoG signals, following Liang & Bougrain (2012), evaluated as part of a diagnostic benchmark comparing four regression methods (classical linear → deep learning) on BCI Competition IV, Dataset 4.

This method is not evaluated as a standalone "best score" candidate — it operationalizes a specific hypothesis: **does a linear, time-invariant mapping suffice to decode continuous finger trajectories from ECoG?**

## Method Overview

A closed-form Wiener filter maps band-specific amplitude-modulation (AM) features, stacked over a short time-lag window, to continuous finger-flexion trajectories via pseudo-inverse linear regression.

### Configuration

| Component | Setting |
|---|---|
| Frequency bands | Three-band FIR decomposition: sub (1–60 Hz), gamma (60–100 Hz), fast-gamma (100–200 Hz) |
| Feature extraction | Band-specific amplitude-modulation (AM) features, 40 ms non-overlapping windows, aligned to the 25 Hz dataglove sampling rate |
| Temporal context | k=25-lag memory stack (~1 s of history) |
| Feature selection | Greedy forward selection, ≤10 channel×band pairs, internal 3/5 train + 2/5 validation split |
| Regression | Closed-form pseudo-inverse Wiener regression (`pinv(XᵀX)`) |
| Parameters | ~261 |

### Dataset & Evaluation Protocol

- **Dataset**: BCI Competition IV, Dataset 4 (Miller & Schalk, 2008) — 3 subjects, official 400 s train / 200 s test split
- **Evaluation**: Within-subject only (S1, S2, S3 evaluated separately; no cross-subject pooling — see paper for rationale)
- **Bad-channel exclusion**: Subject-specific faulty electrodes removed prior to feature extraction (see `Results` below; verified independently against 3 other methods in the benchmark)
- **Metrics**: official_r (Pearson correlation, 4-finger mean excluding ring finger, per official competition scoring), official_R² (variance explained vs. predicting the mean), calibration_gap, MAE — see paper Methods §2.3 for full metric definitions

## Repository Structure

```
.
├── preprocess.py           # Band decomposition + AM feature extraction
├── feature_selection.py    # Greedy forward feature selection
├── wiener_regression.py    # Closed-form pseudo-inverse regression + training
├── evaluate.py              # Metric computation (r / R² / calibration_gap / MAE)
├── run_wiener.py            # End-to-end pipeline entry point
├── results/
│   ├── benchmark_r.csv              # Per-finger, per-subject Pearson r
│   ├── benchmark_r2.csv             # Per-finger, per-subject R²
│   ├── benchmark_mae.csv            # Per-finger, per-subject MAE
│   └── summary.json                 # official_r / official_R² / MAE, aggregated
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_wiener.py --subject S1 --data-root /path/to/bci_competition_iv_ds4
```

Outputs per-subject prediction traces and metric summaries to `results/`.

## Results

| Metric | Value (mean across subjects) |
|---|---|
| official_r | 0.460 |
| official_R² | 0.054 |
| MAE | 0.603 |

| Subject | official_r |
|---|---|
| S1 | 0.45 |
| S2 | 0.32 |
| S3 | 0.61 |

### Electrode Quality Control

An anomalously large train/test amplitude discrepancy (~700,000×) was identified on a single Subject 3 electrode and excluded prior to feature extraction. This finding was independently corroborated by three other methods in the benchmark (Switching Linear Model, FingerFlex, DTCNet) and by an external published study (FBTTR, arXiv:2412.06815).

## Citation

If you use this reproduction, please cite the original method and the dataset:

```
Liang, N. & Bougrain, L. (2012). Decoding finger flexion from band-specific
ECoG signals. Frontiers in Neuroscience, 6, 91.

Miller, K.J. & Schalk, G. (2008). BCI Competition IV, Dataset 4.
```

## License

[Add your license here — e.g., MIT]
