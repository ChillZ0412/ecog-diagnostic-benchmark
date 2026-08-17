"""Shared utilities for DL experiments (trial extraction, preprocessing)."""

import numpy as np
from scipy.signal import decimate

from .data_loader import generate_labels

# ── Signal Processing Constants ──
ECOG_FS_ORIG = 1000      # Original ECoG sampling rate (Hz)
DG_FS = 25                # Data glove sampling rate (Hz)
DL_TARGET_FS = 500        # DL downsampled rate (Hz)
TRIAL_MS = 1000           # Trial window (ms)
TRIAL_SAMPLES = int(TRIAL_MS * DL_TARGET_FS / 1000)   # 500 samples @500Hz
DG_SAMPLES_PER_TRIAL = int(TRIAL_MS * DG_FS / 1000)    # 25 DG samples
DS_RATIO = ECOG_FS_ORIG // DL_TARGET_FS                 # 2


def downsample_ecog(ecog):
    """IIR decimate ECoG from 1000Hz to 500Hz.

    Parameters
    ----------
    ecog : ndarray (n_ch, n_samples_1000hz)
        Original ECoG at 1000 Hz

    Returns
    -------
    ndarray (n_ch, n_samples_500hz)
    """
    n_orig = ecog.shape[1]
    n_new = n_orig // DS_RATIO
    out = np.zeros((ecog.shape[0], n_new), dtype=np.float32)
    for ch in range(ecog.shape[0]):
        out[ch] = decimate(ecog[ch], DS_RATIO, ftype='iir').astype(np.float32)
    return out


def extract_trials(ecog_500, dg_train, seed=None, n_rest_multiplier=1):
    """Extract discrete trials from continuous ECoG + DG data.

    Movement trials: extracted at each transition into a movement state
    (Rest→finger and finger→finger), with a minimum spacing of one trial
    length to avoid overlap. Labels use a full-window majority vote, matching
    ``extract_test_trials``.
    Rest trials: randomly sampled from long rest periods.

    Parameters
    ----------
    ecog_500 : ndarray (n_ch, n_samples_500hz)
        Downsampled ECoG
    dg_train : ndarray (n_samples_25hz, 5)
        Data glove at 25 Hz
    seed : int or None
        For reproducible rest-trial sampling
    n_rest_multiplier : int
        Upper-bound multiplier for rest trials per movement trial

    Returns
    -------
    X : ndarray (n_trials, n_ch, trial_samples)
    y : ndarray (n_trials,)
    """
    rng = np.random.RandomState(seed)
    labels_dg = generate_labels(dg_train)

    # ── Movement trials ──
    # Detect every transition INTO a movement state — both Rest→finger AND
    # finger→finger — so consecutive finger switches that never return to Rest
    # are captured (previously only Rest→finger onsets were detected, which
    # systematically dropped the later fingers of a continuous movement chain).
    # A minimum spacing of one trial length avoids overlapping training trials.
    change_points = np.where(np.diff(labels_dg) != 0)[0] + 1
    onsets = [int(cp) for cp in change_points if labels_dg[cp] != 0]

    movement_trials = []
    last_onset_dg = -DG_SAMPLES_PER_TRIAL
    for onset_dg in onsets:
        if onset_dg - last_onset_dg < DG_SAMPLES_PER_TRIAL:
            continue  # skip to avoid overlapping the previous trial
        onset_ecog = onset_dg * (ECOG_FS_ORIG // DG_FS) // DS_RATIO
        end_ecog = onset_ecog + TRIAL_SAMPLES
        if end_ecog > ecog_500.shape[1]:
            continue
        trial = ecog_500[:, onset_ecog:end_ecog]
        # Full-window majority vote — identical label rule to extract_test_trials
        seg_labels = labels_dg[onset_dg:onset_dg + DG_SAMPLES_PER_TRIAL]
        counts = np.bincount(seg_labels, minlength=7)[1:]  # movement fingers only
        if counts.max() == 0:
            continue  # window dominated by Rest → not a usable movement trial
        dominant_class = counts.argmax() + 1
        movement_trials.append((trial, dominant_class))
        last_onset_dg = onset_dg

    # ── Rest trials ──
    rest_periods = []
    in_rest = False
    rest_start = 0
    for i in range(len(labels_dg)):
        if not in_rest and labels_dg[i] == 0:
            rest_start = i
            in_rest = True
        elif in_rest and labels_dg[i] != 0:
            length = i - rest_start
            if length >= 2 * DG_SAMPLES_PER_TRIAL:
                rest_periods.append((rest_start, i))
            in_rest = False
    if in_rest and (len(labels_dg) - rest_start) >= 2 * DG_SAMPLES_PER_TRIAL:
        rest_periods.append((rest_start, len(labels_dg)))

    n_rest_target = max(len(movement_trials) * n_rest_multiplier, len(movement_trials) // 2)
    rest_trials = []
    rest_samples_per_period = max(1, n_rest_target // max(len(rest_periods), 1))

    for period_start, period_end in rest_periods:
        for _ in range(min(rest_samples_per_period, (period_end - period_start) // DG_SAMPLES_PER_TRIAL)):
            dg_idx = rng.randint(period_start, period_end - DG_SAMPLES_PER_TRIAL)
            ecog_start = dg_idx * (ECOG_FS_ORIG // DG_FS) // DS_RATIO
            ecog_end = ecog_start + TRIAL_SAMPLES
            if ecog_end > ecog_500.shape[1]:
                continue
            trial = ecog_500[:, ecog_start:ecog_end]
            rest_trials.append((trial, 0))

    rng.shuffle(rest_trials)
    rest_trials = rest_trials[:min(len(rest_trials), n_rest_target)]

    # ── Combine and shuffle ──
    all_trials = movement_trials + rest_trials
    if not all_trials:
        raise ValueError("No trials extracted!")

    X = np.stack([t[0] for t in all_trials])
    y = np.array([t[1] for t in all_trials])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def extract_test_trials(ecog_500, dg_test, train_max=None):
    """Extract test trials as non-overlapping windows (no shuffling).

    Parameters
    ----------
    ecog_500 : ndarray (n_ch, n_samples_500hz)
    dg_test  : ndarray (n_samples_25hz, 5)
    train_max : ndarray, shape (5,), optional
        Per-finger maxima from TRAIN data; passed to ``generate_labels`` so the
        test threshold is identical to the training threshold.

    Returns
    -------
    X : ndarray (n_trials, n_ch, trial_samples)
    y : ndarray (n_trials,)
    """
    labels_dg = generate_labels(dg_test, max_angles=train_max)

    trials, trial_labels = [], []
    for i in range(0, len(labels_dg), DG_SAMPLES_PER_TRIAL):
        ecog_start = i * (ECOG_FS_ORIG // DG_FS) // DS_RATIO
        ecog_end = ecog_start + TRIAL_SAMPLES
        if ecog_end > ecog_500.shape[1]:
            break
        trial = ecog_500[:, ecog_start:ecog_end]
        segment_labels = labels_dg[i:i + DG_SAMPLES_PER_TRIAL]
        label = np.bincount(segment_labels, minlength=7).argmax()
        trials.append(trial)
        trial_labels.append(label)

    return np.stack(trials), np.array(trial_labels)
