"""
Data loading and label generation for BCI Competition IV Dataset 4.

The dataset provides continuous ECoG recordings (~400 s train, ~200 s test)
and simultaneous data-glove recordings (5-finger flexion angles).  Labels are
derived by thresholding the glove signal at 10% of each finger's maximum angle
following Yao & Shoaran (2019).
"""

import os
import urllib.request
import zipfile
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import medfilt

from .config import (
    ANGLE_THRESHOLD_RATIO,
    DATA_DIR,
    DATA_FILES,
    MEDIAN_FILTER_KERNEL,
    SAMPLES_PER_GLOVE,
    TEST_LABEL_FILES,
)


# ─────────────────────────────────────────────────────────────────────
# Dataset download
# ─────────────────────────────────────────────────────────────────────

def download_dataset() -> None:
    """Download BCI Competition IV Dataset 4 (~200 MB) if not present."""
    # Check actual data files (not a fictional calibration file)
    if all(os.path.exists(p) for p in DATA_FILES.values()):
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    url = "https://www.bbci.de/competition/download/competition_iv/BCICIV_4_mat.zip"
    zip_path = os.path.join(DATA_DIR, "BCICIV_4_mat.zip")

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Done.")


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def load_subject(subject_id: str) -> Dict[str, np.ndarray]:
    """Load train AND test data for one subject (official BCI Competition IV split).

    sub*_comp.mat provides:
        train_data  — (400000, n_ch) ECoG, 400 s, 1000 Hz
        test_data   — (200000, n_ch) ECoG, 200 s, 1000 Hz
        train_dg    — (400000, 5) data-glove angles (train only)

    sub*_testlabels.mat provides:
        test_dg     — (200000, 5) data-glove angles (test, released post-competition)

    Parameters
    ----------
    subject_id : str
        One of {"sub1", "sub2", "sub3"}.

    Returns
    -------
    dict
        "train_ecog"   : (n_channels, 400000) — training ECoG
        "test_ecog"    : (n_channels, 200000) — test ECoG
        "train_dg"     : (10000, 5) — training data glove at 25 Hz
        "test_dg"      : (5000, 5)  — test data glove at 25 Hz
        "n_channels"   : int — number of ECoG channels

    Raises
    ------
    FileNotFoundError
        If any required .mat file does not exist.
    """
    path = DATA_FILES.get(subject_id)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file for {subject_id} not found at {path}. "
            f"Run download_dataset() first."
        )

    # ── Load competition data ──
    mat = loadmat(path)
    train_ecog = mat["train_data"].T.astype(np.float32)    # → (n_ch, 400000)
    test_ecog  = mat["test_data"].T.astype(np.float32)     # → (n_ch, 200000)
    train_dg_raw = mat["train_dg"].astype(np.float32)       # → (400000, 5)

    # Downsample train data glove to 25 Hz
    train_dg = train_dg_raw[::SAMPLES_PER_GLOVE].copy()     # → (10000, 5)

    # ── Load test labels (ground truth, released after competition) ──
    test_label_path = TEST_LABEL_FILES.get(subject_id)
    if test_label_path is None or not os.path.exists(test_label_path):
        raise FileNotFoundError(
            f"Test labels for {subject_id} not found at {test_label_path}. "
            f"Download from https://www.bbci.de/competition/iv/results/ds4/true_labels.zip"
        )
    test_mat = loadmat(test_label_path)
    test_dg_raw = test_mat["test_dg"].astype(np.float32)    # → (200000, 5)
    test_dg = test_dg_raw[::SAMPLES_PER_GLOVE].copy()       # → (5000, 5)

    # ── Electrode Quality Control (on RAW ECoG, BEFORE z-score) ──
    # Identify PHYSICALLY BROKEN electrodes only. QC uses raw ECoG amplitude /
    # kurtosis — never labels — so it cannot leak test information into training.
    # Three criteria:
    #   (1) Dead/flat channel: train RMS < 10% of median train RMS.
    #   (2) Catastrophic spike (train): excess kurtosis (fisher) above an
    #       absolute floor. The earlier relative `median + 10·MAD` threshold is
    #       subject-dependent: a subject with a tight kurtosis spread gets an
    #       artificially low cutoff and mislabels mildly-spiky-but-functional
    #       channels as bad. True broken electrodes show kurtosis ≳ 10², versus
    #       0–9 for physiological channels (a ≳ 18× gap, so any floor in
    #       [20, 100] selects the identical set).
    #   (3) Broken in TEST only: test/train RMS ratio above an absolute floor.
    #       A channel that is clean during training but fails during test (e.g.
    #       a loose connection) is invisible to a train-only QC, yet must still
    #       be dropped (whole channel, train+test) because a model cannot ingest
    #       inconsistent channel counts.
    from scipy import stats as _stats
    KURT_ABS_THRESHOLD = 50.0      # excess kurtosis (fisher); true failures ≳ 10²
    RMS_RATIO_THRESHOLD = 10.0     # test/train RMS ratio; catches test-only failures

    train_kurt_raw = _stats.kurtosis(train_ecog, axis=1, fisher=True)
    train_rms_raw = np.sqrt((train_ecog ** 2).mean(axis=1))
    test_rms_raw = np.sqrt((test_ecog ** 2).mean(axis=1))

    median_rms = np.median(train_rms_raw)
    dead_threshold = median_rms / 10
    rms_ratio = test_rms_raw / (train_rms_raw + 1e-8)

    bad_channels = set(
        int(c) for c in np.where(train_rms_raw < dead_threshold)[0]         # (1) dead/flat
    ) | set(
        int(c) for c in np.where(train_kurt_raw > KURT_ABS_THRESHOLD)[0]    # (2) catastrophic spike
    ) | set(
        int(c) for c in np.where(rms_ratio > RMS_RATIO_THRESHOLD)[0]        # (3) broken in test
    )
    bad_channels = sorted(bad_channels)
    keep_channels = [c for c in range(train_ecog.shape[0]) if c not in bad_channels]
    train_ecog = train_ecog[keep_channels]
    test_ecog = test_ecog[keep_channels]
    if bad_channels:
        print(f"  [load_subject] {subject_id}: excluded {len(bad_channels)} bad electrode(s): "
              f"{[c + 1 for c in bad_channels]} (kurt>{KURT_ABS_THRESHOLD:.0f} or "
              f"RMS<{dead_threshold:.2f} or test/train-RMS>{RMS_RATIO_THRESHOLD:.0f}x)")

    # ── Per-channel z-score normalization on ECoG ──
    # BC2000 acquisition hardware uses different gain settings across sessions;
    # sub3 test ECoG is ~7x larger than train. Z-scoring per channel normalizes
    # this to unit variance, making train/test distributions comparable.
    # Standard practice in ECoG motor decoding (Liang & Bougrain 2012; Miller 2019).
    # IMPORTANT: fit statistics on TRAIN only; apply the SAME stats to test.
    train_mean = train_ecog.mean(axis=1, keepdims=True)     # (n_ch, 1)
    train_std  = train_ecog.std(axis=1, keepdims=True)      # (n_ch, 1)
    train_ecog = (train_ecog - train_mean) / (train_std + 1e-8)
    test_ecog  = (test_ecog - train_mean) / (train_std + 1e-8)

    # ── Data glove: kept in RAW angle units ──
    # Labels are derived by thresholding at 10% of each finger's maximum angle
    # (Yao & Shoaran, 2019). Z-scoring is NOT applied here: the subtract-mean term
    # is not a scale-only transform and would shift the threshold semantics.
    # Train/test share per-finger maxima computed on TRAIN data at label-
    # generation time (see generate_labels).

    return {
        "train_ecog": train_ecog,
        "test_ecog": test_ecog,
        "train_dg": train_dg,      # raw finger angles (25 Hz)
        "test_dg": test_dg,        # raw finger angles (25 Hz)
        "n_channels": train_ecog.shape[0],
    }


# ─────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────

def generate_labels(
    dg: np.ndarray,
    threshold_ratio: float = ANGLE_THRESHOLD_RATIO,
    median_kernel: int = MEDIAN_FILTER_KERNEL,
    min_duration: Optional[int] = None,
    max_angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert continuous finger angles to 6-class discrete labels.

    Procedure (Yao & Shoaran, 2019):
        1. Compute per-finger threshold = 10% × that finger's maximum angle
           (train maxima; see ``max_angles`` for the train/test-shared threshold).
        2. At each time step, a finger is "active" if its angle exceeds threshold.
        3. If 0 fingers active → class 0 (Rest).
           If 1 finger  active → that finger's class (1–5).
           If >1 fingers active → pick the one with the largest angle.
        4. Median-filter to remove jitter.

    Parameters
    ----------
    dg : np.ndarray, shape (n_time, 5)
        Data-glove angles at 25 Hz (downsampled).
    threshold_ratio : float
        Fraction of max angle used as threshold (default: 0.10).
    median_kernel : int
        Median filter kernel size in samples (25 Hz → 200 ms for kernel=5).
    min_duration : int, optional
        Minimum active duration in samples; transitions shorter than this
        are suppressed.  DEPRECATED — median filter handles this.
    max_angles : np.ndarray, shape (5,), optional
        Per-finger maximum angles used to set the threshold. Defaults to the
        input signal's own maxima. Pass the TRAIN-data maxima so that train and
        test share an identical threshold (avoids a test-dependent threshold).

    Returns
    -------
    labels : np.ndarray, shape (n_time,), dtype=int
        Integer labels: 0 = Rest, 1 = Thumb, 2 = Index, 3 = Middle,
        4 = Ring, 5 = Little.
    """
    n_time = len(dg)
    if max_angles is None:
        max_angles = dg.max(axis=0)                 # (5,)
    thresholds = max_angles * threshold_ratio       # (5,)

    # Determine active fingers per time step
    active_mask = dg > thresholds[np.newaxis, :]    # (n_time, 5)
    n_active = active_mask.sum(axis=1)              # (n_time,)

    # Initialize labels
    labels = np.zeros(n_time, dtype=np.int32)

    # Single-finger active → direct assignment
    single_mask = n_active == 1
    labels[single_mask] = np.argmax(active_mask[single_mask], axis=1) + 1

    # Multiple fingers active → pick the one with largest angle
    multi_mask = n_active > 1
    if multi_mask.any():
        labels[multi_mask] = np.argmax(dg[multi_mask], axis=1) + 1

    # Median filter to suppress transient misclassifications
    labels = medfilt(labels, kernel_size=median_kernel).astype(np.int32)

    return labels


def window_majority_labels(
    labels: np.ndarray,
    win_samples: int,
    step_samples: int,
) -> np.ndarray:
    """Majority-vote labels for dense sliding windows (full-window rule).

    Replaces the earlier *onset-aligned* labelling (``labels[:n_windows]``),
    which assigned each 500 ms window the label of its first sample only and
    therefore ignored the trailing ~450 ms of signal. The full-window majority
    vote is the same rule used by the DL trial extractors
    (``extract_trials`` / ``extract_test_trials``), so traditional and deep
    methods now share an identical label definition.

    A window starting at ECoG sample ``wi * step_samples`` spans
    ``[wi * step_samples, wi * step_samples + win_samples]``. At 25 Hz (one
    label per ``step_samples`` ECoG samples) this maps to label indices
    ``[wi, wi + ceil(win_samples / step_samples))``, over which the majority
    class (including Rest = 0) is taken.

    Parameters
    ----------
    labels : np.ndarray, shape (n_time_25hz,)
        Integer labels at 25 Hz (0 = Rest, 1–5 = fingers).
    win_samples : int
        Window length in ECoG samples (1000 Hz).
    step_samples : int
        Window stride in ECoG samples (1000 Hz); equals one 25-Hz label step.

    Returns
    -------
    maj : np.ndarray, shape (n_time_25hz - n_per_window + 1,)
        Majority-vote label for the window starting at each label index. Callers
        slice ``maj[:n_windows]`` to match the number of ECoG windows.
    """
    n_per_window = int(np.ceil(win_samples / step_samples))
    n_windows = len(labels) - n_per_window + 1
    maj = np.zeros(n_windows, dtype=np.int32)
    for wi in range(n_windows):
        seg = labels[wi:wi + n_per_window]
        maj[wi] = np.bincount(seg, minlength=7).argmax()
    return maj


def get_label_distribution(labels: np.ndarray, names: Optional[list] = None) -> Dict[str, Tuple[int, float]]:
    """Return label distribution as {name: (count, percentage)}."""
    if names is None:
        from .config import FINGER_NAMES
        names = FINGER_NAMES
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    return {names[i]: (int(c), 100 * c / total) for i, c in zip(unique, counts)}
