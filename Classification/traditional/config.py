"""
Configuration constants and hyperparameter dataclasses.

All paths, parameters, and experimental settings are centralized here
to ensure reproducibility and eliminate hardcoded values.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# Data paths — set BCI_DATA_DIR env var to override, or use ./data/
# ─────────────────────────────────────────────────────────────────────

DATA_DIR: str = os.environ.get("BCI_DATA_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
TEST_LABELS_DIR: str = os.path.join(DATA_DIR, "test_labels")
DATA_FILES: Dict[str, str] = {
    "sub1": os.path.join(DATA_DIR, "sub1_comp.mat"),
    "sub2": os.path.join(DATA_DIR, "sub2_comp.mat"),
    "sub3": os.path.join(DATA_DIR, "sub3_comp.mat"),
}
TEST_LABEL_FILES: Dict[str, str] = {
    "sub1": os.path.join(TEST_LABELS_DIR, "sub1_testlabels.mat"),
    "sub2": os.path.join(TEST_LABELS_DIR, "sub2_testlabels.mat"),
    "sub3": os.path.join(TEST_LABELS_DIR, "sub3_testlabels.mat"),
}
RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "results")

# ─────────────────────────────────────────────────────────────────────
# Dataset constants (BCI Competition IV, Dataset 4)
# ─────────────────────────────────────────────────────────────────────

ECOG_SAMPLING_RATE: int = 1000          # Hz (BCI2000 native)
GLOVE_SAMPLING_RATE: int = 25           # Hz (data glove output rate)
SAMPLES_PER_GLOVE: int = ECOG_SAMPLING_RATE // GLOVE_SAMPLING_RATE  # = 40

SUBJECT_CHANNELS: Dict[str, int] = {"sub1": 62, "sub2": 48, "sub3": 64}
N_FINGERS: int = 5
FINGER_NAMES: List[str] = ["Rest", "Thumb", "Index", "Middle", "Ring", "Little"]

# ─────────────────────────────────────────────────────────────────────
# Label generation
# ─────────────────────────────────────────────────────────────────────

ANGLE_THRESHOLD_RATIO: float = 0.10    # 10% of max angle  (Yao & Shoaran, 2019)
MEDIAN_FILTER_KERNEL: int = 5          # samples at 25 Hz → 200 ms window


@dataclass(frozen=True)
class FeatureConfig:
    """Shared feature extraction parameters."""

    window_length_ms: int = 500         # ECoG window in ms for spectral (LGB) features; longer windows improve PSD resolution
    step_ms: int = 40                   # stride to align with 25 Hz data glove
    fs: int = ECOG_SAMPLING_RATE

    @property
    def window_samples(self) -> int:
        return int(self.window_length_ms * self.fs / 1000)

    @property
    def step_samples(self) -> int:
        return int(self.step_ms * self.fs / 1000)

    # Spectral bands for classification  (Yao and Shoaran, 2019)
    spectral_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "LMP":      (0.3,    4),
        "Alpha":    (8.0,   13),
        "Beta":     (13.0,  30),
        "LowGamma": (30.0,  55),
        "HighG1":   (65.0, 115),
        "HighG2":   (125.0, 175),
    })


@dataclass(frozen=True)
class LightGBMConfig:
    """LightGBM classifier hyperparameters."""

    n_estimators: int = 200
    learning_rate: float = 0.1
    num_leaves: int = 31
    class_weight: str = "balanced"
    random_state: int = 42
    verbose: int = -1


@dataclass(frozen=True)
class SVMConfig:
    """SVM classifier hyperparameters."""

    kernel: str = "rbf"
    C: float = 1.0
    class_weight: str = "balanced"
    random_state: int = 42


@dataclass(frozen=True)
class CSPConfig:
    """CSP + LDA classifier hyperparameters (Blankertz et al., 2008)."""

    # CSP spatial filtering
    n_components: int = 3              # CSP component pairs per OvR binary problem
    freq_band: Tuple[float, float] = (65.0, 175.0)  # High gamma — best cross-subject CSP band
    # LDA classifier
    solver: str = "svd"                # LDA solver (svd, lsqr, eigen)
    shrinkage: Optional[str] = None   # LDA shrinkage (None, "auto", float)
    # Window segmentation
    window_ms: int = 500               # CSP analysis window (ms) — window-length ablation optimum
    step_ms: int = 40                  # stride (ms)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    subjects: List[str] = field(default_factory=lambda: ["sub1", "sub2", "sub3"])
    random_state: int = 42
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lgb: LightGBMConfig = field(default_factory=LightGBMConfig)
    csp: CSPConfig = field(default_factory=CSPConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)      # for SpectralSVM ablation

    @property
    def n_train_windows(self) -> int:
        """Estimated number of windows from training data (~400s at 1000 Hz)."""
        total_samples = 400_000
        return (total_samples - self.features.window_samples) // self.features.step_samples

    @property
    def n_test_windows(self) -> int:
        """Estimated number of windows from test data (~200s at 1000 Hz)."""
        total_samples = 200_000
        return (total_samples - self.features.window_samples) // self.features.step_samples
