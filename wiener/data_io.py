"""
Stage 1 - Data I/O for BCI Competition IV, Data set 4.

Provides:
  * load_subject(n)      -> real data from the .mat files
  * make_synthetic(n)    -> fake data with the SAME structure, for smoke tests

Expected file layout (standard competition release):
  data/sub{n}_comp.mat        keys: train_data, train_dg, test_data
  data/sub{n}_testlabels.mat  keys: test_dg

NOTE on the synthetic glove: the REAL dataglove signal was recorded at 25 Hz
and zero-order-held up to 1000 Hz (measured: within-block std ~ 1e-17). The
generator reproduces that by default (`zoh=True`) so smoke tests exercise the
same structure as the real recording. Pass `zoh=False` for a genuinely
continuous 1000 Hz trace, useful for checking that the Stage 3 downsample
diagnostics actually fire.
"""
from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

import config as C


@dataclass
class SubjectData:
    """One subject's data. All arrays are (time, channels/fingers), fs = 1000 Hz."""
    subject: int
    train_ecog: np.ndarray
    train_glove: np.ndarray
    test_ecog: np.ndarray
    test_glove: np.ndarray

    @property
    def n_channels(self) -> int:
        return self.train_ecog.shape[1]


def load_subject(n: int, data_dir=None) -> SubjectData:
    """Load subject n (1, 2, or 3) from the competition .mat files."""
    data_dir = C.DATA_DIR if data_dir is None else data_dir
    comp = loadmat(str(data_dir / f"sub{n}_comp.mat"))
    labels = loadmat(str(data_dir / f"sub{n}_testlabels.mat"))

    return SubjectData(
        subject=n,
        train_ecog=np.asarray(comp["train_data"], dtype=np.float64),
        train_glove=np.asarray(comp["train_dg"], dtype=np.float64),
        test_ecog=np.asarray(comp["test_data"], dtype=np.float64),
        test_glove=np.asarray(labels["test_dg"], dtype=np.float64),
    )


def make_synthetic(n: int, seed: int = 0, zoh: bool = True) -> SubjectData:
    """
    Generate fake-but-structurally-correct data for subject n.

    Not realistic - just enough structure for an end-to-end smoke test:
      * correct shapes / sampling rate / channel counts
      * glove zero-order-held at 25 Hz, like the real recording
      * an 80 Hz (gamma-band) carrier amplitude-modulated by each finger's
        flexion, injected into one channel per finger, so band power is
        genuinely informative and later stages have something to find
    """
    rng = np.random.default_rng(seed + n)
    n_ch = C.SUBJECT_CHANNELS[n]
    T_train = C.TRAIN_SECONDS * C.FS_ECOG
    T_test = C.TEST_SECONDS * C.FS_ECOG
    T = T_train + T_test
    t = np.arange(T) / C.FS_ECOG

    glove = np.zeros((T, C.N_FINGERS))
    for f in range(C.N_FINGERS):
        freq = 0.5 + 0.25 * f
        glove[:, f] = (np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
                       + 0.3 * rng.standard_normal(T))

    if zoh:
        w = C.AM_WINDOW_SAMPLES
        glove = np.repeat(glove[::w], w, axis=0)[:T]

    ecog = rng.standard_normal((T, n_ch))
    carrier = np.sin(2 * np.pi * 80.0 * t)
    for f in range(C.N_FINGERS):
        ch = (f * 7) % n_ch
        ecog[:, ch] += 3.0 * np.clip(glove[:, f], 0, None) * carrier

    return SubjectData(
        subject=n,
        train_ecog=ecog[:T_train],
        train_glove=glove[:T_train],
        test_ecog=ecog[T_train:],
        test_glove=glove[T_train:],
    )


def write_synthetic_mat(n: int, data_dir=None, seed: int = 0):
    """Write synthetic data to .mat files matching the real layout."""
    from scipy.io import savemat
    data_dir = C.DATA_DIR if data_dir is None else data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    sd = make_synthetic(n, seed=seed)
    savemat(str(data_dir / f"sub{n}_comp.mat"),
            {"train_data": sd.train_ecog, "train_dg": sd.train_glove,
             "test_data": sd.test_ecog})
    savemat(str(data_dir / f"sub{n}_testlabels.mat"), {"test_dg": sd.test_glove})
