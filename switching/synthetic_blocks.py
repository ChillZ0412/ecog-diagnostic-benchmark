"""
Block-structured synthetic data for Method 2 smoke testing.

Separate from data_io.make_synthetic() on purpose: that generator is shared
with the Wiener pipeline (Method 1) and is left untouched. This module only
adds a second, more realistic synthetic generator with an actual move/rest
trial structure, needed so Method 2's state classifier (Stage 4) has a
meaningful rest class to be tested against.

Trial paradigm matches the numbers reported for the real dataset (Yao et al.
2022, Sec 2.1): each finger movement lasts 2s, followed by a 2s rest period,
30 movement trials per finger -> 5 * 30 * 4s = 600s per subject, which is
exactly config.TRAIN_SECONDS + config.TEST_SECONDS (400 + 200 = 600). Trial
order is a random permutation of the 150 (finger, repeat) pairs, not blocked
by finger, matching a randomized-cue experimental design.

Everything else (channel count per subject, ECoG carrier-injection scheme,
zero-order-hold of the glove) reuses the same conventions as
data_io.make_synthetic() so the two generators are structurally comparable.
"""
import numpy as np

import config as C
from data_io import SubjectData

TRIAL_MOVE_SEC = 2.0
TRIAL_REST_SEC = 2.0
TRIALS_PER_FINGER = 30


def _trial_order(seed: int) -> np.ndarray:
    """Random permutation of 150 (finger) labels, 30 reps each, 1-indexed."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(1, C.N_FINGERS + 1), TRIALS_PER_FINGER)
    rng.shuffle(labels)
    return labels


def _move_bump(n_samples: int) -> np.ndarray:
    """Smooth 0 -> peak -> 0 flexion bump over a move segment (raised-cosine).

    NOTE: 0.5*(1-cos(t)) for t in [0,pi] is WRONG here -- that's monotonically
    increasing (0 at start, 1 at end), which would put the peak at the trial
    boundary and create a discontinuous jump back to rest. Using 2*t inside
    the cosine instead gives the actual 0 -> 1 -> 0 bell shape, peaking at
    the window's midpoint.
    """
    t = np.linspace(0, np.pi, n_samples)
    return 0.5 * (1 - np.cos(2 * t))  # in [0, 1], peak at n_samples/2


def make_synthetic_blocks(n: int, seed: int = 0, zoh: bool = True) -> SubjectData:
    """
    Generate block-structured fake data for subject n: a sequence of
    (move 2s / rest 2s) trials cycling through a shuffled finger order,
    30 trials per finger. Returns the same SubjectData shape as
    data_io.make_synthetic(), so it's a drop-in replacement for Method 2's
    smoke tests.
    """
    rng = np.random.default_rng(seed + n + 1000)  # offset seed so it doesn't
                                                    # coincide with make_synthetic
    n_ch = C.SUBJECT_CHANNELS[n]
    fs = C.FS_ECOG

    move_n = int(round(TRIAL_MOVE_SEC * fs))
    rest_n = int(round(TRIAL_REST_SEC * fs))
    trial_n = move_n + rest_n

    order = _trial_order(seed + n)
    T = len(order) * trial_n
    assert T == (C.TRAIN_SECONDS + C.TEST_SECONDS) * fs, (
        f"trial paradigm produces {T/fs}s, expected "
        f"{C.TRAIN_SECONDS + C.TEST_SECONDS}s -- check TRIALS_PER_FINGER / "
        f"TRIAL_MOVE_SEC / TRIAL_REST_SEC against config.py"
    )

    glove = np.zeros((T, C.N_FINGERS))
    bump = _move_bump(move_n)
    baseline_noise = 0.05 * rng.standard_normal((T, C.N_FINGERS))

    for i, finger in enumerate(order):
        start = i * trial_n
        # amplitude varies a bit trial-to-trial, like real flexion depth
        amp = 1.0 + 0.15 * rng.standard_normal()
        glove[start:start + move_n, finger - 1] += amp * bump
        # small cross-talk into neighboring fingers, like real data's
        # "other fingers moved together... with much smaller amplitude"
        for other in range(C.N_FINGERS):
            if other != finger - 1:
                glove[start:start + move_n, other] += 0.1 * amp * bump * rng.uniform(0, 1)

    glove = np.clip(glove + baseline_noise, 0, None)

    if zoh:
        w = C.AM_WINDOW_SAMPLES
        glove = np.repeat(glove[::w], w, axis=0)[:T]

    ecog = rng.standard_normal((T, n_ch))
    carrier = np.sin(2 * np.pi * 80.0 * np.arange(T) / fs)
    for f in range(C.N_FINGERS):
        ch = (f * 7) % n_ch
        # 80Hz carrier, AM-modulated by flexion -- gives Stage 3/4's spectral/AR
        # features something to detect (presence/absence of band-limited power).
        ecog[:, ch] += 3.0 * glove[:, f] * carrier
        # low-frequency baseband component, directly proportional to flexion --
        # gives Stage 5/6/7's H_k regression something to detect. Without this,
        # H_k's raw-time-domain-after-0.4s-lowpass features contain NO flexion
        # information: a 0.4s Savitzky-Golay window attenuates an 80Hz carrier
        # by ~10x (verified separately), so the AM component alone is invisible
        # to that pathway even though it's exactly what Stage 3/4 needs.
        ecog[:, ch] += 2.0 * glove[:, f]

    T_train = C.TRAIN_SECONDS * fs
    return SubjectData(
        subject=n,
        train_ecog=ecog[:T_train],
        train_glove=glove[:T_train],
        test_ecog=ecog[T_train:],
        test_glove=glove[T_train:],
    )
