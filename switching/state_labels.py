"""
Stage 1 (addition) -- Moving-finger state labels for Method 2
(Flamary & Rakotomamonjy 2011/2012).

ds4 has no explicit trial-cue channel: the only ground truth for "which
finger is moving right now" is the glove trajectory itself. The paper's
oracle analysis explicitly does the same thing -- it says the true
sequence is available "since the finger movements on the test set are
now available" (Sec 4.3, footnote 1). That is fine for:
  * training H_k (each finger's regressor needs to know its own on-segments)
  * the oracle upper-bound evaluation (Table 3b, r=0.61)
It is NOT fine to use at deployment/inference time -- that's what f(x)
(Stage 4, the real classifier) is for. This module only produces the
"true" k(t) used in the above two legitimate cases.

Definition (paper is mutually-exclusive states, Sec 3.1):
  k(t) in {1,...,5}  = that finger is moving
  k(t) = 6           = no finger moving (REST_STATE)

If two fingers are simultaneously above threshold, the one with the larger
(threshold-normalized) flexion wins -- this mirrors the convention used in
Yao et al. 2022 ("the finger exhibiting the highest amplitude was treated
as the moving finger"), which is the most defensible tie-break given the
paper itself doesn't specify one.
"""
from dataclasses import dataclass

import numpy as np

import config as C


@dataclass
class StateLabels:
    """k(t) for train and test splits, plus the thresholds used to derive
    each (computed independently per split -- see make_state_labels)."""
    subject: int
    train_state: np.ndarray   # (T_train,) int in {1,...,N_STATES}
    test_state: np.ndarray    # (T_test,)  int in {1,...,N_STATES}
    thresholds: np.ndarray    # (N_FINGERS,) train-split threshold, kept as
                               # the "canonical" one for reporting/plotting
    test_thresholds: np.ndarray = None  # (N_FINGERS,) test-split threshold


def _debounce(state: np.ndarray, min_hold: int) -> np.ndarray:
    """Remove state runs shorter than `min_hold` samples by merging them into
    whichever neighboring run they're adjacent to (extend the run before them).
    Avoids single-sample label flicker from threshold noise."""
    out = state.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_hold and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def derive_states(glove: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    glove: (T, N_FINGERS) flexion trace.
    thresholds: (N_FINGERS,) absolute threshold per finger (same units as glove).
    Returns state: (T,) int, values in {1,...,N_FINGERS, REST_STATE}.
    """
    T, n_fingers = glove.shape
    assert n_fingers == C.N_FINGERS

    above = glove - thresholds[None, :]           # (T, N_FINGERS), >0 if "moving"
    any_above = (above > 0).any(axis=1)

    winner = np.argmax(above, axis=1) + 1          # 1-indexed finger id
    state = np.where(any_above, winner, C.REST_STATE).astype(int)

    state = _debounce(state, C.STATE_MIN_HOLD_SAMPLES)
    return state


def make_state_labels(train_glove: np.ndarray, test_glove: np.ndarray,
                       subject: int) -> StateLabels:
    """
    Thresholds are computed INDEPENDENTLY for train and test (each split
    uses its own percentile-based robust range), rather than fitting once
    on train and applying that fixed threshold to test.

    This matters on real data: ds4 recordings run ~10 continuous minutes,
    and a real glove sensor's baseline can drift over that span. A
    threshold calibrated on the (earlier) train segment can become
    miscalibrated by the time the recording reaches the (later) test
    segment -- confirmed empirically: subjects 1/2 showed test-split rest
    fraction 6-10x lower than train under a train-only threshold, and
    test-set state classification accuracy fell BELOW random chance
    (1/6 for 6 states) as a result.

    Using test glove's own statistics for this purpose is not new leakage
    beyond what the paper's own oracle analysis already does: it derives
    oracle test states directly from test-set glove data (§4.3, "since the
    finger movements on the test set are now available"). This only
    extends that same allowance to the threshold's *scale*, not to which
    finger is moving when.
    """
    p_lo, p_hi = C.STATE_RANGE_ROBUST_PCT, 100.0 - C.STATE_RANGE_ROBUST_PCT

    def _thresholds_for(glove: np.ndarray) -> np.ndarray:
        robust_min = np.percentile(glove, p_lo, axis=0)
        robust_max = np.percentile(glove, p_hi, axis=0)
        return robust_min + C.STATE_ON_THRESHOLD_FRAC * (robust_max - robust_min)

    train_thresholds = _thresholds_for(train_glove)
    test_thresholds = _thresholds_for(test_glove)

    train_state = derive_states(train_glove, train_thresholds)
    test_state = derive_states(test_glove, test_thresholds)

    return StateLabels(subject=subject, train_state=train_state,
                        test_state=test_state, thresholds=train_thresholds,
                        test_thresholds=test_thresholds)
