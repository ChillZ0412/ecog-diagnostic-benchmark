"""
Stage 4 — Short-term memory design matrix
(Liang & Bougrain 2012, Section 2.2.2).

The paper's linear model is

    d(t_n) = W^T X(t_n),   X(t_n) = [x(t_n), x(t_n-1), ..., x(t_n-k)]^T

with "the best results ... achieved when k = 25". At the 25 Hz feature rate
that is 26 taps spanning ~1.04 s of history per feature.

This is the single most commonly mis-implemented part of the method. It is
NOT a fixed 37 ms lag applied to the ECoG (that is the delay mentioned in the
dataset description and used by e.g. Flamary & Rakotomamonjy). It is a full
multi-lag stack, and the regressor learns its own weighting over that 1 s
window -- which is why an explicit delay correction is unnecessary here: the
k=25 stack subsumes it.

--------------------------------------------------------------------------
BOUNDARY HANDLING
--------------------------------------------------------------------------
The first k rows have no complete history. We DROP them ('valid' semantics)
rather than zero-padding, because zeros would be a lie about past power (all
AM features are strictly positive) and would bias the first k fits. At 25 Hz,
k=25 costs 25 of ~10000 samples = 0.25% of the data. `stack_targets()` trims
the target identically so the two never drift apart.

--------------------------------------------------------------------------
SIZE
--------------------------------------------------------------------------
Stacking multiplies the column count by (k+1) = 26. For the full 186-feature
set of subject 1 that is 4836 columns; at 400 s (10000 rows) the design matrix
is ~387 MB in float64. This is why Stage 6 selects features FIRST and stacks
only the chosen subset: 10 features -> 260 columns -> ~20 MB. Always pass a
`columns` subset when you can; `estimate_memory()` will tell you the cost
before you allocate it.

--------------------------------------------------------------------------
INTERCEPT
--------------------------------------------------------------------------
The paper's equation has no bias term. But AM features are strictly positive
and the glove signal has a clearly non-zero mean, so a model forced through
the origin wastes capacity fitting an offset it cannot represent. We expose
`add_intercept` (default True) and treat "no intercept" as the paper-faithful
ablation, to be compared at Stage 7 rather than decided by assertion.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config as C


def estimate_memory(n_rows: int, n_features: int, k: int = None,
                    add_intercept: bool = True, dtype=np.float64) -> dict:
    """Report the size of the design matrix before building it."""
    k = C.MEMORY_K if k is None else k
    rows = max(n_rows - k, 0)
    cols = n_features * (k + 1) + (1 if add_intercept else 0)
    nbytes = rows * cols * np.dtype(dtype).itemsize
    return {"rows": rows, "cols": cols, "megabytes": nbytes / 1e6}


def build_memory_stack(X: np.ndarray,
                       k: int = None,
                       columns: Optional[Sequence[int]] = None,
                       add_intercept: bool = True) -> np.ndarray:
    """
    Build the short-term memory design matrix.

    X       : (T, F) AM features at 25 Hz
    columns : optional subset of feature indices to stack (use this!)
    returns : (T-k, len(columns)*(k+1) [+1])

    Row i corresponds to time t = i + k, and within each feature's block the
    lags run j = 0, 1, ..., k, i.e. [x(t), x(t-1), ..., x(t-k)].
    """
    k = C.MEMORY_K if k is None else k
    X = np.asarray(X, dtype=np.float64)
    if columns is not None:
        X = X[:, list(columns)]

    if X.shape[0] <= k:
        raise ValueError(f"need more than k={k} samples, got {X.shape[0]}")

    # [i, f, m] = X[i+m, f]  ->  reverse last axis  ->  [i, f, j] = X[i+k-j, f]
    sw = sliding_window_view(X, k + 1, axis=0)[:, :, ::-1]
    D = sw.reshape(sw.shape[0], -1)

    if add_intercept:
        D = np.column_stack([D, np.ones(D.shape[0], dtype=D.dtype)])
    return np.ascontiguousarray(D)


def stack_targets(y: np.ndarray, k: int = None) -> np.ndarray:
    """
    Trim targets to match `build_memory_stack` output.

    Row i of the design matrix predicts y[i + k], so the aligned target is
    simply y[k:].
    """
    k = C.MEMORY_K if k is None else k
    return np.asarray(y, dtype=np.float64)[k:]


def build_xy(X: np.ndarray,
             y: np.ndarray,
             k: int = None,
             columns: Optional[Sequence[int]] = None,
             add_intercept: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: build the design matrix and the matching targets together."""
    k = C.MEMORY_K if k is None else k
    D = build_memory_stack(X, k=k, columns=columns, add_intercept=add_intercept)
    t = stack_targets(y, k=k)
    n = min(D.shape[0], t.shape[0])
    return D[:n], t[:n]


def column_labels(feature_names: Sequence, columns: Optional[Sequence[int]] = None,
                  k: int = None, add_intercept: bool = True) -> list:
    """
    Human-readable label per design-matrix column: (band, channel, lag).
    Useful for inspecting which lag a fitted weight belongs to.
    """
    k = C.MEMORY_K if k is None else k
    idx = range(len(feature_names)) if columns is None else columns
    labels = [(feature_names[i][0], feature_names[i][1], j)
              for i in idx for j in range(k + 1)]
    if add_intercept:
        labels.append(("intercept", -1, -1))
    return labels
