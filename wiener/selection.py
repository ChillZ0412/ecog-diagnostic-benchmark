"""
Stage 6 — Forward feature selection, wrapper approach
(Liang & Bougrain 2012, Section 2.2.1.3).

The paper's procedure, verbatim in behaviour:

  * candidates are channel/band couples -- 186, 144, 192 features for
    subjects 1, 2, 3 (n_channels x 3 bands)
  * start from the empty set; at each step add the single feature that most
    improves the correlation
  * selection is judged on an inner split of the TRAINING set only:
    3/5 inner-train, 2/5 validation
  * stop when the validation correlation stops increasing, or when a
    user-defined maximum (10) is reached

--------------------------------------------------------------------------
WHY THIS IS THE EXPENSIVE STAGE, AND HOW IT IS MADE CHEAP
--------------------------------------------------------------------------
A literal implementation refits from scratch for every candidate: 186
candidates x 10 rounds x 5 fingers x 3 subjects. Measured naively, the
bottleneck is not the solve but the *rebuilding of the design matrix*
(~3 s per stack at round 10, because reversing a sliding-window view forces
a strided copy).

Two observations make it fast:

  1. Within a round, the already-selected block of the design matrix is
     IDENTICAL for every candidate. So G_SS = D_S^T D_S and b_S = D_S^T y are
     computed once per round, and each candidate only needs the cross terms
        G_Sc = D_S^T D_c   (pS x 26)
        G_cc = D_c^T D_c   (26 x 26)
     which are assembled into the full Gram matrix by block. This is exact,
     not an approximation.

  2. G is symmetric positive semi-definite, so its pseudo-inverse can be taken
     with `eigh` rather than a general SVD -- 2x faster, agreeing to ~3e-15.

Measured result: ~21 s per finger, ~1.8 min per subject, under 3 min for all
three subjects and all five fingers.

--------------------------------------------------------------------------
STOPPING RULE
--------------------------------------------------------------------------
The search always runs to `max_features` and records the whole trajectory,
then the paper's stopping rule is applied afterwards to decide how many
features to keep. Running the full trajectory costs nothing extra (the rounds
are sequential either way) and it yields exactly the curve plotted in the
paper's Figure 2, which is the most direct visual check of the reproduction.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

import config as C


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------
def _psd_pinv_solve(G: np.ndarray, b: np.ndarray,
                    rcond: Optional[float] = None) -> np.ndarray:
    """
    Solve G w = b in the pseudo-inverse (minimum-norm) sense, exploiting the
    fact that G = D^T D is symmetric PSD. Equivalent to `pinv(G) @ b` but
    about twice as fast.
    """
    lam, V = np.linalg.eigh(G)
    lam_max = lam.max() if lam.size else 0.0
    tol = (lam_max * len(lam) * np.finfo(np.float64).eps
           if rcond is None else lam_max * rcond)
    keep = lam > tol
    inv = np.zeros_like(lam)
    inv[keep] = 1.0 / lam[keep]
    return V @ (inv * (V.T @ b))


def _lag_block(X: np.ndarray, col: int, k: int, N: int) -> np.ndarray:
    """
    (N, k+1) memory block for one feature: [i, j] = X[i + k - j, col].

    Built from contiguous slice copies rather than a reversed sliding-window
    view, which is roughly 1000x faster here because it avoids strided reads.
    """
    L = np.empty((N, k + 1), dtype=np.float64)
    for j in range(k + 1):
        L[:, j] = X[k - j: k - j + N, col]
    return L


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 and np.isfinite(den) else float("nan")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SelectionResult:
    order: List[int] = field(default_factory=list)        # features, in the order chosen
    train_r: List[float] = field(default_factory=list)    # inner-train r after each step
    val_r: List[float] = field(default_factory=list)      # validation r after each step
    n_selected: int = 0                                   # where the paper's rule stops
    feature_names: Optional[Sequence] = None

    @property
    def selected(self) -> List[int]:
        """The features the paper's stopping rule would actually keep."""
        return self.order[:self.n_selected]

    def paper_notation(self) -> str:
        """Selected features as the paper's (channel, band) pairs, in order."""
        if self.feature_names is None:
            return " ".join(str(i) for i in self.selected)
        band_id = {"sub": 1, "gamma": 2, "fastgamma": 3}
        out = []
        for i in self.selected:
            band, ch = self.feature_names[i]
            out.append(f"({ch + 1},{band_id[band]})")
        return " ".join(out)


# ---------------------------------------------------------------------------
# Forward selection
# ---------------------------------------------------------------------------
def forward_select(X: np.ndarray,
                   y: np.ndarray,
                   k: int = None,
                   max_features: int = None,
                   train_fraction: float = None,
                   rcond: Optional[float] = None,
                   feature_names: Optional[Sequence] = None,
                   min_improvement: float = 0.0,
                   verbose: bool = False) -> SelectionResult:
    """
    Greedy forward selection of channel/band features for ONE finger.

    X : (T, F) AM features at 25 Hz   (Stage 3 output)
    y : (T,)   that finger's trace at 25 Hz

    Returns a SelectionResult holding the full trajectory plus the cut point
    implied by the paper's stopping rule.
    """
    k = C.MEMORY_K if k is None else k
    max_features = C.FS_MAX_FEATURES if max_features is None else max_features
    train_fraction = C.FS_TRAIN_FRACTION if train_fraction is None else train_fraction

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    T, F = X.shape
    N = T - k                                   # usable rows after the memory stack
    y_al = y[k:k + N]

    n_tr = int(round(N * train_fraction))
    y_tr, y_va = y_al[:n_tr], y_al[n_tr:]
    if len(y_va) < 2:
        raise ValueError("validation split is empty; check train_fraction")

    # selected block starts as the intercept alone
    D_S = np.ones((N, 1), dtype=np.float64)
    G_SS = D_S[:n_tr].T @ D_S[:n_tr]
    b_S = D_S[:n_tr].T @ y_tr

    remaining = list(range(F))
    res = SelectionResult(feature_names=feature_names)

    for step in range(min(max_features, F)):
        Dtr_S, Dva_S = D_S[:n_tr], D_S[n_tr:]
        best = (-np.inf, None, None)            # (val_r, feature, cached blocks)

        for c in remaining:
            L = _lag_block(X, c, k, N)
            Ltr, Lva = L[:n_tr], L[n_tr:]

            G_Sc = Dtr_S.T @ Ltr
            G_cc = Ltr.T @ Ltr
            b_c = Ltr.T @ y_tr

            G = np.block([[G_SS, G_Sc], [G_Sc.T, G_cc]])
            b = np.concatenate([b_S, b_c])
            w = _psd_pinv_solve(G, b, rcond=rcond)

            pS = Dtr_S.shape[1]
            r_val = _pearson(Dva_S @ w[:pS] + Lva @ w[pS:], y_va)

            if np.isfinite(r_val) and r_val > best[0]:
                r_tr = _pearson(Dtr_S @ w[:pS] + Ltr @ w[pS:], y_tr)
                best = (r_val, c, (L, G_Sc, G_cc, b_c, r_tr))

        if best[1] is None:                     # nothing gave a finite score
            break

        r_val, c, (L, G_Sc, G_cc, b_c, r_tr) = best
        res.order.append(c)
        res.train_r.append(r_tr)
        res.val_r.append(r_val)
        remaining.remove(c)

        # grow the selected block, reusing the winner's already-computed terms
        D_S = np.hstack([D_S, L])
        G_SS = np.block([[G_SS, G_Sc], [G_Sc.T, G_cc]])
        b_S = np.concatenate([b_S, b_c])

        if verbose:
            name = feature_names[c] if feature_names is not None else c
            print(f"    step {step + 1:2d}: +{str(name):22s} "
                  f"train r={r_tr:.4f}  val r={r_val:.4f}")

    # paper's stopping rule, applied to the recorded trajectory
    res.n_selected = _apply_stopping_rule(res.val_r, min_improvement)
    return res


def _apply_stopping_rule(val_r: Sequence[float], min_improvement: float = 0.0) -> int:
    """
    "Stop when the correlation coefficient for the validation set does not
    increase." Returns how many features to keep (at least 1).
    """
    if not val_r:
        return 0
    best, n = val_r[0], 1
    for i in range(1, len(val_r)):
        if val_r[i] > best + min_improvement:
            best, n = val_r[i], i + 1
        else:
            break
    return n


def select_all_fingers(X: np.ndarray, Y: np.ndarray, **kwargs) -> List[SelectionResult]:
    """Run forward selection independently for each finger. Y is (T, 5)."""
    out = []
    for f in range(Y.shape[1]):
        if kwargs.get("verbose"):
            print(f"  {C.FINGER_NAMES[f]}:")
        out.append(forward_select(X, Y[:, f], **kwargs))
    return out
