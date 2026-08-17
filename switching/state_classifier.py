"""
Stage 4 -- Joint sparse (group-lasso) moving-finger state classifier f(x)
(Flamary & Rakotomamonjy 2011/2012, Sec 3.2 "Model estimation", Eq. 1-2).

Solves, from scratch (no scikit-learn), by block-coordinate descent:

    C_hat = argmin_C  ||Y - XC||_F^2 + lambda_s * sum_i ||C_i,:||_2

Y in {+1,-1}^(T x n_states), X in R^(T x d), C in R^(d x n_states).
This is the exact objective in the paper's Eq. 2; the paper solves it with
a custom block-coordinate-descent algorithm from Rakotomamonjy (2009),
"Algorithms for multiple basis pursuit denoising" -- reimplemented here at
the row level rather than via scikit-learn's MultiTaskLasso, so the solver
itself (not just the objective) matches what the paper describes.

--------------------------------------------------------------------------
DERIVATION (group soft-thresholding update for row i)
--------------------------------------------------------------------------
Fix every row of C except row i (=: v). Write R = Y - X C + x_i v^T (the
residual with feature i's own contribution removed), a = x_i^T x_i (scalar),
b = x_i^T R (n_states-vector). The sub-problem in v is:

    minimize_v   a ||v||^2 - 2 b^T v + lambda_s ||v||_2

Stationarity at v=0 requires the subgradient of the L2 term (the unit ball)
to contain 2b/lambda_s, i.e. ||b|| <= lambda_s/2 -> v* = 0.
Otherwise v* is parallel to b; substituting v = t*b and solving for t gives

    v* = (1/a) * max(0, 1 - lambda_s/(2*||b||)) * b

Rows driven exactly to 0 are the ones the model decides are uninformative
for ALL 6 states simultaneously (row/group sparsity) -- this is what makes
it a group lasso rather than 6 independent per-state Lassos.

--------------------------------------------------------------------------
DECISION RULE (paper Eq. 1)
--------------------------------------------------------------------------
f(x) = argmax_k f_k(x) = argmax_k x^T c_k -- winner-take-all over the raw
regression scores. This is a regression-to-labels classifier, not a
probabilistic one; no softmax/logistic step anywhere, matching the paper.
No intercept term is used, matching Eq. 1-2 as written (X C only).
"""
from dataclasses import dataclass

import numpy as np


def build_target_matrix(state: np.ndarray, n_states: int) -> np.ndarray:
    """state: (T,) int in {1,...,n_states} -> Y: (T, n_states) in {+1,-1}."""
    T = len(state)
    Y = -np.ones((T, n_states))
    Y[np.arange(T), state - 1] = 1.0
    return Y


def fit_group_lasso(X: np.ndarray, Y: np.ndarray, lambda_s: float,
                     max_iter: int = 200, tol: float = 1e-6,
                     C_init: np.ndarray = None):
    """
    Block-coordinate descent for the group-lasso multi-task regression above.

    X: (T, d), Y: (T, n_states)
    Returns (C, n_iter, converged): C is (d, n_states); n_iter is how many
    full passes over all d rows were run; converged is whether max
    per-element change in C dropped below `tol` before max_iter was hit.
    """
    T, d = X.shape
    n_states = Y.shape[1]

    col_sq_norms = np.sum(X**2, axis=0)  # a_i, precomputed once
    col_sq_norms = np.where(col_sq_norms == 0, 1e-12, col_sq_norms)  # guard

    C = np.zeros((d, n_states)) if C_init is None else C_init.copy()
    R = Y - X @ C  # running residual, (T, n_states)

    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        C_prev = C.copy()
        for i in range(d):
            xi = X[:, i]
            R += np.outer(xi, C[i, :])          # un-remove feature i's contribution
            b = xi @ R                           # (n_states,)
            norm_b = np.linalg.norm(b)
            if norm_b <= lambda_s / 2.0:
                C[i, :] = 0.0
            else:
                C[i, :] = (1.0 - lambda_s / (2.0 * norm_b)) / col_sq_norms[i] * b
            R -= np.outer(xi, C[i, :])           # re-remove with the NEW row value

        delta = np.max(np.abs(C - C_prev))
        if delta < tol:
            converged = True
            break

    return C, n_iter, converged


def predict_state(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Winner-take-all decoding: argmax_k x^T c_k, returned 1-indexed."""
    scores = X @ C  # (T, n_states)
    return np.argmax(scores, axis=1) + 1


def accuracy(true_state: np.ndarray, pred_state: np.ndarray) -> float:
    return float(np.mean(true_state == pred_state))


@dataclass
class StateClassifier:
    lambda_s: float
    max_iter: int = 200
    tol: float = 1e-6
    C: np.ndarray = None
    n_iter_: int = None
    converged_: bool = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "StateClassifier":
        self.C, self.n_iter_, self.converged_ = fit_group_lasso(
            X, Y, self.lambda_s, self.max_iter, self.tol
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return predict_state(X, self.C)

    def row_sparsity(self) -> float:
        """Fraction of feature rows driven exactly to the zero vector."""
        return float(np.mean(np.all(self.C == 0, axis=1)))
