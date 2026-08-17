from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import config as C

def pearson_r(pred, target):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target)); pred, target = pred[:n], target[:n]
    p = pred - pred.mean(); t = target - target.mean()
    denom = np.linalg.norm(p) * np.linalg.norm(t)
    if denom == 0 or not np.isfinite(denom): return float("nan")
    return float((p @ t) / denom)

def r2_score(pred, target):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target)); pred, target = pred[:n], target[:n]
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean()) ** 2).sum())
    return float("nan") if sst == 0 else 1.0 - sse / sst

def calibrated_r2(pred, target):
    r = pearson_r(pred, target)
    return float("nan") if not np.isfinite(r) else r * r

def nrmse(pred, target):
    """NRMSE = RMSE / std(target). See regressor.py docstring for rationale."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target)); pred, target = pred[:n], target[:n]
    std_t = target.std()
    if std_t == 0 or not np.isfinite(std_t): return float("nan")
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    return rmse / std_t

def mae_score(pred, target):
    """
    Mean Absolute Error, same units as target (original dataglove flexion
    values in this project, no normalization applied) -- lower is better,
    0 is perfect prediction.

    Units note (verified 2026-08-04): this method's glove targets come
    straight from data_io.load_subject() with zero transforms applied.
    FingerFlex's MinMaxScaler-normalized predictions were independently
    inverse-transformed and cross-validated against this same raw range
    (e.g. S1 Middle: both sides give [-0.9527, +7.5789]), confirming all
    three regression methods share the same original units. MAE computed
    here is directly comparable across methods, no conversion needed.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    n = min(len(pred), len(target)); pred, target = pred[:n], target[:n]
    return float(np.mean(np.abs(pred - target)))

@dataclass
class WienerFit:
    weights: np.ndarray
    solver: str
    mu: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None
    has_intercept: bool = True
    diagnostics: dict = field(default_factory=dict)
    def _prepare(self, D):
        D = np.asarray(D, dtype=np.float64)
        if self.mu is None: return D
        if self.has_intercept:
            body = (D[:, :-1] - self.mu) / self.sd
            return np.column_stack([body, D[:, -1]])
        return (D - self.mu) / self.sd
    def predict(self, D): return self._prepare(D) @ self.weights
    def score(self, D, d): return pearson_r(self.predict(D), d)

def fit_wiener(D, d, solver=None, rcond=None, standardize=None, has_intercept=None, diagnostics=True):
    solver = C.SOLVER if solver is None else solver
    rcond = C.RCOND if rcond is None else rcond
    standardize = C.STANDARDIZE if standardize is None else standardize
    has_intercept = C.ADD_INTERCEPT if has_intercept is None else has_intercept
    D = np.asarray(D, dtype=np.float64); d = np.asarray(d, dtype=np.float64).ravel()
    n = min(D.shape[0], d.shape[0]); D, d = D[:n], d[:n]
    mu = sd = None
    if standardize:
        body = D[:, :-1] if has_intercept else D
        mu = body.mean(axis=0); sd = body.std(axis=0); sd[sd == 0] = 1.0
        body = (body - mu) / sd
        D = np.column_stack([body, D[:, -1]]) if has_intercept else body
    if solver == "svd":
        w, *_ = np.linalg.lstsq(D, d, rcond=rcond)
    elif solver in ("pinv_normal", "inv"):
        G = D.T @ D; rhs = D.T @ d
        if solver == "pinv_normal":
            w = np.linalg.pinv(G, rcond=rcond if rcond is not None else 1e-15) @ rhs
        else:
            try: w = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError: w = np.full(D.shape[1], np.nan)
    else:
        raise ValueError(f"unknown solver: {solver!r}")
    fit = WienerFit(weights=w, solver=solver, mu=mu, sd=sd, has_intercept=has_intercept)
    if diagnostics:
        s = np.linalg.svd(D, compute_uv=False)
        smax = s.max() if s.size else np.nan
        tol = smax * max(D.shape) * np.finfo(np.float64).eps
        fit.diagnostics = {
            "n_samples": D.shape[0], "n_params": D.shape[1],
            "cond_X": float(smax / s.min()) if s.min() > 0 else float("inf"),
            "cond_XtX": float((smax / s.min()) ** 2) if s.min() > 0 else float("inf"),
            "effective_rank": int((s > tol).sum()),
            "weight_norm": float(np.linalg.norm(w)),
            "train_r": pearson_r(D @ w, d),
        }
    return fit

def fit_all_fingers(D, Y, **kwargs):
    return [fit_wiener(D, Y[:, f], **kwargs) for f in range(Y.shape[1])]
