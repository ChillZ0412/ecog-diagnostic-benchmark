"""
Evaluation metrics, result formatting, and persistence.

Provides classification metrics (Accuracy, Balanced Accuracy, Macro F1, Cohen's κ)
in a unified reporting format.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import FINGER_NAMES, RESULTS_DIR


# ─────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute all classification metrics in one call.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_proba : np.ndarray, shape (n_samples, n_classes), optional
        Predicted class probabilities. Required for AUC-ROC.
    label_names : list of str, optional
        Class names (default: FINGER_NAMES).

    Returns
    -------
    dict with keys:
        accuracy, balanced_accuracy, macro_f1, weighted_f1,
        cohen_kappa, confusion_matrix, confusion_matrix_norm (row-%),
        per_class (F1, precision, recall per class),
        auc_roc (per-class + macro, only if y_proba provided)
    """
    if label_names is None:
        label_names = FINGER_NAMES

    n_classes = len(label_names)
    labels = range(n_classes)
    cm = confusion_matrix(y_true, y_pred)

    # Row-normalized confusion matrix (%) — each row sums to 100%
    cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)) * 100

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_norm": [[round(v, 1) for v in row] for row in cm_norm.tolist()],
        "per_class": {
            label_names[i]: {
                "f1": float(f1_per_class[i]),
                "precision": float(prec[i]),
                "recall": float(rec[i]),
            }
            for i in range(n_classes)
        },
    }

    # AUC-ROC (one-vs-rest) — only if probabilities are available
    if y_proba is not None:
        try:
            auc_per_class = {}
            for i in range(n_classes):
                # One-vs-rest binary labels
                y_bin = (y_true == i).astype(int)
                try:
                    auc_per_class[label_names[i]] = float(
                        roc_auc_score(y_bin, y_proba[:, i])
                    )
                except ValueError:
                    auc_per_class[label_names[i]] = None  # class not present in test set
            # Macro-average AUC
            valid_aucs = [v for v in auc_per_class.values() if v is not None]
            auc_per_class["macro_avg"] = float(np.mean(valid_aucs)) if valid_aucs else None
            result["auc_roc"] = auc_per_class
        except Exception:
            result["auc_roc"] = {"error": "AUC-ROC computation failed"}

    return result


# ─────────────────────────────────────────────────────────────────────
def format_results_table(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Format a multi-method classification results table as a string.

    Parameters
    ----------
    results : dict
        {method_name: {metric_name: value, ...}}.

    Returns
    -------
    str
        Formatted table ready for console printing or saving.
    """
    headers = ["Method", "Acc", "Bal.Acc", "F1", "κ", "AUC", "Train(s)", "Infer(ms)"]
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa", "auc_roc_macro", "train_time", "infer_time"]

    lines = []
    header_line = "  ".join(f"{h:<12}" if i == 0 else f"{h:>10}" for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for method, metrics in results.items():
        row_str = f"{method:<12}"
        for k in keys:
            v = metrics.get(k)
            if k == "infer_time":
                v = v * 1000 if v is not None else None
                row_str += f"{v:>10.1f}" if v is not None else f"{'—':>10}"
            elif k == "train_time":
                row_str += f"{v:>10.2f}" if v is not None else f"{'—':>10}"
            else:
                row_str += f"{v:>10.3f}" if v is not None else f"{'—':>10}"
        lines.append(row_str)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────

def save_results(
    results: Dict,
    filename: str,
    results_dir: Optional[str] = None,
) -> str:
    """Save benchmark results to JSON.

    Parameters
    ----------
    results : dict
        Results to save.
    filename : str
        Output filename (e.g., "traditional_sub1.json").
    results_dir : str, optional
        Output directory (default: RESULTS_DIR).

    Returns
    -------
    str
        Full path to the saved file.
    """
    out_dir = results_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, filename)

    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_convert(results), f, indent=2, ensure_ascii=False)

    return filepath
