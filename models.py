"""
Traditional machine learning models for ECoG finger movement decoding.

Models:
    LightGBMClassifier — spectral features + gradient boosting
    CSPLDA             — CSP spatial filter + LDA, Blankertz et al. (2008)
    SpectralSVM        — spectral features + RBF SVM (ablation)
"""

import time
import warnings
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import CSPConfig, LightGBMConfig, SVMConfig


# ═════════════════════════════════════════════════════════════════════
# LightGBM Classifier
# ═════════════════════════════════════════════════════════════════════

class LightGBMClassifier:
    """LightGBM classifier on spectral band-power features.

    Parameters
    ----------
    cfg : LightGBMConfig
    """

    def __init__(self, cfg: LightGBMConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = None
        self.train_time: float = 0.0
        self.infer_time: float = 0.0
        # Suppress sklearn feature-names warning on raw numpy arrays
        warnings.filterwarnings(
            "ignore", category=UserWarning,
            message="X does not have valid feature names"
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LightGBMClassifier":
        """Fit LightGBM with feature standardization.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        import lightgbm as lgb

        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = lgb.LGBMClassifier(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            verbose=self.cfg.verbose,
        )
        self.model.fit(X_scaled, y_train)
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        t0 = time.time()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        self.infer_time = time.time() - t0
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_params(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "n_estimators": self.cfg.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "class_weight": self.cfg.class_weight,
        }


# ═════════════════════════════════════════════════════════════════════
# CSP + LDA Classifier  (Blankertz et al., 2008)
# ═════════════════════════════════════════════════════════════════════

class CSPLDA:
    """CSP spatial filtering + LDA classifier.

    CSP features are extracted externally via ``extract_csp_features()``;
    this class handles only the LDA training and inference.

    Parameters
    ----------
    cfg : CSPConfig
    """

    def __init__(self, cfg: CSPConfig):
        self.cfg = cfg
        self.lda = None
        self.train_time: float = 0.0
        self.infer_time: float = 0.0
        self._n_features: int = 0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CSPLDA":
        """Fit LDA on CSP log-variance features.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            CSP log-variance features.
        y_train : np.ndarray, shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        t0 = time.time()
        self._n_features = X_train.shape[1]

        self.lda = LinearDiscriminantAnalysis(
            solver=self.cfg.solver,
            shrinkage=self.cfg.shrinkage,
        )
        self.lda.fit(X_train, y_train)
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        t0 = time.time()
        y_pred = self.lda.predict(X)
        self.infer_time = time.time() - t0
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for AUC-ROC)."""
        return self.lda.predict_proba(X)

    def get_params(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "solver": self.cfg.solver,
            "shrinkage": self.cfg.shrinkage,
            "freq_band": self.cfg.freq_band,
            "n_components": self.cfg.n_components,
            "n_features": self._n_features,
        }


# ═════════════════════════════════════════════════════════════════════
# Spectral + SVM Classifier (backup / comparison)
# ═════════════════════════════════════════════════════════════════════

class SpectralSVM:
    """SVM classifier on spectral band-power features (for ablation study).

    Parameters
    ----------
    cfg : SVMConfig
    """

    def __init__(self, cfg: SVMConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.model = None
        self.train_time: float = 0.0
        self.infer_time: float = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "SpectralSVM":
        """Fit SVM on standardized spectral features."""
        from sklearn.svm import SVC

        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X_train)

        self.model = SVC(
            kernel=self.cfg.kernel,
            C=self.cfg.C,
            class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state,
            probability=True,   # enables predict_proba for AUC-ROC
        )
        self.model.fit(X_scaled, y_train)
        self.train_time = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        t0 = time.time()
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        self.infer_time = time.time() - t0
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for AUC-ROC)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_params(self) -> dict:
        """Return model metadata for reporting."""
        return {
            "kernel": self.cfg.kernel,
            "C": self.cfg.C,
        }
