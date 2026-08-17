"""
Classification Methods Benchmark for ECoG Finger Movement Decoding.

Reference implementations:
- Spectral features + LightGBM (Yao and Shoaran, 2019)
- CSP spatial filtering + LDA (Blankertz et al., 2008)
- Spectral features + SVM (ablation)

Dataset: BCI Competition IV, Dataset 4 (ECoG, 3 subjects)
Split: official 400s train / 200s test
"""
from .config import (
    BenchmarkConfig,
    FeatureConfig,
    LightGBMConfig,
    CSPConfig,
    SVMConfig,
)
from .data_loader import load_subject, generate_labels, download_dataset, get_label_distribution
from .features import (
    extract_spectral_features,
    extract_csp_features,
)
from .models import (
    LightGBMClassifier,
    CSPLDA,
    SpectralSVM,
)
from .evaluation import (
    evaluate_classification,
    format_results_table,
    save_results,
)
