from pathlib import Path
DATA_DIR = Path(__file__).parent / "data"
FS_ECOG = 1000
FS_GLOVE = 1000
N_FINGERS = 5
FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]
SUBJECT_CHANNELS = {1: 62, 2: 48, 3: 64}
TRAIN_SECONDS = 400
TEST_SECONDS = 200
BANDS = {"sub": (1.0, 60.0), "gamma": (60.0, 100.0), "fastgamma": (100.0, 200.0)}
AM_WINDOW_MS = 40.0
FS_FEATURE = int(round(1000.0 / AM_WINDOW_MS))
AM_WINDOW_SAMPLES = int(round(FS_ECOG * AM_WINDOW_MS / 1000.0))
MEMORY_K = 25
FS_MAX_FEATURES = 10
FS_TRAIN_FRACTION = 3 / 5
PAPER_AVG_ALL = 0.48
COMPETITION_OFFICIAL = 0.46
FIR_METHOD = "firwin"
FIR_NUMTAPS = {"sub": 3301, "gamma": 1001, "fastgamma": 1001}
FIR_DTYPE = "float32"
APPLY_NOTCH_60HZ = False
NOTCH_FREQS = (60.0, 120.0, 180.0)
NOTCH_Q = 30.0
TARGET_DOWNSAMPLE = "first"
# --- Stage 5: regressor --------------------------------------------------
SOLVER = "pinv_normal"   # 'pinv_normal' (paper) | 'svd' | 'inv' (ablation)
RCOND = None             # singular-value cutoff; None -> numpy default
STANDARDIZE = False      # z-score features before solving (ablation)
ADD_INTERCEPT = True

# ============================================================
# Method 2: Switching Linear Models (Flamary & Rakotomamonjy 2011/2012)
# Additive block only -- nothing above this line is touched or reused
# differently than it already is for the Wiener pipeline.
# ============================================================

# --- state definition ---
N_STATES = N_FINGERS + 1          # k=1..5 fingers, k=6 = rest
REST_STATE = N_FINGERS + 1

# --- state derivation from glove trajectory (needed because ds4 has no
# explicit trial-cue channel; "true" state sequence is derived from the
# glove signal itself, exactly as the paper does for its oracle analysis:
# "since the finger movements on the test set are now available" (§4.3,
# footnote 1). This is NOT leakage for the oracle/H_k-only smoke test --
# it becomes leakage only if used at deployment/inference time. ---
STATE_ON_THRESHOLD_FRAC = 0.20    # finger counted "moving" above this
                                   # fraction of its own train-set flexion range

# Per-subject bad/artifact electrodes, reused DIRECTLY from the Wiener team's
# training-set-only heavy-tail detection (their handoff doc section 4) --
# same physical ECoG hardware/subjects, so their findings transfer without
# needing to redo the detection here. Independently corroborated on our side:
# a test/train max-abs-value ratio check on Subject 3 flagged channel 49 at
# ~1100x (matching their ~700,000x report at a different processing stage),
# which is already in this list.
BAD_CHANNELS = {
    1: [4, 25, 28, 50, 54],
    2: [20, 22, 30, 31, 37],
    3: [14, 22, 43, 44, 49],
}
STATE_RANGE_ROBUST_PCT = 1.0      # the "range" in the threshold formula uses
                                   # the [STATE_RANGE_ROBUST_PCT, 100-STATE_RANGE_ROBUST_PCT]
                                   # percentile pair instead of the literal
                                   # min/max. Found necessary from real-data
                                   # inspection: real subject 3's data has an
                                   # extreme single-sample low outlier (likely
                                   # a sensor glitch) on one finger that, with
                                   # a literal min(), dragged that finger's
                                   # threshold below its own 10th percentile
                                   # -- misclassifying ~54% of samples as that
                                   # finger moving. 1st/99th percentile is
                                   # robust to that while leaving well-behaved
                                   # subjects (1, 2) essentially unchanged.
STATE_MIN_HOLD_SAMPLES = 100      # debounce: ignore state flips shorter than this
                                   # (at FS_ECOG=1000Hz -> 100ms), avoids single-
                                   # sample noise flicker in the derived label

# --- Stage 2/3: AR feature extraction for state classifier f(x) ---
AR_WINDOW_SAMPLES = 300           # non-overlapping window, paper §3.2 exact value
AR_ORDER = 2                      # paper only says "the two first AR coefficients
                                   # are used", not the model order itself. Using
                                   # AR(2) as the literal-reading default -- OPEN
                                   # QUESTION flagged for professor at report time.
AR_TIME_SHIFTS_MS = [0]           # ts: signal also evaluated at +ts/-ts;
                                   # paper says ts is chosen by validation, no
                                   # default given -> grid-search once real data in

# --- Stage 3: channel selection for f(x) ---
STATE_CLF_MAX_CHANNELS = None     # K: chosen to maximize validation correlation

# --- Stage 4: joint sparse (group-lasso) state classifier ---
GROUP_LASSO_LAMBDA_S = None       # lambda_s: tuned on validation set, no paper default

# --- Stage 5: regression features for H_k ---
SG_POLYORDER = 3                  # Savitzky-Golay order, paper §3.3 exact value
SG_WINDOW_SEC = 0.4               # Savitzky-Golay window width, paper §3.3 exact value
TAU_MS = None                     # tau: time-lag for [x(t-tau), x(t), x(t+tau)]

# --- Stage 6: per-state ridge regression H_k ---
RIDGE_LAMBDA_K = None             # lambda_k: one per state k, tuned on validation set
H_FEATURE_PRUNE_M = None          # M: largest-|h| features kept, tuned per (k, subject)

# --- Reference results (paper Table 2/3, for sanity-checking your numbers) ---
PAPER_LINEAR_BASELINE = 0.305     # single global linear regression, no switching
PAPER_ORACLE_STATE = 0.613        # switching decoder, TRUE state sequence (near-term target)
PAPER_ESTIMATED_STATE = 0.427     # switching decoder, ESTIMATED state sequence (BCI comp. result)
