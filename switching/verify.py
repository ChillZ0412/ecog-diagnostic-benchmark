"""
Local verification script -- run this in the same folder as config.py,
data_io.py, synthetic_blocks.py, state_labels.py, ar_features.py,
channel_selection.py to confirm Stages 0-3 work on your machine
(catches numpy/scipy version differences before we build Stage 4).

Usage:
    python verify.py
"""
import numpy as np

import config as C
from data_io import make_synthetic
from synthetic_blocks import make_synthetic_blocks
from state_labels import make_state_labels
from ar_features import extract_ar_features
from channel_selection import rank_channels, select_top_k

print("1. config OK, N_STATES =", C.N_STATES)

sd = make_synthetic_blocks(1, seed=0)
print("2. synthetic_blocks OK, shape =", sd.train_ecog.shape)

sl = make_state_labels(sd.train_glove, sd.test_glove, subject=1)
print("3. state_labels OK, states present =", sorted(set(sl.train_state.tolist())))

afs = extract_ar_features(sd.train_ecog[:20000], channels=range(5),
                           shifts_ms=[0, 20], n_coeffs_keep=2)
print("4. ar_features OK, shape =", afs.features.shape,
      "finite:", bool(np.isfinite(afs.features).all()))

ranking = rank_channels(sd.train_ecog, sl.train_state, n_states=C.N_STATES)
top5 = select_top_k(ranking, k=5)
print("5. channel_selection OK, top-5 channels =", sorted(top5.tolist()))

print("\nALL OK")
