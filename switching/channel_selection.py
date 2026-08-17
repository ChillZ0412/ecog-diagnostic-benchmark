"""
Stage 3 -- Channel selection for the moving-finger state classifier f(x)
(Flamary & Rakotomamonjy 2011/2012, Sec 3.2 "Channel Selection").

Paper procedure:
  1. Compute a REDUCED feature vector per channel: no time-shifted versions,
     only the first AR coefficient (this is exactly ar_features.extract_ar_features
     called with shifts_ms=[0], n_coeffs_keep=1 -- see that module's docstring).
  2. For each of the 6 states k (5 fingers + rest), fit a linear regression
     y = x^T c_k on the training set, where y in {+1,-1} indicates whether
     state k is active at that sample.
  3. Importance score per channel j = sum_{k=1..6} |c_k[j]|.
  4. Keep the top-K channels by score. The paper picks K by maximizing
     validation-set cross-correlation of the DOWNSTREAM decoder -- that
     requires Stage 4 (the actual classifier) to exist, so this module only
     produces the ranking + a `select_top_k()` helper; the K search itself
     is deferred to a later stage once Stage 4 is in place
     (config.STATE_CLF_MAX_CHANNELS stays None until then).

Regression is solved by plain least squares (np.linalg.lstsq), not ridge --
the paper's Eq. for this step has no regularization term (unlike Eq. 2/3
for the full classifier and H_k, which do). n_channels is small (48-64) so
this is solved on the full training set at full 1kHz resolution.
"""
from dataclasses import dataclass

import numpy as np

from ar_features import extract_ar_features


@dataclass
class ChannelRanking:
    importance: np.ndarray       # (n_channels,) sum_k |c_k[j]|
    coeffs: np.ndarray           # (n_states, n_channels) c_k per state
    ranked_channels: np.ndarray  # (n_channels,) channel indices, descending importance


def rank_channels(train_ecog: np.ndarray, train_state: np.ndarray,
                   n_states: int) -> ChannelRanking:
    """
    train_ecog: (T, n_channels)
    train_state: (T,) int in {1,...,n_states}
    """
    afs = extract_ar_features(train_ecog, channels=None, shifts_ms=[0], n_coeffs_keep=1)
    X = afs.features  # (T, n_channels), one column per channel (coeff 0, shift 0)
    n_channels = X.shape[1]

    coeffs = np.zeros((n_states, n_channels))
    for k in range(1, n_states + 1):
        y = np.where(train_state == k, 1.0, -1.0)
        c_k, *_ = np.linalg.lstsq(X, y, rcond=None)
        coeffs[k - 1] = c_k

    importance = np.sum(np.abs(coeffs), axis=0)  # (n_channels,)
    ranked = np.argsort(importance)[::-1]

    return ChannelRanking(importance=importance, coeffs=coeffs, ranked_channels=ranked)


def select_top_k(ranking: ChannelRanking, k: int) -> np.ndarray:
    """Return the top-k channel indices (not sorted numerically, sorted by
    importance descending -- caller can np.sort() if numeric order matters)."""
    return ranking.ranked_channels[:k]
