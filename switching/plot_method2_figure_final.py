"""
Generate the FINAL oracle+estimated vs actual finger-flexion figure for
the Switching Linear Models results slide, using the tuned hyperparameters
from final_run.py (not the earlier preliminary ones).

Run in the same folder as the other files, with data/ populated:

    python plot_method2_figure_final.py

Plots Subject 3, thumb -- oracle (b) and estimated (c) are both strong and
close together for this subject/finger (oracle=0.521, estimated=0.406),
which is the cleanest illustration of "when the classifier works, the
deployable (estimated) decode approaches the theoretical (oracle) upper
bound" -- the central finding of the whole debugging chain.

Output: method2_final_trace.png in the current directory.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from data_io import load_subject_clean
from state_labels import make_state_labels
from evaluate_decoder import fit_decoder, evaluate
from finger_regressor import pearson_r

SUBJECT = 3
FINGER_IDX = 0  # 0=thumb
PLOT_SECONDS = 40

TAU_SAMPLES = 500  # subject 3's tuned tau (from tune_regression.py)
M = 30

sd = load_subject_clean(SUBJECT)
sl = make_state_labels(sd.train_glove, sd.test_glove, subject=SUBJECT)

print("Fitting decoder with FINAL tuned hyperparameters...")
bundle = fit_decoder(
    sd.train_ecog, sd.train_glove, sl.train_state,
    n_select_channels=15, lambda_s=0.0, tau_samples=TAU_SAMPLES, lambda_k=1.0, M=M,
)

res = evaluate(bundle, sd.test_ecog, sd.test_glove, sl.test_state)
r_oracle = res["oracle"][FINGER_IDX]
r_estimated = res["estimated"][FINGER_IDX]
print(f"Subject {SUBJECT}, {C.FINGER_NAMES[FINGER_IDX]}: "
      f"oracle r={r_oracle:.3f}  estimated r={r_estimated:.3f}")

# rebuild the actual/oracle/estimated traces for plotting (mirrors evaluate()'s
# internals, since evaluate() only returns correlations, not the arrays)
from regression_features import savgol_smooth_all_channels, build_regression_features
from ar_features import extract_ar_features

smoothed_test = savgol_smooth_all_channels(sd.test_ecog)
X_reg_test, (lo, hi) = build_regression_features(smoothed_test, bundle.tau_samples)
y_test_aligned = sd.test_glove[lo:hi]
state_true_aligned = sl.test_state[lo:hi]

T = X_reg_test.shape[0]


def decode_with_states(state_sequence):
    y_pred = np.zeros((T, C.N_FINGERS))
    for k, model in bundle.Hk_models.items():
        mask = state_sequence == k
        if mask.sum() == 0:
            continue
        y_pred[mask] = model.predict(X_reg_test[mask])
    return y_pred


y_pred_oracle = decode_with_states(state_true_aligned)

afs_test = extract_ar_features(sd.test_ecog, channels=bundle.top_channels,
                                shifts_ms=bundle.ar_shifts_ms, n_coeffs_keep=2)
pred_state_full = bundle.classifier.predict(afs_test.features)
pred_state_aligned = pred_state_full[lo:hi]
y_pred_estimated = decode_with_states(pred_state_aligned)

n_plot = min(PLOT_SECONDS * C.FS_ECOG, T)
t = np.arange(n_plot) / C.FS_ECOG

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(t, y_test_aligned[:n_plot, FINGER_IDX], color="#1F3864", linewidth=1.3, label="Actual")
ax.plot(t, y_pred_oracle[:n_plot, FINGER_IDX], color="#548235", linewidth=1.0,
        linestyle="--", label=f"Oracle (r={r_oracle:.2f})")
ax.plot(t, y_pred_estimated[:n_plot, FINGER_IDX], color="#C00000", linewidth=1.0,
        linestyle=":", label=f"Estimated (r={r_estimated:.2f})")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Finger flexion (a.u.)")
ax.set_title(f"Subject {SUBJECT} \u2014 {C.FINGER_NAMES[FINGER_IDX].capitalize()} finger: "
             f"Actual vs. Oracle vs. Estimated", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig("method2_final_trace.png", dpi=200)
print("saved method2_final_trace.png")
