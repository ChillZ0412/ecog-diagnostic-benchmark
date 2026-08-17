"""
Plot 2 — Regression Benchmark: Current Status Across Methods.

This is the cross-method comparison chart, matching the teammate's
classification-benchmark visual language (subjects on x-axis, methods as
grouped bars, dashed reference line). Two important honesty constraints,
different from the classification chart, are enforced here:

  1. Switching Linear Models is NOT tuned yet (state classifier near-chance
     for S1/S2). Its bars are rendered with a hatch pattern and are
     explicitly labeled "(preliminary)" in the legend, so a reader cannot
     mistake it for a finished, comparable number.
  2. DL regressor has no results yet. Rather than omitting it silently
     (which could look like it was forgotten) or fabricating a placeholder
     bar (which could look like real data), it is rendered as an explicit
     "pending" annotation in its group position, with no bar at all.

UPDATED: Wiener numbers now reflect the train-only electrode QC screen
(default-on in run_reproduction.py). Subject 3's error bar shrinks sharply
(SD 0.28 -> 0.06 on the all-finger panel) because the ring finger is no
longer a catastrophic negative outlier -- fixed via a leakage-free
preprocessing step, not a post-hoc/manual fix.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

subjects = ["sub1", "sub2", "sub3"]

# --- panel 1: official metric (ring excluded) -----------------------------
wiener_official = {"S1": 0.4472, "S2": 0.3230, "S3": 0.6110}
wiener_official_sd = {"S1": 0.2498, "S2": 0.1659, "S3": 0.0644}
paper_official_avg = 0.46

# --- panel 2: all-finger average ------------------------------------------
wiener_avg5 = {"S1": 0.4660, "S2": 0.3390, "S3": 0.6012}
wiener_avg5_sd = {"S1": 0.2203, "S2": 0.1481, "S3": 0.0600}
paper_avg5_avg = 0.48

# switching linear models: estimated (deployable) r, subject-level only,
# no per-finger breakdown exists yet -> no error bars fabricated
switching_estimated = {"S1": 0.147, "S2": 0.176, "S3": 0.004}

WIENER_COLOR = "#1F3864"
SWITCH_COLOR = "#D4A017"

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
x = np.arange(len(subjects))
bar_w = 0.32

panels = [
    (axes[0], "Official Metric (ring excluded)", wiener_official, wiener_official_sd,
     paper_official_avg, "Paper official avg = 0.46"),
    (axes[1], "All-Finger Average", wiener_avg5, wiener_avg5_sd,
     paper_avg5_avg, "Paper avg = 0.48"),
]

for ax, title, wvals, wsd, paper_ref, paper_label in panels:
    w = [wvals[f"S{i+1}"] for i in range(3)]
    wsds = [wsd[f"S{i+1}"] for i in range(3)]
    sw = [switching_estimated[f"S{i+1}"] for i in range(3)]

    ax.bar(x - bar_w / 2, w, width=bar_w * 0.92, color=WIENER_COLOR,
          yerr=wsds, capsize=4, label="Wiener filter (final, w/ electrode QC)",
          edgecolor="white", linewidth=0.6, zorder=3)
    ax.bar(x + bar_w / 2, sw, width=bar_w * 0.92, color=SWITCH_COLOR,
          hatch="///", edgecolor="white", linewidth=0.6, alpha=0.85,
          label="Switching Linear Models (preliminary)", zorder=3)

    # DL regressor: explicit "pending" annotation, no bar at all
    for xi in x:
        ax.text(xi + bar_w * 1.35, 0.02, "DL:\npending", ha="center",
               va="bottom", fontsize=8, color="#999999", style="italic")

    ax.axhline(paper_ref, color="#555555", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(2.55, paper_ref + 0.015, paper_label, fontsize=9, color="#555555",
           ha="right", style="italic")
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(-0.15, 0.85)
    ax.grid(axis="y", color="#e5e5e5", zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

axes[0].legend(loc="upper left", fontsize=9, frameon=True)

fig.suptitle("Regression Performance — Cross-Method Benchmark Status",
            fontsize=16, fontweight="bold", y=1.02)
fig.text(0.5, -0.04,
        "Wiener: train-only electrode QC (leakage-free) resolved Subject 3's "
        "prior ring-finger artifact (r: -0.004 \u2192 +0.562), with no cost to "
        "any other finger. Switching Linear Models: state classifier not yet "
        "tuned for S1/S2 \u2014 numbers shown are preliminary.",
        ha="center", fontsize=9, color="#666666", style="italic")

fig.tight_layout()
fig.savefig("regression_benchmark_status.png", dpi=200, bbox_inches="tight")
print("saved -> regression_benchmark_status.png")

