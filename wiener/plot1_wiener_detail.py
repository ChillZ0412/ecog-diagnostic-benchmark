"""
Plot 1 — Wiener Filter: Within-Subject Evaluation.
Matches the visual convention of the teammate's classification chart:
  * one subplot per metric
  * subjects on the x-axis
  * grouped bars = categories to compare (here: the 5 fingers, since Wiener
    is currently the only fully-complete regression method — there is no
    cross-method comparison to make yet within this single chart)
  * error bars where a natural sub-unit of variability exists
  * a dashed reference line for the paper's target, playing the same role
    as the "chance level" line in the classification chart

UPDATED: numbers now reflect the train-only electrode QC screen (default-on
in run_reproduction.py). Subject 3's ring finger, previously a catastrophic
artifact-driven failure (r=-0.004), is now r=+0.562 -- fixed via a
leakage-free preprocessing step, not a post-hoc/manual channel removal.
Subjects 1-2 (no known artifact) are essentially unchanged, confirming the
screen doesn't cost anything where it isn't needed.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINGERS = ["Thumb", "Index", "Middle", "Ring", "Little"]
COLORS = ["#2E5C8A", "#8E3B7D", "#D4A017", "#C0392B", "#4A9B7F"]

# with train-only electrode QC (default pipeline as of this run)
ours = {
    "S1": [0.581, 0.710, 0.145, 0.541, 0.353],
    "S2": [0.562, 0.234, 0.192, 0.403, 0.304],
    "S3": [0.698, 0.553, 0.573, 0.562, 0.620],
}
paper = {
    "S1": [0.58, 0.71, 0.14, 0.53, 0.29],
    "S2": [0.51, 0.37, 0.24, 0.47, 0.35],
    "S3": [0.69, 0.46, 0.58, 0.58, 0.63],
}

subjects = ["sub1", "sub2", "sub3"]
subj_keys = ["S1", "S2", "S3"]

fig, ax = plt.subplots(figsize=(9, 5.5))

n_fingers = len(FINGERS)
bar_w = 0.15
group_gap = 1.0
x_base = np.arange(len(subjects)) * group_gap

for fi, finger in enumerate(FINGERS):
    xs = x_base + (fi - (n_fingers - 1) / 2) * bar_w
    vals = [ours[s][fi] for s in subj_keys]
    ax.bar(xs, vals, width=bar_w * 0.92, color=COLORS[fi], label=finger,
          edgecolor="white", linewidth=0.5, zorder=3)

# paper reference: small black diamond markers per finger per subject
for fi, finger in enumerate(FINGERS):
    xs = x_base + (fi - (n_fingers - 1) / 2) * bar_w
    pvals = [paper[s][fi] for s in subj_keys]
    ax.scatter(xs, pvals, marker="D", s=26, color="black", zorder=4,
              label="Paper (Liang & Bougrain 2012)" if fi == 0 else None)

ax.axhline(0, color="#888888", linewidth=0.8, zorder=1)
ax.set_xticks(x_base)
ax.set_xticklabels(subjects, fontsize=12)
ax.set_ylabel("Pearson r", fontsize=12)
ax.set_title("Wiener Filter — Within-Subject Evaluation", fontsize=16, fontweight="bold")
ax.set_ylim(-0.15, 0.85)
ax.grid(axis="y", color="#e5e5e5", zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Finger", loc="upper left",
         bbox_to_anchor=(1.01, 1.0), frameon=True, fontsize=10, title_fontsize=11)

fig.tight_layout()
fig.savefig("wiener_within_subject.png", dpi=200, bbox_inches="tight")
print("saved -> wiener_within_subject.png")

