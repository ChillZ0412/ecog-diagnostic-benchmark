"""
FingerFlex 验证曲线绘制脚本
读取 lightning_logs/version_X/metrics.csv 里逐epoch记录的 corr_mean_val，
画出三个subject完整的验证曲线（不是几个零散checkpoint点，是PyTorch Lightning
自动记录的真实逐epoch数据）。

用法：
    python plot_validation_curves.py

运行前请先确认下面 VERSION_MAP 里的 version 编号和实际subject对应正确
（可以先用 `cat lightning_logs\\version_X\\metrics.csv` 抽查确认）。
"""

import pandas as pd
import matplotlib.pyplot as plt

# ---- 根据时间戳推算的对应关系，运行前建议先抽查确认 ----
VERSION_MAP = {
    "S1 (62ch)": "lightning_logs/version_1/metrics.csv",
    "S2 (48ch)": "lightning_logs/version_3/metrics.csv",
    "S3 (63ch, ch49 excluded)": "lightning_logs/version_5/metrics.csv",
}

COLORS = {
    "S1 (62ch)": "#1F3864",
    "S2 (48ch)": "#ED7D31",
    "S3 (63ch, ch49 excluded)": "#70AD47",
}

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

for label, path in VERSION_MAP.items():
    df = pd.read_csv(path)

    # metrics.csv 里同一个epoch可能有多行(train/val分别记录)，
    # 只保留有 corr_mean_val 数值的行，按epoch聚合(取该epoch内非空的最后一条)
    if "corr_mean_val" not in df.columns:
        print(f"[警告] {path} 里没有找到 corr_mean_val 列，实际列名: {list(df.columns)}")
        continue

    val_df = df.dropna(subset=["corr_mean_val"])[["epoch", "corr_mean_val"]]
    val_df = val_df.groupby("epoch", as_index=False).last()  # 每个epoch取最后一次记录
    val_df = val_df.sort_values("epoch")

    print(f"{label}: {len(val_df)} 个epoch的数据点, 范围 epoch {val_df['epoch'].min()}~{val_df['epoch'].max()}, "
          f"最高corr_mean_val={val_df['corr_mean_val'].max():.4f} (epoch {val_df.loc[val_df['corr_mean_val'].idxmax(), 'epoch']:.0f})")

    ax.plot(val_df["epoch"], val_df["corr_mean_val"], marker='o', markersize=4,
            linewidth=2, label=label, color=COLORS[label])

    # 标出最优点
    best_idx = val_df["corr_mean_val"].idxmax()
    best_epoch = val_df.loc[best_idx, "epoch"]
    best_val = val_df.loc[best_idx, "corr_mean_val"]
    ax.annotate(f'{best_val:.3f}', xy=(best_epoch, best_val),
                xytext=(0, 8), textcoords="offset points",
                ha='center', fontsize=8, color=COLORS[label], fontweight='bold')

ax.axhline(0.74, color='#C00000', linestyle='--', linewidth=1, label="Paper's target (0.74)", zorder=0)

ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Validation Pearson r (corr_mean_val)', fontsize=11)
ax.set_title('FingerFlex: Validation Curve by Subject', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', color='#E7E6E6', linewidth=0.6, zorder=0)

plt.tight_layout()
plt.savefig('fingerflex_validation_curves.png', dpi=300, bbox_inches='tight')
print("\n已保存: fingerflex_validation_curves.png")
plt.show()
