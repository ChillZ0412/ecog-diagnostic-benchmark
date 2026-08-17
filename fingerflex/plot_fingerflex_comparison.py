import matplotlib.pyplot as plt
import numpy as np

# 三个subject的结果（来自训练/测试日志）
subjects = ['sub1\n(62ch)', 'sub2\n(48ch)', 'sub3\n(64ch)']
val_corr = [0.6841, 0.5887, 0.6684]   # 验证集最优 corr_mean_val
test_corr = [0.660, 0.582, 0.671]     # 测试集 corr_mean（eval()修复后）

navy = '#1F3864'
gray = '#A6A6A6'

x = np.arange(len(subjects))
width = 0.32

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

bars1 = ax.bar(x - width/2, val_corr, width, label='Validation (corr_mean_val)', color=navy)
bars2 = ax.bar(x + width/2, test_corr, width, label='Test (corr_mean)', color=gray)
ax.axhline(0.74, color='#C00000', linestyle='--', linewidth=1, label='Paper reported best (0.74)')

# 在每根柱子上标数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Pearson correlation coefficient (r)', fontsize=11)
ax.set_title('FingerFlex: Validation vs Test Correlation by Subject', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subjects, fontsize=10)
ax.set_ylim(0, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例整体放到图表下方，完全脱离绘图区域，彻底避免和柱子重叠
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=1, frameon=True)

plt.tight_layout()
plt.savefig('fingerflex_subject_comparison.png', dpi=300, bbox_inches='tight')
print("已保存: fingerflex_subject_comparison.png")
plt.show()
