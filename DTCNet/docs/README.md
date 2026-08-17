# DTCNet Regression — Wang et al. (2025) 忠实复现

> Dilated-Transposed Convolution Network 用于 ECoG 手指弯曲回归（BCI Competition IV Dataset 4）。
>
> Wang et al. (2025), *Frontiers in Computational Neuroscience*. DOI: [10.3389/fncom.2025.1627819](https://doi.org/10.3389/fncom.2025.1627819)

## 复现环境

```bash
pip install -r requirements.txt
# GPU (RTX 4060+):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

- Python 3.13, PyTorch 2.6+cu124, numpy, scipy

## 数据准备

1. 下载 BCI Competition IV Dataset 4（[bbci.de](https://www.bbci.de/competition/iv/#dataset4)）
2. 获取**真实测试标签**（竞赛发布包不含，赛后 release）
3. 预处理：

```bash
python preprocess_raw.py
```

输出到 `data_dtcnet/`：`sub{1,2,3}_{train,test}_{spec,finger}.npy`

## 训练

```bash
python run_direct.py --subject all
```

- GPU: ~5 h（RTX 4060）
- 输出: `results_final/`（模型、指标、loss 曲线）

## 预处理（忠实还原原文 2.2.2 节）

| 顺序 | 步骤 |
|:--|:--|
| ① | **Normalization**：per-channel z-score + median removal（原始 ECoG，train 统计量）|
| ② | bandpass 40–300 Hz |
| ③ | notch 60 Hz（工频）|
| ④ | Morlet 小波 → 频谱图（40 频段，n_cycles=7）|
| ⑤ | 时间降采样 1000→100 Hz |

坏电极排除（振幅异常 >10× median，跨方法独立确认）：sub1 ch55、sub2 ch21+ch38、sub3 ch50。

手指数据：dataglove 25 Hz 原始标签，::10 降采样对齐 100 Hz（保持真实读数，不插值）。

## 模型架构

| 组件 | 细节 |
|:--|:--|
| 输入 | Morlet 频谱图 (ch × 40 freq × 256 time @ 100 Hz) |
| Feature Reduction | 1×1 Conv + 3×1 Conv（ch×freq → 48）|
| Encoder | 5 层膨胀卷积（48→64→96→128→128；dilation 1,2,3,1,2）+ AvgPool |
| Decoder | 5 层转置卷积 + skip connections |
| 输出 | 逐时间步轨迹 (batch, 5, 256)——1×1 conv 映射到 5 指 |
| 参数量 | 675K–708K（随电极数变化）|
| 训练 | Adam (lr=8.42e-5, wd=1e-6), MSE+Cosine loss, dropout 0.1 |

> **已知与原文的差异**：Encoder 通道数（我们 [48,64,96,128,128] vs 原文 [64,128,256,512,512]）。
> 原文字称 550–790K 参数，但 [512] 通道算得 ~4M，二者矛盾。我们选择匹配参数量而非通道数，
> 此矛盾已在论文中说明。

## 评估指标

| 指标 | 定义 |
|:--|:--|
| **r** | Pearson 相关系数（逐指，跨时间）|
| **official_r** | 剔除无名指（Ring）的 4 指均值（主口径）|
| **avg_r** | 5 指均值（辅助）|
| **R²** | 决定系数 1 − SSE/SST |
| **calibrated R²** | 逐指 r²（最优仿射重标定后的 R² 上限）|
| **calibration gap** | r² − R²（可恢复的尺度/偏移误差）|
| **MAE** | 平均绝对误差（dataglove 原始单位）|

评估口径：逐时间步轨迹输出，取每个滑动窗口的最后时间步拼接（stride=1 标准做法）。

## 统计检验对齐

```bash
python export_results.py   # 生成 benchmark 表 (Markdown + CSV)
python extract_stats.py    # 提取 15 个原始 r 值给组员做 Wilcoxon
```

## 文件

```
model.py              — DTCNet 架构
train.py              — Dataset、训练循环、评估
run_direct.py         — 启动器（3 subject，自动跳过已完成）
preprocess_raw.py     — 预处理（per-channel 归一化 + Morlet）
gen_ablation_data.py  — 消融 B 数据生成（频率分辨率）
export_results.py     — benchmark 表导出
extract_stats.py      — 15 个 r 值提取（统计检验对齐）
```

## Citation

```bibtex
@article{wang2025dtcnet,
  title={DTCNet: finger flexion decoding with three-dimensional ECoG data},
  author={Wang, Y. and others},
  journal={Frontiers in Computational Neuroscience},
  year={2025},
  doi={10.3389/fncom.2025.1627819}
}
```
