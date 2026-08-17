# DTC 交付包索引（供组员整合到论文）

> 本文档列出 DTCNet 部分的全部产物，方便组员整合到统一 benchmark 论文。
> DTC 部分的论文 draft（methodology/result/discussion）见 `DTC_Paper_Draft.md`。
> 标准声明见 `Paper_Declarations.md`。
> 本索引位于 DTCNet 文件夹的 `docs/` 下，以下路径均为 DTCNet 文件夹内的相对路径。

---

## 一、代码文件（`code/`，已锁定，全部 SCI 规范命名）

| 文件 | 用途 |
|:--|:--|
| `model.py` | DTCNet 架构（逐时间步轨迹输出 / 单点输出两种模式）|
| `train.py` | 数据加载、训练循环、评估（含 MSE+Cosine loss）|
| `preprocess_raw.py` | 预处理（per-channel 归一化 → bandpass → Morlet → 降采样）|
| `gen_ablation_data.py` | 频率分辨率消融数据生成（1/10/20 band）|
| `run_direct.py` | 训练启动器（支持 --output/--data-root/--out 参数）|
| `run_ablation.sh` | 消融串联脚本（频率分辨率 + 输出层）|
| `export_results.py` | 4 张 15 格 benchmark 表导出 |
| `extract_stats.py` | 15 个原始 r 值提取（给 Wilcoxon 检验）|
| `analyze_supplementary.py` | 逐指相关性 + 训练曲线分析 |
| `draw_figures.py` | 3 张图生成（架构图/热力图/消融趋势）|
| `verify_25hz.py` | 25Hz 降采样验证（与 FingerFlex 口径对齐）|
| `gen_keepbad_data.py` | 生成「保留坏电极」数据（控制变量验证条件 B）|
| `requirements.txt` | 依赖（torch/numpy/scipy）|

---

## 二、实验结果（`results/`）

| 路径 | 内容 |
|:--|:--|
| `results/main/` | 主实验（轨迹+per-channel）sub1=0.4554, sub2=0.4674, sub3=0.6410 |
| `results/ablation_freq_1band/` | 频率分辨率消融：1 band（无 Morlet）mean = 0.157 |
| `results/ablation_freq_10band/` | 频率分辨率消融：10 band mean = 0.492 |
| `results/ablation_freq_20band/` | 频率分辨率消融：20 band mean = 0.537（峰值）|
| `results/ablation_single/` | 输出层消融（单点输出）sub1=0.296, sub2=0.470, sub3=0.643 |
| `benchmark/` | 4 张 15 格表：r / R² / calibrated R² / gap（md+csv 共 10 文件）|

> 注：每目录含 `results.json`（完整指标）+ `sub{1,2,3}_loss.npy`（训练曲线）。
> 大文件 checkpoint（.pt）未打包，可用 `code/` 内脚本复现。

---

## 三、图表（`figures/`）

| 文件 | 用途 | 对应论文部分 |
|:--|:--|:--|
| `fig_architecture.png` | DTCNet 架构示意图 | Methodology |
| `fig_finger_correlation.png` | 5 指真实运动相关矩阵（热力图）| Results 逐指分析 |
| `fig_freq_ablation.png` | 频率分辨率消融趋势（1→10→20→40 band）| Supplementary |
| `training_curves.png` | 3 subject 训练 loss 曲线 | Supplementary |
| `finger_correlation.csv` | 逐指相关矩阵数值 | Supplementary |

> 注：组员可按需选择/修改图内文字风格（中文/英文）。

---

## 四、DTC 关键数据速查（论文写作直接引用）

### 主实验（official_r 剔除无名指）
| Subject | Thumb | Index | Middle | Ring | Little | avg_r | official_r |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 1 | 0.55 | 0.63 | 0.29 | 0.51 | 0.36 | 0.47 | 0.46 |
| 2 | 0.59 | 0.48 | 0.36 | 0.54 | 0.45 | 0.48 | 0.47 |
| 3 | 0.79 | 0.61 | 0.50 | 0.68 | 0.67 | 0.65 | 0.64 |
| Mean | 0.64 | 0.57 | 0.38 | 0.57 | 0.49 | 0.53 | 0.52 |

### 频率分辨率消融（mean official_r）
- 1 band（无 Morlet）：0.157
- 10 band：0.492
- **20 band：0.537（峰值，反直觉发现）**
- 40 band（主实验）：0.521

### 输出层消融（轨迹 vs 单点）
| Subject | 轨迹输出 | 单点输出 | 差异 |
|:--|:--|:--|:--|
| 1 | 0.455 | 0.296 | +0.16 |
| 2 | 0.467 | 0.470 | 0.00 |
| 3 | 0.641 | 0.643 | 0.00 |
| Mean | 0.521 | 0.470 | +0.05 |

> 关键：逐时间步监督的优势**在不同受试者间并不一致**（sub1 明显，sub2/sub3 相当），
> 论文中以"观察"呈现（非"结论"），详见 DTC_Paper_Draft.md。

### 25Hz 降采样验证（与 FingerFlex 对齐）
- 100Hz block_average → 25Hz，official_r 差异最大 0.0043（< 0.03 阈值）
- 结论：DTC 的 100Hz 与 FingerFlex 的 25Hz 评估口径**等价**，可直接对齐

### 校准指标（校准 gap 解释负 R²）
| Subject | R² | calibrated R² | gap |
|:--|:--|:--|:--|
| 1 | −0.245 | 0.226 | 0.472 |
| 2 | −0.494 | 0.225 | 0.719 |
| 3 | 0.191 | 0.422 | 0.231 |

### 逐指真实运动耦合（推翻"耦合导致难解"假设）
- Ring-Little 0.321（最强耦合），Thumb 最独立
- 中指耦合很弱（0.02-0.20）但解码最差 → 耦合不是解码难度主因

### 15 个 r 值（给组员 Wilcoxon）
```
r15 = [0.5455, 0.6282, 0.2896, 0.5065, 0.3584,    # sub1
       0.5893, 0.4751, 0.3604, 0.5379, 0.4447,    # sub2
       0.7871, 0.6117, 0.5003, 0.6774, 0.6651]    # sub3
```

---

## 五、复现修正（Implementation Notes，论文 Methodology）

1. **归一化**：per-channel z-score + median removal，在**原始 ECoG**（bandpass 前）进行
2. **输出层**：逐时间步轨迹（sequence-to-sequence），而非单点
3. **坏电极**：sub1 ch55、sub2 ch21/ch38、sub3 ch50（跨方法独立确认）。控制变量验证排除 vs 保留差异约 0.004，可忽略
4. **参数量矛盾**：编码器通道 [48,64,96,128,128] 是为匹配原文报道的 550-790K 参数

---

## 六、关键科学发现（Discussion 用）

1. **Morlet 频谱关键**：1 band 无 Morlet 时性能崩溃（0.157），证明频域分解对 ECoG 必要
2. **20 band 是 sweet spot**：40 band 反而略低于 20 band（0.521 vs 0.537），反直觉但稳定
3. **逐时间步监督的作用受试者依赖**：sub1 显著（+0.16），sub2/sub3 无差异（观察，非结论）
4. **手指耦合不是解码难度主因**：中指耦合弱但解码最差，拇指独立但解码最好
5. **校准 gap 解释负 R²**：模型方向学对了，仅尺度/偏移失配（gap 0.47/0.72）
6. **坏电极影响可忽略**：排除 vs 保留差异约 0.004（控制变量验证），之前的"略降"是随机性

---

## 七、完成状态与后续待办

| 项 | 状态 |
|:--|:--|
| 输出层消融 | ✅ 已完成 |
| 25Hz 降采样验证 | ✅ 已完成（差异 < 0.03，等价）|
| 控制变量重跑（坏电极）| ✅ 已完成（排除 vs 保留差异约 0.004，可忽略）|
| 回归侧 Wilcoxon 统计检验 | 📋 组员跑（15 个 r 值已备）|

---

## 八、给组员的整合建议

1. **DTC 论文 draft**（`DTC_Paper_Draft.md`）按 benchmark 方法描述风格写，可直接整合到 Methodology 第 3.X 节
2. **4 张 benchmark 表** 已按 FingerFlex 对齐（Thumb/Index/Middle/Ring/Little + official_r + avg_r），可直接并入统一结果表
3. **统计检验**用 15 个 r 值做 Wilcoxon 配对，Bonferroni α = 0.0083（4 方法 × 6 比较）
4. **消融**：正文只放"必要性说明"（一句结论），详细消融放 Supplementary
5. **FingerFlex 数字**：0.655 作基准，0.7357 放 footnote（sub3 修复后局部）
6. **标准声明**见 `Paper_Declarations.md`，组员整合到对应章节
7. **图**见 `figures/`，组员按需选择/修改
8. **评估口径**：DTC 100Hz 与 FingerFlex 25Hz 已验证等价（差异 < 0.03）

---

*本索引是 DTC 交付包的入口。如有补充或修改，直接编辑对应文件即可。*
