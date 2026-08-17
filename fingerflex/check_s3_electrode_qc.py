"""
S3电极质检交叉验证：检查FingerFlex用到的sub3原始ECoG数据，
是否也存在Wiener/切换线性模型已经发现的同一批坏道（尤其ch49，train/test振幅比>1100x）

逻辑：对每个通道，分别算train段（前400s）和test段（后200s）信号的振幅统计量（这里用std），
如果某个通道的 train_std / test_std（或反过来）比值远高于其他通道，说明这个通道在
train/test之间的信号特性发生了剧烈跳变，大概率是电极接触不良/断线等物理伪迹，而不是
真实的神经信号变化。

只读取原始.mat文件做诊断，不需要跑prepare_data.ipynb的完整预处理流程，跑得很快。
"""

import numpy as np
import scipy.io
import pathlib

PATH = f"{pathlib.Path().resolve()}/data/pure_data"
SUBJECT = "sub3"

RATIO_FLAG_THRESHOLD = 50  # 振幅比超过这个倍数就标记为可疑（Wiener那边ch49报告的是>1100x，
                             # 这里阈值设低一些，避免漏掉没那么极端但依然异常的通道）


def main():
    print(f"正在诊断 {SUBJECT} 的原始ECoG数据...")

    train_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_comp.mat")["train_data"].astype("float64")  # (time, channels)
    test_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_testlabels.mat")

    # sub_testlabels.mat 只含标签(test_dg)，原始test ECoG在sub_comp.mat里的test_data字段（若存在）
    # 若没有test_data字段，退化为用train内部前后两半做初步比较（并打印提示）
    comp_data = scipy.io.loadmat(f"{PATH}/{SUBJECT}_comp.mat")
    if "test_data" in comp_data:
        test_ecog = comp_data["test_data"].astype("float64")
        source = "sub_comp.mat 的 test_data 字段"
    else:
        print("[提示] sub_comp.mat 里没有找到test_data字段，改用train_data的后1/3和前2/3做对比（近似诊断，非严格train/test振幅比）")
        n = train_data.shape[0]
        test_ecog = train_data[int(n * 2 / 3):]
        train_data = train_data[:int(n * 2 / 3)]
        source = "train_data内部前后切分（近似）"

    n_channels = train_data.shape[1]
    print(f"数据来源: {source}，共 {n_channels} 个通道\n")

    train_std = train_data.std(axis=0)
    test_std = test_ecog.std(axis=0)

    # 振幅比：取大/小，保证比值恒>=1，方便统一判断
    ratio = np.maximum(train_std, test_std) / np.minimum(train_std, test_std)

    order = np.argsort(-ratio)  # 从大到小排序

    print(f"{'通道(0-indexed)':<16}{'train_std':<14}{'test_std':<14}{'ratio':<10}")
    print("-" * 54)
    for idx in order[:15]:  # 只看比值最大的前15个通道
        flag = " <-- 可疑" if ratio[idx] > RATIO_FLAG_THRESHOLD else ""
        print(f"{idx:<16}{train_std[idx]:<14.4f}{test_std[idx]:<14.4f}{ratio[idx]:<10.2f}{flag}")

    flagged = np.where(ratio > RATIO_FLAG_THRESHOLD)[0]
    print(f"\n超过阈值({RATIO_FLAG_THRESHOLD}x)的可疑通道（0-indexed）: {flagged.tolist()}")
    print(f"如果Wiener文档里提到的ch49也在这个列表里（注意0-indexed vs 1-indexed可能差1，"
          f"即Wiener的第49个通道对应这里的索引48），说明FingerFlex用到的原始数据确实受到"
          f"同一物理坏道污染，因为FingerFlex目前没有做通道选择，全部62/64通道都会被送进模型。")


if __name__ == "__main__":
    main()
