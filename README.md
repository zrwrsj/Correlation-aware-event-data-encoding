# Correlation-Aware Event Data Encoding via Temporal Segmentation

本项目是论文 **Correlation-Aware Event Data Encoding via Temporal Segmentation** 的分类任务代码。项目面向事件相机数据，研究如何在事件流转换为帧的过程中保留更多时间信息，并减少不同语义通道之间的特征干扰。

论文提出两部分方法：

1. **STS&DPA（Subsection-Time-Surface and Different-Polarity-Apart）**：分别统计正、负极性的事件数量，并将事件流按事件索引切分为 `K` 个子段。在每个子段内生成正、负极性的时间表面，再对同一极性的 `K` 个时间通道分别进行 PCA 降维，最终得到 2 个事件数量通道和 2 个时间通道，共 4 个通道。
2. **CGC（Correlation-based Group Convolution）**：根据四个通道的相关性构造四个分支：数量通道对、时间通道对、负极性通道对、正极性通道对。每个分支独立卷积后进行拼接，以减小弱相关通道之间的特征混淆。

## 论文结果

论文中使用 Spiking-VGG11、QKformer 和 SpikeYOLO，在 DVS-Gesture、CIFAR10-DVS、N-Caltech101 以及 GEN1 上进行了实验。下表列出论文报告的代表性结果；这些数值是论文实验结果，不代表本仓库已经自动下载数据并完成复现实验。

| 数据集 | 模型 | 编码 | 指标 |
| --- | --- | --- | ---: |
| DVS-Gesture | CGC + Spiking-VGG11 | STS&DPA | 98.11% |
| CIFAR10-DVS | CGC + Spiking-VGG11 | STS&DPA | 80.43% |
| N-Caltech101 | CGC + Spiking-VGG11 | STS&DPA | 82.61% |
| DVS-Gesture | CGC + QKformer | STS&DPA | 98.48% |
| CIFAR10-DVS | CGC + QKformer | STS&DPA | 84.77% |
| N-Caltech101 | CGC + QKformer | STS&DPA | 83.31% |
| GEN1 | CGC + SpikeYOLO | STS&DPA | 69.72% mAP@50，41.65% mAP@50:95 |

论文的消融实验表明，DVS-Gesture 和 CIFAR10-DVS 通常使用 4 个子段，N-Caltech101 使用 2 个子段，GEN1 使用 8 个子段。代码中 `segments` 是该参数。

## 代码与论文的对应关系

| 文件 | 作用 |
| --- | --- |
| `spikingjelly_revise/datasets/__init__.py` | 事件流切分、事件帧生成、DPA/TS/STS 编码和 PCA 降维 |
| `spikingjelly_revise/datasets/cifar10_dvs.py` | CIFAR10-DVS 数据集读取 |
| `spikingjelly_revise/datasets/dvs128_gesture.py` | DVS-Gesture 数据集读取 |
| `spikingjelly_revise/datasets/n_caltech101.py` | N-Caltech101 数据集读取 |
| `dataset.py` | 数据集封装和 `DataLoader` 构建 |
| `transform_DVS.py` | 事件帧的列表化、张量化、缩放和数据增强 |
| `model.py` | `VGG11`、`CGCVGG11` 和 `Learn_CGCVGG11` |
| `neuron.py` | LIF 神经元和 Rectangle surrogate gradient |
| `train.py` | SNN 训练、验证、混合精度和 checkpoint 保存 |
| `spikingjelly_revise/` | 项目内置的 SpikingJelly 代码 |

## 编码方法

`spikingjelly_revise/datasets/__init__.py` 中的 `method` 参数含义如下：

| `method` | 输入通道 | 含义 |
| --- | ---: | --- |
| `method0` | 2 | 正、负极性事件数量（DPA） |
| `method1` | 4 | DPA 数量通道 + 全时间范围时间表面 |
| `method2` | 4 | DPA 数量通道 + 分段时间表面，并按极性使用 PCA 降维；对应论文 STS&DPA |
| `method3` | `2 + 2 * segments` | 保留每个子段的正、负时间表面，不在预处理阶段做 PCA；可配合 `Learn_CGCVGG11` 学习降维 |

代码中的 `CGCVGG11` 接收 4 通道输入，适合论文中的 `method2 + CGC` 路径。`Learn_CGCVGG11` 接收默认 `segments=4` 产生的 10 通道输入，并在网络内部先学习压缩时间通道，再执行 CGC。因此，`method3` 的通道数变化时，需要同步修改 `Learn_CGCVGG11` 的输入切片和第一层配置。

## 环境安装

建议使用 Python 3.9 或更高版本，并根据本机 CUDA 版本安装匹配的 PyTorch。随后安装其余依赖：

```bash
python -m pip install numpy scipy scikit-learn matplotlib tqdm torchsummary torchtoolbox
```

如果尚未安装 PyTorch，请参考 <https://pytorch.org/> 安装与 CUDA 匹配的版本，例如：

```bash
python -m pip install torch torchvision torchaudio
```

项目没有单独的 `requirements.txt`；`spikingjelly_revise` 已随项目提供，不需要另外安装同名包。

## 数据集准备

`train.py` 当前主流程调用的是 CIFAR10-DVS：

```text
默认路径：../CIFAR10_DVS
```

也可以通过 `--data_dir` 指定路径。数据集首次使用时，SpikingJelly 代码会根据 `data_type='frame'`、`T_data`、`method` 和 `segments` 生成缓存的事件帧目录，例如：

```text
CIFAR10_DVS/T8_method2_s4/
```

代码中还提供了 `DVS_Gesture` 和 `Ncaltech101` 封装，但 `train.py` 的当前 `main()` 默认只实例化 `cifar10_DVS()`；如果要训练另外两个数据集，需要在 `train.py` 中切换数据集构建函数和类别数。

N-Caltech101 的增强数据封装期望目录结构包含：

```text
<data_dir>/T8_method2_s2/<class_name>/*.npz
```

每个 `.npz` 文件需要包含名为 `frames` 的数组。

## 训练

先进入项目根目录：

```bash
cd "Correlation-Aware Event Data Encoding via Temporal Segmentation"
```

论文 STS&DPA + CGC 的 CIFAR10-DVS 配置可以使用：

```bash
python train.py --data_dir ../CIFAR10_DVS -encode_method method2 -model CGCVGG11 -segments 4 -T 16 -T_data 16 -b 16 -epochs 300 -lr 0.01 -device cuda:0 -outdir ./logs/cifar10_sts_cgc
```

默认脚本配置是可学习降维路径：

```bash
python train.py --data_dir ../CIFAR10_DVS -encode_method method3 -model Learn_CGCVGG11 -segments 4 -T 6 -T_data 8 -b 16 -epochs 300 -device cuda:0 -outdir ./logs/cifar10_learnable
```

主要参数：

```text
--data_dir       数据集目录
-encode_method   method0 / method1 / method2 / method3
-model           VGG11 / CGCVGG11 / Learn_CGCVGG11
-segments        每个事件帧内的事件子段数
-T               训练时随机采样的 SNN 时间步数
-T_data          数据集转换得到的帧数
-b               batch size
-epochs          训练轮数
-device          例如 cuda:0 或 cpu
-outdir          日志和模型输出目录
-resume          是否从 checkpoint 恢复
```

训练输出包括 `args.txt`、`process.txt`、`best_model.pth`、`checkpoint_latest.pth` 以及损失和准确率曲线。训练脚本中的恢复路径目前写死为 `logs_0/train9/checkpoint_latest.pth`，使用 `-resume True` 前请确认该路径存在，或按实际实验目录修改代码。

## 推理和模型调用

模型输入是四维张量：

```text
[batch, channels, height, width]
```

例如 `method2` 与 `CGCVGG11` 的输入形状为 `[B, 4, 64, 64]`；`method3` 与 `Learn_CGCVGG11` 在 `segments=4` 时的输入形状为 `[B, 10, 64, 64]`。SNN 的膜电位状态应在每个样本或序列处理结束后重置，训练代码使用 `functional.reset_net(model)` 完成该操作。

## 当前实现注意事项

- 当前工作区中的 `train.py` 和 `transform_DVS.py` 含有 Git 合并冲突标记（`<<<<<<<`、`=======`、`>>>>>>>`）。在清理冲突并确认保留的代码版本之前，训练命令不能直接执行；这是工作区状态问题，不是本文方法的一部分。
- 论文中的 GEN1 检测实验依赖 SpikeYOLO，SpikeYOLO 训练代码不在本仓库中；本仓库主要覆盖事件帧编码和分类模型。
- `train.py` 当前只在主流程中使用 CIFAR10-DVS，DVS-Gesture 和 N-Caltech101 的数据封装可从 `dataset.py` 调用。
- `method2` 的 PCA 是按单个事件帧、按极性独立计算的确定性降维，不需要额外训练 PCA 参数。
- 事件帧尺寸在 `dataset.py` 中缩放为 `64 x 64`；原始事件传感器分辨率通常为 `128 x 128`。
- 当前 `procession_DVS` 和 `procession_DVS_test` 内部将输入直接移动到 `cuda`，因此使用 CPU 前需要先把代码中的 `.to('cuda')` 改为 `.to(args.device)` 或等效写法。
- 训练需要 GPU 时，请确认 `-device`、PyTorch CUDA 版本和显存容量匹配；CPU 运行通常需要显著更长时间。

## 引用

如果本项目或 STS&DPA/CGC 方法对你的研究有帮助，请引用对应论文：

```bibtex
@unpublished{xia_correlation_aware_event_encoding,
  title  = {Correlation-Aware Event Data Encoding via Temporal Segmentation},
  author = {Xia, Qisheng and Luo, Hui and Chen, Faling and Luo, Haibo},
  note   = {Use the final publication metadata when available}
}
```

论文原稿中的准确期刊、卷期和 DOI 信息以 `reference.bib` 或最终发表版本为准。
