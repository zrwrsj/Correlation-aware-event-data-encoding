# Correlation-Aware Event Data Encoding via Temporal Segmentation

This repository contains the classification code for the paper *Correlation-Aware Event Data Encoding via Temporal Segmentation*. It studies event-camera representations that preserve temporal information while reducing interference between weakly related channels.

The paper proposes two components:

1. **STS&DPA (Subsection-Time-Surface and Different-Polarity-Apart)**: positive and negative event counts are accumulated separately. The event stream is split into `K` segments by event index. Each segment produces positive and negative time surfaces, and PCA reduces the `K` temporal channels for each polarity to one channel. The final representation contains two event-count channels and two temporal channels.
2. **CGC (Correlation-based Group Convolution)**: the four channels are processed through four correlation-motivated branches: count channels, temporal channels, negative-polarity channels, and positive-polarity channels. The branch outputs are concatenated before the main network.

## Reported Results

The paper evaluates Spiking-VGG11, QKformer, and SpikeYOLO on DVS-Gesture, CIFAR10-DVS, N-Caltech101, and GEN1. The values below are reported paper results; they do not imply that the datasets have been downloaded or that the experiments have already been reproduced in this repository.

| Dataset | Model | Encoding | Result |
| --- | --- | --- | ---: |
| DVS-Gesture | CGC + Spiking-VGG11 | STS&DPA | 98.11% |
| CIFAR10-DVS | CGC + Spiking-VGG11 | STS&DPA | 80.43% |
| N-Caltech101 | CGC + Spiking-VGG11 | STS&DPA | 82.61% |
| DVS-Gesture | CGC + QKformer | STS&DPA | 98.48% |
| CIFAR10-DVS | CGC + QKformer | STS&DPA | 84.77% |
| N-Caltech101 | CGC + QKformer | STS&DPA | 83.31% |
| GEN1 | CGC + SpikeYOLO | STS&DPA | 69.72% mAP@50; 41.65% mAP@50:95 |

The paper's ablation study generally uses 4 segments for DVS-Gesture and CIFAR10-DVS, 2 segments for N-Caltech101, and 8 segments for GEN1. This value is controlled by the `segments` argument in the code.

## Repository Map

| File or directory | Purpose |
| --- | --- |
| `spikingjelly_revise/datasets/__init__.py` | Event slicing, event-frame generation, DPA/TS/STS encoding, and PCA reduction |
| `spikingjelly_revise/datasets/cifar10_dvs.py` | CIFAR10-DVS loader |
| `spikingjelly_revise/datasets/dvs128_gesture.py` | DVS-Gesture loader |
| `spikingjelly_revise/datasets/n_caltech101.py` | N-Caltech101 loader |
| `dataset.py` | Dataset wrappers and `DataLoader` construction |
| `transform_DVS.py` | Event-frame conversion, tensor conversion, resizing, and augmentation |
| `model.py` | `VGG11`, `CGCVGG11`, and `Learn_CGCVGG11` |
| `neuron.py` | LIF neuron and Rectangle surrogate gradient |
| `train.py` | SNN training, evaluation, mixed precision, and checkpoint saving |
| `spikingjelly_revise/` | Bundled SpikingJelly implementation used by the project |

## Encoding Methods

The `method` argument in `spikingjelly_revise/datasets/__init__.py` has the following meanings:

| `method` | Channels | Description |
| --- | ---: | --- |
| `method0` | 2 | Positive and negative event counts (DPA) |
| `method1` | 4 | DPA count channels plus full-interval time surfaces |
| `method2` | 4 | DPA count channels plus segmented time surfaces reduced by per-polarity PCA; this corresponds to the paper's STS&DPA representation |
| `method3` | `2 + 2 * segments` | Keeps every positive and negative temporal segment; reduction is learned by `Learn_CGCVGG11` instead of performed during preprocessing |

`CGCVGG11` expects four input channels and is the direct `method2 + CGC` path from the paper. With the default `segments=4`, `Learn_CGCVGG11` expects 10 input channels, learns a temporal-channel reduction, and then applies CGC. Changing the number of segments for `method3` requires matching changes to the input slicing and first-layer configuration in `Learn_CGCVGG11`.

## Installation

Use Python 3.9 or newer and install a PyTorch build compatible with your CUDA version. Install the remaining dependencies with:

```bash
python -m pip install numpy scipy scikit-learn matplotlib tqdm torchsummary torchtoolbox
```

If PyTorch is not installed, follow the instructions at <https://pytorch.org/>. A basic CPU installation is:

```bash
python -m pip install torch torchvision torchaudio
```

There is no separate `requirements.txt` in this repository. The required SpikingJelly code is included under `spikingjelly_revise/`.

## Dataset Preparation

The current `main()` function in `train.py` uses CIFAR10-DVS by default:

```text
Default path: ../CIFAR10_DVS
```

Use `--data_dir` to provide another path. On first use, the bundled dataset code generates cached event frames according to `data_type='frame'`, `T_data`, `method`, and `segments`. For example:

```text
CIFAR10_DVS/T8_method2_s4/
```

The repository also provides `DVS_Gesture` and `Ncaltech101` wrappers in `dataset.py`, but the current training entry point instantiates `cifar10_DVS()` only. To train on the other datasets, switch the dataset builder and class count in `train.py`.

The N-Caltech101 augmentation wrapper expects a directory such as:

```text
<data_dir>/T8_method2_s2/<class_name>/*.npz
```

Each `.npz` file must contain an array named `frames`.

## Training

From the repository root:

```bash
cd "Correlation-Aware Event Data Encoding via Temporal Segmentation"
```

The paper-style STS&DPA + CGC CIFAR10-DVS configuration is:

```bash
python train.py --data_dir ../CIFAR10_DVS -encode_method method2 -model CGCVGG11 -segments 4 -T 16 -T_data 16 -b 16 -epochs 300 -lr 0.01 -device cuda:0 -outdir ./logs/cifar10_sts_cgc
```

The script's default learnable-reduction path is:

```bash
python train.py --data_dir ../CIFAR10_DVS -encode_method method3 -model Learn_CGCVGG11 -segments 4 -T 6 -T_data 8 -b 16 -epochs 300 -device cuda:0 -outdir ./logs/cifar10_learnable
```

Important arguments:

```text
--data_dir       Dataset directory
-encode_method   method0 / method1 / method2 / method3
-model           VGG11 / CGCVGG11 / Learn_CGCVGG11
-segments        Number of event subdivisions per event frame
-T               Number of SNN time steps sampled during training
-T_data          Number of frames generated by the dataset converter
-b               Batch size
-epochs          Number of training epochs
-device          For example, cuda:0 or cpu
-outdir          Directory for logs and model outputs
-resume          Whether to resume from a checkpoint
```

Training writes `args.txt`, `process.txt`, `best_model.pth`, `checkpoint_latest.pth`, and loss/accuracy plots. The resume path in the current script is hard-coded as `logs_0/train9/checkpoint_latest.pth`; verify or update this path before using `-resume True`.

## Inference and Model Input

The models expect a four-dimensional tensor:

```text
[batch, channels, height, width]
```

For example, `method2` with `CGCVGG11` uses `[B, 4, 64, 64]`. With `segments=4`, `method3` with `Learn_CGCVGG11` uses `[B, 10, 64, 64]`. Reset the SNN membrane states after each sample or sequence; the training code uses `functional.reset_net(model)`.

## Implementation Notes

- The current working copy contains unresolved Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in `train.py` and `transform_DVS.py`. Training cannot run until those conflicts are resolved and the intended code version is retained.
- The GEN1 detection experiment in the paper depends on SpikeYOLO, whose training code is not included here. This repository primarily covers event-frame encoding and classification models.
- The current `train.py` main flow uses CIFAR10-DVS only. DVS-Gesture and N-Caltech101 wrappers are available from `dataset.py`.
- `method2` performs deterministic PCA reduction independently for each event frame and polarity; no separate PCA training step is required.
- `dataset.py` resizes event frames to `64 x 64`; the original event sensor resolution is commonly `128 x 128`.
- `procession_DVS` and `procession_DVS_test` currently move inputs directly to `cuda`. To run on CPU, replace `.to('cuda')` with `.to(args.device)` or an equivalent device-aware expression.
- For GPU training, ensure that `-device`, the PyTorch CUDA build, and available GPU memory are compatible.

## Citation

If this repository or the STS&DPA/CGC method is useful in your research, cite the paper. The manuscript does not yet provide final publication metadata, so update this entry with the final journal, year, volume, pages, and DOI when available:

```bibtex
@unpublished{xia_correlation_aware_event_encoding,
  title  = {Correlation-Aware Event Data Encoding via Temporal Segmentation},
  author = {Xia, Qisheng and Luo, Hui and Chen, Faling and Luo, Haibo},
  note   = {Use the final publication metadata when available}
}
```
