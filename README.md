# Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Task](https://img.shields.io/badge/Task-Speech%20Emotion%20Recognition-lightgrey)]()
[![Interspeech](https://img.shields.io/badge/Interspeech-2026-8A2BE2)]()

Official implementation of **Geometric Second-Order Feature Correlation
Learning for Self-Supervised Speech Emotion Recognition**.

This work has been accepted by **Interspeech 2026**.

SOC encodes frozen SSL frame-level representations as covariance descriptors and uses Log-Euclidean mapping to bridge the mismatch between SPD manifold-valued statistics and Euclidean classifiers.
<p align="center">
  <img src="assets/method.png" alt="SOC method overview" width="900">
</p>

## Highlights

- Uses covariance descriptors, SPD geometry, and Log-Euclidean vectorization.
- Plug-in PyTorch `SOCPooling` module for frozen SSL speech representations.
- Lightweight downstream aggregation layer without SSL pre-training or backbone fine-tuning.

## Environment

The experiments in the paper were conducted on an **NVIDIA RTX 4090** GPU.

Tested environment:

- Ubuntu 20.04
- Python 3.9
- CUDA 11.8
- PyTorch 2.0+

## Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/secret-code-source/SOC.git
cd SOC
pip install -r requirements.txt
```

Run a CPU smoke test with synthetic `.npy` features:

```bash
python scripts/smoke_test.py
```

After preparing metadata and pre-extracted SSL features, train SOC with:

```bash
python scripts/train.py --config configs/ravdess_hubert.yaml
python scripts/train.py --config configs/esd_hubert.yaml
```

Run all commands from the repository root.

## Use SOC in Your Model

```python
import torch
from soc_ser import SOCPooling

x = torch.randn(8, 120, 768)  # [batch, frames, ssl_dim]
pool = SOCPooling(in_dim=768, spd_dim=24)
z = pool(x)

print(z.shape)  # [8, 300]
```

## Data

For data preprocessing, SSL feature extraction, and speaker-independent partitioning, please refer to
the [EmoBox repository](https://github.com/emo-box/emobox).

This training pipeline expects frozen SSL features to be pre-extracted and
stored as `.npy` files whose names match the metadata `key` field. It does not
perform online SSL feature extraction during training.

See `data/README.md` for the expected layout.

## Feature Extraction

```bash
python scripts/extract_features.py --config configs/ravdess_hubert.yaml
```

## Training

```bash
python scripts/train.py --config configs/ravdess_hubert.yaml
python scripts/train.py --config configs/esd_hubert.yaml
```

The default RAVDESS config uses five speaker-independent folds.

## Optional Visualization

We provide an optional t-SNE utility for visualizing SOC embeddings from a
trained checkpoint. See `docs/tsne_visualization.md` for usage.

## Citation

If you use this repository, please cite our Interspeech 2026 paper:

```bibtex
@inproceedings{soc_interspeech2026,
  title     = {Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition},
  author    = {Shuanglin Li, Ruxiao Qian, Siyang Song},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
