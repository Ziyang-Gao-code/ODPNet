# ODPNet: Orthogonal Dual-Path Network for Real-Time Semantic Segmentation of Urban Road Scenes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

DCGNet is a real-time semantic segmentation network designed for urban road scene understanding.
![Overall Architecture](figs/overall.png)

## Overview

Balancing segmentation accuracy with inference speed remains a formidable challenge in real-time autonomous driving scenarios. Existing approaches often suffer from the receptive field rigidity of conventional square pooling and the inherent distribution misalignment between semantic and spatial features.

To address these limitations, we propose the **Decoupled Context and Detail-Guided Network (DCGNet)**. Specifically, we introduce the **Atrous Decoupled Pyramid Pooling Module (ADPPM)** to overcome the redundancy of standard pooling operations. Furthermore, to effectively bridge the gap between high-level semantics and fine-grained details, we propose the **Multi-View Aggregation Module (MVAM)**. Utilizing trans-dimensional attention, MVAM harmonizes the heterogeneous representations, ensuring precise feature recalibration.

## Datasets

### Setup Instructions

1. Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets.
2. Unzip them into the following directories:
   - `data/cityscapes`
   - `data/camvid`
3. Verify that the paths in `data/list` match your dataset image locations.

## Results

### Cityscapes Dataset

| Method | Pretrain | Val (% mIOU) | Test (% mIOU) | FPS (torch) |
|:---:|:---:|:---:|:---:|:---:|
| **DCGNet-Lite** | No | 77.6 | 77.4 | 162.1 |
| **DCGNet-Base** | No | 78.4 | 78.3 | 64.2 |
| **DCGNet-Deep** | No | 78.9 | 78.8 | 48.2 |
| **DCGNet-Lite** | ImageNet | **78.9** | **78.7** | **162.1** |
| **DCGNet-Base** | ImageNet | 79.9 | 79.8 | 64.2 |
| **DCGNet-Deep** | ImageNet | 80.4 | 80.3 | 48.2 |

### CamVid Dataset

| Method | Pretrain | Val (% mIOU) | Test (% mIOU) | FPS (torch) |
|:---:|:---:|:---:|:---:|:---:|
| **DCGNet-Lite** | No | - | 72.9 | 208.4 |
| **DCGNet-Base** | No | - | 74.6 | 152.5 |
| **DCGNet-Lite** | Cityscapes | - | 80.4 | 208.4 |
| **DCGNet-Base** | Cityscapes | - | 82.2 | 152.5 |

## Visualizations

We provide qualitative visualization results to demonstrate the superior performance of **DCGNet** in complex urban driving scenarios.
### Cityscapes Results
![Cityscapes Segmentation](figs/cityscapes_segmentation.png)

### CamVid Results
![CamVid Segmentation](figs/camvid_segmentation.png)

## Key Features

- **Multiple Model Variants**: Lite, Base, and Deep versions to balance accuracy and speed.
- **Real-time Inference**: High-speed processing for practical applications.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
