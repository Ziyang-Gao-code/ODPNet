# DCGNet: Decoupled Context and Detail-Guided Network for Real-Time Semantic Segmentation of Urban Road Scenes


## Introduction

DCGNet is a real-time semantic segmentation network designed for urban road scene understanding. It adopts a decoupled dual-branch architecture that separately processes context and detail information, enabling efficient and accurate segmentation results.

## Overvie


## Datasets

### Setup Instructions

1. Download the [Cityscapes](https://www.cityscapes-dataset.com/) and [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) datasets
2. Unzip them into the following directories:
   - `data/cityscapes`
   - `data/camvid`
3. Verify that the paths in `data/list` match your dataset image locations

## Results

### Cityscapes Dataset

| Method | Pretrain | Val (% mIOU) | Test (% mIOU) | FPS(torch) |
|:-------|:---------|:-------------|:--------------|:-----------|
| DCGNet-Lite | No | - | - | - |
| DCGNet-Base | No | - | - | - |
| DCGNet-Deep | No | - | - | - |
| DCGNet-Lite | ImageNet | - | - | - |
| DCGNet-Base | ImageNet | - | - | - |
| DCGNet-Deep | ImageNet | - | - | - |

### CamVid Dataset

| Method | Pretrain | Val (% mIOU) | Test (% mIOU) | FPS(torch) |
|:-------|:---------|:-------------|:--------------|:-----------|
| DCGNet-Lite | No | - | - | - |
| DCGNet-Base | No | - | - | - |
| DCGNet-Lite | Cityscapes | - | - | - |
| DCGNet-Base | Cityscapes | - | - | - |

### Qualitative Results

<p align="center">
  <img src="./images/results_cityscapes.png" width="800" alt="Cityscapes Results">
  <br>
  <em>Segmentation results on Cityscapes dataset</em>
</p>

<p align="center">
  <img src="./images/results_camvid.png" width="800" alt="CamVid Results">
  <br>
  <em>Segmentation results on CamVid dataset</em>
</p>

## Key Features

- **Multiple Model Variants**: Lite, Base, and Deep versions to balance accuracy and speed
- **Real-time Inference**: High-speed processing for practical applications
- **Transfer Learning**: Support for ImageNet and Cityscapes pretraining

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dcgnet.git
cd dcgnet

# Install dependencies
pip install -r requirements.txt
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{,
  title={DCGNet: Decoupled Context and Detail-Guided Network for Real-Time Semantic Segmentation of Urban Road Scenes},
  author={},
  journal={},
  year={}
}
```

## Acknowledgments

Our implementation is modified based on [PIDNet-Semantic-Segmentation](https://github.com/XuJiacong/PIDNet) and [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation). Thanks for their nice contribution.
