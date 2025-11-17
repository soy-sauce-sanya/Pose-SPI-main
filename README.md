# Pose-SPI: Image-free Single-Pixel Keypoint Detection Privacy-Preserving Human Pose Estimation using Single-Pixel Imaging

An official PyTorch implementation of image-free human keypoint detection using single-pixel imaging (SPI) technology. This approach enables privacy-preserving pose estimation by extracting keypoints directly from single-pixel measurements without reconstructing the full image.

## 📄 Paper

**"Image-free single-pixel keypoint detection for privacy preserving human pose estimation"**

This repository contains the implementation of the method described in the paper, demonstrating how single-pixel cameras can be used for human pose estimation while preserving privacy.

## 🔒 Privacy-Preserving Approach

Traditional computer vision systems capture full images, which can reveal sensitive information about individuals and their environment. This project uses **single-pixel imaging** to address privacy concerns:

### How Single-Pixel Imaging Works

```
Traditional Camera:              Single-Pixel Camera:
┌─────────────────┐             ┌─────────────────┐
│ Millions of     │             │ Structured      │
│ pixels capture  │             │ illumination    │
│ full image      │             │ patterns        │
│                 │             │ (e.g., Hadamard)│
└─────────────────┘             └─────────────────┘
        ↓                               ↓
┌─────────────────┐             ┌─────────────────┐
│ Full RGB image  │             │ Single detector │
│ 📸 [Privacy     │             │ measures total  │
│    concerns]    │             │ light intensity │
└─────────────────┘             └─────────────────┘
        ↓                               ↓
┌─────────────────┐             ┌─────────────────┐
│ Image processing│             │ Direct keypoint │
│ → Extract pose  │             │ extraction from │
│                 │             │ measurements    │
└─────────────────┘             └─────────────────┘
        ↓                               ↓
    Pose data                   Pose data ✓ Privacy!
```

### Key Privacy Benefits

- **No image reconstruction**: Keypoints are extracted directly from measurements
- **Minimal visual information**: Single-pixel detector captures aggregate light, not spatial details
- **Inherent privacy**: Cannot recover faces, identities, or environmental details
- **Hardware-level protection**: Privacy preserved at the sensor level, not just software

### Applications

- **Healthcare monitoring**: Patient pose tracking without capturing identifiable images
- **Smart homes**: Activity recognition while preserving occupant privacy
- **Workplace safety**: Posture monitoring without surveillance concerns
- **Elderly care**: Fall detection without invasive imaging
- **Sports analytics**: Performance analysis without recording athletes' faces

## Features

- **Two Model Architectures**:
  - Simple CNN-based detector optimized for single-pixel measurements
  - EfficientNet-based model leveraging pretrained features (adapted for SPI)
- **Hadamard Pattern Simulation**: CustomConv layer mimics Hadamard sampling patterns
- **Flexible Measurement Counts**: Adjustable number of measurements via `reduce` parameter
- **COCO Format Support**: Compatible with standard keypoint annotations
- **Real-time Monitoring**: TensorBoard integration for training visualization
- **Standard Evaluation**: COCO evaluation metrics for keypoint detection

## Table of Contents

- [How It Works](#how-it-works)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Understanding Single-Pixel Imaging](#understanding-single-pixel-imaging)
- [Important Notes](#important-notes)
- [Citation](#citation)

## How It Works

### Single-Pixel Measurement Pipeline

1. **Pattern Generation**: Scene is illuminated with structured patterns (Hadamard, random, etc.)
2. **Single-Pixel Detection**: Total reflected/transmitted light measured by single detector
3. **Multiple Measurements**: Process repeated with different patterns (e.g., 64×64 = 4096 measurements)
4. **Direct Keypoint Extraction**: Neural network extracts keypoints from measurement vector
5. **Privacy Preserved**: No intermediate image reconstruction needed

### Architecture Overview

```
Single-Pixel Measurements (64×64 patterns)
    ↓
CustomConv (Hadamard Product Simulation)
    ↓
Encoder (Extract features from measurements)
    ↓
FSRCNN (Upsample feature representations)
    ↓
┌─────────────────────────────────────┐
│  Model 1: SimpleCNN                 │
│  (Lightweight detector)             │
│                                     │
│  Model 2: EfficientNet + RCNN       │
│  (Enhanced feature extraction)      │
└─────────────────────────────────────┘
    ↓
Output: Keypoints (N × 3)
[x, y, visibility] for each keypoint
```

## Installation

### Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Pose-SPI-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. If TensorBoard is not installed:
```bash
pip install tensorboard
```

### Dependencies

- PyTorch 1.12.1
- TorchVision 0.15.2
- NumPy 1.23.3
- Pillow 10.0.0
- pycocotools 2.0.6
- TensorBoard

## Dataset

### Toy Dataset

The repository includes a toy dataset for experimentation. Note that for this single-pixel imaging approach:

- **Input Format**: RGB images (64×64) are used to **simulate** single-pixel measurements
- **In Practice**: Real single-pixel cameras would provide measurement vectors directly
- **Simulation**: The training pipeline converts images to grayscale and processes them as if they were single-pixel measurements
- **Annotations**: COCO format JSON with 14 keypoints per person

**Dataset location**:
- Images: `toydataset/images/train2017/` (100 sample images)
- Annotations: `toydataset/annotations/annotations.json`

### Using Custom Datasets

For simulated single-pixel imaging:

1. Organize 64×64 images representing measurement patterns
2. Create COCO-format annotations with keypoint ground truth
3. The model will treat grayscale intensities as single-pixel measurements
4. Pass paths via command-line arguments

**For real single-pixel data**:
- Modify data loader to accept measurement vectors directly
- Reshape measurements to (batch, 1, 64, 64) format
- Follow the same training procedure

## Training

### Simple Keypoint Detector

Train the lightweight model optimized for single-pixel measurements:

```bash
python train.py
```

**Default settings**: 1000 epochs, batch size 4, learning rate 0.0001, 13 keypoints

### EfficientNet-based Model

Train with pretrained backbone (adapted for single-pixel inputs):

```bash
python train_keypointrcnn.py
```

### Customizing Training Parameters

```bash
python train.py \
  --data_path ./path/to/measurements \
  --annotation_path ./path/to/annotations.json \
  --epochs 1000 \
  --batch 4 \
  --lr 0.0001 \
  --output_dir ./weights \
  --num_keypoints 13
```

**Arguments**:
- `--data_path`: Directory containing measurement data (or simulated images)
- `--annotation_path`: Path to COCO format annotations JSON
- `--epochs`: Number of training epochs (default: 1000)
- `--batch`: Mini-batch size (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--output_dir`: Directory to save model weights (default: "weights")
- `--num_keypoints`: Number of keypoints to detect (default: 13)

### Model Parameters

The `KeypointDetection` model accepts additional parameters for single-pixel imaging:

```python
model = KeypointDetection(
    num_keypoints=13,
    product=True,   # Use Hadamard-like multiplication (True) or convolution (False)
    reduce=False    # Reduce number of measurements (True = 1024, False = 4096)
)
```

**Key Parameters**:
- `product=True`: Mimics Hadamard pattern multiplication (typical for SPI)
- `reduce=True`: Simulates fewer measurements (32×32 = 1024 instead of 64×64 = 4096)

### Monitoring Training

Use TensorBoard to visualize training:

```bash
tensorboard --logdir tensorboard_logs
```

Access at `http://localhost:6006` to view:
- Training loss curves
- Validation metrics
- Keypoint visualizations

## Evaluation

COCO-style evaluation for keypoint detection:

```python
from utils import evaluate_keypoint_detection

evaluate_keypoint_detection(
    model=model,
    device=device,
    ann_file='path/to/annotations.json',
    data_loader=val_loader,
    num_keypoints=13
)
```

**Metrics**:
- Average Precision (AP) @ various OKS thresholds
- Average Recall (AR) @ various detection counts
- AP/AR for different object sizes

## Project Structure

```
Pose-SPI-main/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── dataset.rar                        # Compressed dataset
│
├── models/                            # Neural network architectures
│   ├── model_grayscale_input.py      # Simple model for SPI measurements
│   ├── model_grayscale_keypointrcnn.py  # EfficientNet-based SPI model
│   ├── efficentnet.py                # EfficientNet implementation
│   └── efficientnet_utils.py         # EfficientNet utilities
│
├── data_loader.py                     # COCO dataset loader (SPI simulation)
├── train.py                           # Training script (Simple model)
├── train_keypointrcnn.py              # Training script (EfficientNet model)
├── utils.py                           # Utilities (evaluation, visualization)
│
├── coco_eval/                         # COCO evaluation toolkit
│   ├── coco_eval.py                  # Evaluation implementation
│   └── mask_cocoeval.py              # Mask utilities
│
├── toydataset/                        # Sample dataset
│   ├── README.txt                    # Dataset description
│   ├── annotations/
│   │   └── annotations.json          # COCO annotations
│   └── images/
│       └── train2017/                # 100 sample images (SPI simulation)
│
└── tensorboard_logs/                  # Training logs (auto-generated)
```

## Model Details

### Model 1: Simple Single-Pixel Detector

**Architecture**: `models/model_grayscale_input.py`

**Components**:
- **CustomConv**: Simulates Hadamard product (element-wise multiplication with learnable patterns)
- **Encoder**: Extracts features from single-pixel measurements
- **FSRCNN**: Upsamples feature representations
- **SimpleCNN**: Lightweight keypoint detection head

**Use case**: Fast inference for real-time privacy-preserving pose estimation

**Training**: `python train.py`

**Key Features**:
- `product=True`: Mimics structured illumination patterns
- `reduce=False`: Uses full 4096 measurements (64×64)

### Model 2: EfficientNet-based Single-Pixel Detector

**Architecture**: `models/model_grayscale_keypointrcnn.py`

**Components**:
- **Encoder**: Same as simple model (SPI measurement processing)
- **FSRCNN**: Feature upsampling
- **EfficientNet Backbone**: Pretrained features adapted for SPI
- **KeypointPredictor**: RCNN-style regression head

**Use case**: Higher accuracy when computational resources allow

**Training**: `python train_keypointrcnn.py`

## Understanding Single-Pixel Imaging

### What is Single-Pixel Imaging?

Single-pixel imaging is a computational imaging technique that:

1. **Uses a single detector** instead of megapixel camera arrays
2. **Illuminates scenes** with structured patterns (Hadamard, Fourier, random)
3. **Measures total light** reflected/transmitted for each pattern
4. **Reconstructs or analyzes** the scene from multiple measurements

### Why Single-Pixel for Privacy?

Traditional cameras capture full spatial information, which:
- ❌ Reveals faces and identities
- ❌ Shows environmental details
- ❌ Can be misused or hacked
- ❌ Raises privacy concerns in sensitive settings

Single-pixel imaging:
- ✅ Captures only aggregate measurements
- ✅ Cannot reconstruct detailed images without full measurement set
- ✅ Allows task-specific information extraction (keypoints) without full image
- ✅ Provides hardware-level privacy protection

### Measurement Requirements

**Full reconstruction**: ~N² measurements (for N×N image)
**Keypoint detection** (this work): Can work with fewer measurements

Example for 64×64 resolution:
- Full reconstruction: ~4096 measurements
- This method: Trained on 4096 (full) or ~60 (reduced) measurements
- Future work: Could implement real-time prediction and/or video prediction using a Digital Micromirror Device (DMD) 

### Hadamard Product

The `CustomConv` layer with `product=True` simulates Hadamard product:
```python
measurement = learnable_pattern * input_signal
```

## Important Notes

### Single-Pixel Imaging Specifics

- **Input Interpretation**: The 64×64 "images" represent measurement patterns, not traditional photos
- **Privacy Guarantee**: Keypoints extracted directly; full image never reconstructed
- **Measurement Count**: Configurable via `reduce` parameter (~60 vs 4096 measurements)
- **Pattern Simulation**: `CustomConv` with `product=True` mimics Hadamard product

### Training Considerations

- **Simulated Data**: This implementation uses images to simulate SPI measurements
- **Real Hardware**: For actual single-pixel cameras, modify data loader to accept measurement vectors
- **Batch Size**: Small batches (2-8) recommended for regularization
- **Learning Rate**: Default 0.0001 optimized for toy dataset; adjust for different data

### Model Architecture Constraints

- **Measurement Resolution**: Models expect 64×64 measurement patterns
- **Fully Connected Layers**: Dimensions depend on number of measurements
  - `reduce=False`: 4096 measurements → fc(4096, 1024)
  - `reduce=True`: 1024 measurements → fc(1024, 1024)
- **Keypoint Count**: Default 13, configurable via `--num_keypoints`

### Best Practices

1. **Overfitting Test**: Verify model can overfit small subset (5-10 samples)
2. **Measurement Reduction**: Start with full measurements, then experiment with `reduce=True`
3. **Pattern Learning**: Monitor CustomConv weights to see learned sampling patterns
4. **Privacy Validation**: Verify that reconstructing images from features is not feasible

## Comparison: Traditional vs Single-Pixel Imaging

| Aspect | Traditional Camera | Single-Pixel Imaging (This Work) |
|--------|-------------------|----------------------------------|
| **Sensor** | Megapixel array | Single photodetector |
| **Spatial Resolution** | High (megapixels) | Determined by # measurements |
| **Privacy** | ❌ Full image captured | ✅ No image reconstruction |
| **Cost** | Higher (sensor array) | Lower (single detector) |
| **Speed** | Fast (parallel capture) | Slower (sequential patterns) |
| **Use Case** | General vision | Privacy-critical pose estimation |
| **Wavelength Range** | Visible light | Infrared, terahertz, X-ray possible |

## Troubleshooting

### CUDA Out of Memory

```bash
python train.py --batch 2  # Reduce batch size
```

### Different Number of Measurements

To use different measurement patterns (e.g., 32×32):

1. Modify `CustomConv` weight size in `models/model_grayscale_input.py`
2. Update encoder FC layer input dimensions
3. Adjust data loader to provide correct measurement size

### Custom Keypoint Configuration

Ensure consistency:
1. Dataset annotations: correct number of keypoints
2. Model: `--num_keypoints` argument
3. COCO eval: matching `kpt_oks_sigmas` (if using COCO evaluation)

## Citation

If you use this code in your research, please cite the paper:

Aleksandr Tsoy, Zonghao Liu, Huan Zhang, Mi Zhou, Wenming Yang, Hongya Geng, Kui Jiang, Xin Yuan, and Zihan Geng, "[Image-free single-pixel keypoint detection for privacy preserving human pose estimation](https://opg.optica.org/ol/abstract.cfm?URI=ol-49-3-546)," Opt. Lett. 49, 546-549 (2024)

```bibtex
@article{tsoy2024image,
  title={Image-free single-pixel keypoint detection for privacy preserving human pose estimation},
  author={Tsoy, Aleksandr and Liu, Zonghao and Zhang, Huan and Zhou, Mi and Yang, Wenming and Geng, Hongya and Jiang, Kui and Yuan, Xin and Geng, Zihan},
  journal={Optics Letters},
  volume={49},
  number={3},
  pages={546--549},
  year={2024},
  publisher={Optica Publishing Group},
  doi={10.1364/OL.514213}
}
```

---
