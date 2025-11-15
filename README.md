# Pose-SPI: Human Keypoint Detection

A PyTorch-based implementation for human keypoint detection from grayscale images. This project provides two model architectures for pose estimation on low-resolution (64x64) images, suitable for research and applications in human activity recognition, motion analysis, and pose estimation.

## Features

- **Two Model Architectures**:
  - Simple CNN-based keypoint detector with lightweight architecture
  - EfficientNet-based model leveraging pretrained features for enhanced accuracy
- **COCO Format Support**: Compatible with COCO-style keypoint annotations
- **Flexible Training Pipeline**: Customizable hyperparameters and dataset paths
- **Real-time Monitoring**: TensorBoard integration for training visualization
- **COCO Evaluation Metrics**: Standard keypoint detection evaluation using OKS (Object Keypoint Similarity)

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Important Notes](#important-notes)
- [Contributing](#contributing)

## Architecture Overview

Both model variants share a common pipeline structure:

```
Input (Grayscale 64x64)
    ↓
Encoder (CustomConv → MaxPool → FC)
    ↓
FSRCNN (Feature Extraction → Mapping → Upsampling → Expansion)
    ↓
┌─────────────────────────────────────┐
│  Model 1: SimpleCNN                 │
│  (Conv × 2 → AvgPool × 2)          │
│                                     │
│  Model 2: EfficientNet + RCNN       │
│  (Pretrained Backbone → Predictor)  │
└─────────────────────────────────────┘
    ↓
Output: N × 3 [x, y, visibility]
```

### Key Components

1. **Encoder**: Reduces spatial dimensions using learnable convolution, max pooling, and fully connected layers
2. **FSRCNN** (Fast Super-Resolution CNN): Upsamples features while preserving spatial details
3. **Keypoint Detection Head**:
   - **Simple Model**: Lightweight CNN with 2 convolutions and average pooling
   - **EfficientNet Model**: Pretrained EfficientNet-B0 backbone with RCNN-style predictor

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

3. If TensorBoard is not installed, install it separately:
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

The repository includes a toy dataset (`toydataset/`) for quick experimentation:

- **Images**: 100 RGB images (64x64 pixels) with background removed
- **Annotations**: COCO format JSON with 14 keypoints per person
- **Location**:
  - Images: `toydataset/images/train2017/`
  - Annotations: `toydataset/annotations/annotations.json`

### Using Custom Datasets

To use your own dataset:

1. Organize images in a directory (e.g., `my_dataset/images/`)
2. Create COCO-format annotations JSON file
3. Ensure keypoint format matches: `[x, y, visibility]` for each keypoint
4. Pass paths via command-line arguments (see [Training](#training))

## Training

### Simple Keypoint Detector

Train the lightweight CNN-based model:

```bash
python train.py
```

**Default settings**: 1000 epochs, batch size 4, learning rate 0.0001, 13 keypoints

### EfficientNet-based Model

Train the EfficientNet-based model with pretrained backbone:

```bash
python train_keypointrcnn.py
```

### Customizing Training Parameters

Both training scripts support the following arguments:

```bash
python train.py \
  --data_path ./path/to/images \
  --annotation_path ./path/to/annotations.json \
  --epochs 1000 \
  --batch 4 \
  --lr 0.0001 \
  --output_dir ./weights \
  --num_keypoints 13
```

**Arguments**:
- `--data_path`: Directory containing training images
- `--annotation_path`: Path to COCO format annotations JSON
- `--epochs`: Number of training epochs (default: 1000)
- `--batch`: Mini-batch size (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--output_dir`: Directory to save model weights (default: "weights")
- `--num_keypoints`: Number of keypoints to detect (default: 13)

### Monitoring Training

Use TensorBoard to visualize training progress in real-time:

```bash
tensorboard --logdir tensorboard_logs
```

Then open your browser to `http://localhost:6006` to view:
- Training loss curves
- Validation metrics
- Keypoint visualization overlays

## Evaluation

The project includes COCO-style evaluation metrics for keypoint detection:

```python
from utils import evaluate_keypoint_detection

evaluate_keypoint_detection(
    model=model,
    device=device,
    ann_file='path/to/annotations.json',
    data_loader=val_data_loader,
    num_keypoints=13
)
```

**Metrics computed**:
- Average Precision (AP) @ various OKS thresholds
- Average Recall (AR) @ various detection counts
- AP/AR for medium and large person sizes

## Project Structure

```
Pose-SPI-main/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
├── dataset.rar                        # Compressed dataset
│
├── models/                            # Neural network architectures
│   ├── model_grayscale_input.py      # Simple CNN keypoint detector
│   ├── model_grayscale_keypointrcnn.py  # EfficientNet-based detector
│   ├── efficentnet.py                # EfficientNet implementation
│   └── efficientnet_utils.py         # EfficientNet utilities
│
├── data_loader.py                     # COCO dataset loader
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
│       └── train2017/                # 100 sample images
│
└── tensorboard_logs/                  # Training logs (auto-generated)
```

## Model Details

### Model 1: Simple Keypoint Detector

**Architecture**: `models/model_grayscale_input.py`

- **Encoder**: CustomConv → MaxPool → FC layers
- **FSRCNN**: Super-resolution upsampling module
- **SimpleCNN Head**: 2 Conv layers + 2 AvgPool layers → Keypoint predictions

**Use case**: Lightweight model for real-time applications or resource-constrained environments

**Training**: `python train.py`

### Model 2: EfficientNet-based Keypoint Detector

**Architecture**: `models/model_grayscale_keypointrcnn.py`

- **Encoder**: CustomConv → MaxPool → FC layers
- **FSRCNN**: Super-resolution upsampling module
- **EfficientNet Backbone**: Pretrained EfficientNet-B0 for feature extraction
- **RCNN Head**: Keypoint regression with L1/Smooth L1 loss

**Use case**: Higher accuracy applications where pretrained features improve performance

**Training**: `python train_keypointrcnn.py`

## Important Notes

### Hyperparameters

- **Default settings** are optimized for the included toy dataset (100 images)
- When using different datasets, adjust:
  - **Learning rate**: Larger datasets may benefit from higher learning rates
  - **Epochs**: More data typically requires more training time
  - **Batch size**: Small batch sizes (2-8) often work best for small models

### Model Architecture Constraints

- **Input size**: Models expect 64x64 pixel images (hardcoded)
- **Fully connected layers**: Input feature counts depend on image size
  - If using different input sizes, modify FC layer dimensions in encoder
- **Keypoint count**: Default is 13, but configurable via `--num_keypoints`

### COCO Evaluation

- **kpt_oks_sigmas**: Contains empirically calculated values for 17 keypoints
- For accurate evaluation:
  - Use original 17 keypoints, OR
  - Modify `kpt_oks_sigmas` to match your keypoint configuration

### Best Practices

1. **Small Batch Training**: Small batch sizes add noise that improves model robustness
2. **Overfitting Test**: Before full training, verify the model can overfit a small subset (5-10 images)
   - Confirms no bugs in code
   - Validates that the model has sufficient capacity
3. **Validation Set**: Currently uses training set for validation (TODO)
   - For production use, create separate validation/test sets

### Reproducibility

Training uses fixed random seeds and deterministic settings:
```python
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Visualization

The `utils.py` module provides keypoint visualization:

```python
from utils import draw_keypoints

result_img = draw_keypoints(
    pred_keypoints=predictions,
    gt_keypoints=ground_truth,
    images=images
)
```

**Color coding**:
- Red circles: Ground truth keypoints
- Green circles: Predicted keypoints
- Blue lines: Ground truth skeleton
- Cyan lines: Predicted skeleton

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py --batch 2
```

### Different Input Size

Modify the fully connected layer dimensions in the encoder:
- Open `models/model_grayscale_input.py` or `models/model_grayscale_keypointrcnn.py`
- Update `nn.Linear(input_features, output_features)` based on your image size

### Custom Keypoint Configuration

Ensure consistency across:
1. Dataset annotations (`num_keypoints` in JSON)
2. Command-line argument `--num_keypoints`
3. COCO evaluation `kpt_oks_sigmas` (if using COCO eval)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is available for research and educational purposes.

## Citation

If you use this code in your research, please cite this repository.

## Acknowledgments

- EfficientNet implementation based on [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- COCO evaluation tools adapted from [pycocotools](https://github.com/cocodataset/cocoapi)
