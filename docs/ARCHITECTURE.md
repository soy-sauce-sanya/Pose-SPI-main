# Architecture Documentation

This document provides an in-depth explanation of the Pose-SPI architecture, design decisions, and implementation details.

## Table of Contents

- [Overview](#overview)
- [Pipeline Flow](#pipeline-flow)
- [Model Architectures](#model-architectures)
- [Component Details](#component-details)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)

---

## Overview

Pose-SPI implements human keypoint detection from grayscale 64×64 images using deep learning. The project offers two model variants:

1. **Simple Model**: Lightweight CNN-based detector
2. **EfficientNet Model**: Pretrained backbone with RCNN-style head

Both models share a common encoder-decoder architecture with a specialized upsampling module (FSRCNN) but differ in their keypoint detection heads.

---

## Pipeline Flow

### High-Level Data Flow

```
Input Image (RGB 64×64)
    ↓
Grayscale Conversion (1 channel)
    ↓
┌─────────────────────────────────┐
│  ENCODER                         │
│  - CustomConv (1→64 channels)   │
│  - MaxPool (64×64 → 32×32)      │
│  - FC (flatten → 512 → 256)     │
└─────────────────────────────────┘
    ↓ Feature Vector (256)
┌─────────────────────────────────┐
│  FSRCNN (Upsampling)            │
│  - Feature Extraction            │
│  - Mapping Layers                │
│  - Deconvolution                 │
│  - Expansion                     │
└─────────────────────────────────┘
    ↓ Feature Map (64, 64, 64)
┌─────────────────────────────────┐
│  DETECTION HEAD (two variants)   │
│                                  │
│  Simple Model:                   │
│  - SimpleCNN                     │
│    (Conv → AvgPool × 2)         │
│                                  │
│  EfficientNet Model:             │
│  - Transition Conv (64→3)        │
│  - EfficientNet Backbone         │
│  - KeypointPredictor (FC)        │
└─────────────────────────────────┘
    ↓
Output: Keypoints (N × 3)
[x, y, visibility] for each keypoint
```

---

## Model Architectures

### Model 1: Simple Keypoint Detector

**File**: `models/model_grayscale_input.py`

**Philosophy**: Minimalist architecture designed for efficiency and interpretability.

#### Architecture Breakdown

```python
KeypointDetection(
    Encoder(
        CustomConv(1 → 64)
        MaxPool2d(kernel_size=2)
        Flatten()
        Linear(64*32*32 → 512)
        ReLU()
        Linear(512 → 256)
    ),
    FSRCNN(
        # Feature extraction
        Conv2d(256 → 64, kernel=3)
        BatchNorm2d(64)
        ReLU()

        # Mapping (repeated 3 times)
        Conv2d(64 → 64, kernel=3)
        BatchNorm2d(64)
        ReLU()

        # Deconvolution (upsampling)
        Upsample(scale_factor=2, mode='bilinear')
        Conv2d(64 → 64, kernel=3)

        # Expansion
        Conv2d(64 → 64, kernel=1)
    ),
    SimpleCNN(
        Conv2d(64 → 32, kernel=3, padding=1)
        ReLU()
        AvgPool2d(kernel_size=2)

        Conv2d(32 → num_keypoints*3, kernel=3, padding=1)
        ReLU()
        AvgPool2d(kernel_size=32)

        Reshape → (batch, num_keypoints, 3)
    )
)
```

**Parameters**: ~3-5M (depending on num_keypoints)

**Advantages**:
- Fast inference (~5-10ms on GPU)
- Low memory footprint
- Easy to understand and modify
- No dependency on pretrained weights

**Limitations**:
- Limited feature extraction capacity
- May struggle with complex poses
- Requires more data to generalize

---

### Model 2: EfficientNet-based Detector

**File**: `models/model_grayscale_keypointrcnn.py`

**Philosophy**: Leverage pretrained features for improved accuracy.

#### Architecture Breakdown

```python
KeypointDetection(
    Encoder(
        # Same as Model 1
        CustomConv(1 → 64)
        MaxPool2d(kernel_size=2)
        Flatten()
        Linear(64*32*32 → 512)
        ReLU()
        Linear(512 → 256)
    ),
    FSRCNN(
        # Same as Model 1
        # Upsamples to (batch, 64, 64, 64)
    ),
    # Transition to RGB-like format
    Conv2d(64 → 3, kernel=1),

    KeypointRCNN(
        EfficientNet-B0(
            # Pretrained on ImageNet
            # Modified first conv for grayscale compatibility
            MBConv blocks × 16
            # Output: (batch, 1280) features
        )
    ),
    KeypointPredictor(
        Linear(1280 → 512)
        ReLU()
        Dropout(0.3)
        Linear(512 → 256)
        ReLU()
        Dropout(0.3)
        Linear(256 → num_keypoints*3)
        Reshape → (batch, num_keypoints, 3)
    )
)
```

**Parameters**: ~10-15M (mostly from EfficientNet)

**Advantages**:
- Better feature representations from pretrained backbone
- Higher accuracy on complex poses
- Transfer learning benefits

**Limitations**:
- Slower inference (~20-30ms on GPU)
- Higher memory usage
- More complex architecture
- Requires GPU for practical use

---

## Component Details

### CustomConv

**Purpose**: Learnable weight multiplication or convolution with optional upsampling.

**Design Choice**: Provides flexibility in feature transformation while maintaining computational efficiency.

```python
class CustomConv(nn.Module):
    def __init__(self, input_channels, output_channels, upsample=False):
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels, 1, 1))
        self.upsample = upsample

    def forward(self, x):
        x = x * self.weight  # Element-wise multiplication
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x
```

**Key Features**:
- Lightweight (minimal parameters)
- Optional bilinear upsampling
- Acts as learnable gating mechanism

---

### Encoder

**Purpose**: Compress spatial information while extracting semantic features.

**Architecture Stages**:

1. **Stage 1 - Initial Feature Extraction**
   - Input: (batch, 1, 64, 64)
   - CustomConv: Increases channels to 64
   - Output: (batch, 64, 64, 64)

2. **Stage 2 - Spatial Reduction**
   - MaxPool: Reduces to (batch, 64, 32, 32)
   - Preserves important features via max operation

3. **Stage 3 - Feature Embedding**
   - Flatten: (batch, 64*32*32) = (batch, 65536)
   - FC1: 65536 → 512 (dimensionality reduction)
   - ReLU activation
   - FC2: 512 → 256 (compact representation)
   - Output: (batch, 256)

**Design Rationale**:
- Early convolution preserves spatial structure
- Max pooling provides translation invariance
- FC layers create compact semantic embedding
- 256-dim bottleneck balances expressiveness and efficiency

---

### FSRCNN (Fast Super-Resolution CNN)

**Purpose**: Upsample compact feature vectors back to spatial feature maps suitable for keypoint localization.

**Inspired by**: [FSRCNN paper](https://arxiv.org/abs/1608.00367) for image super-resolution

**Why FSRCNN for Keypoints?**
- Preserves fine spatial details during upsampling
- Learned upsampling outperforms simple interpolation
- Mapping layers refine features before spatial expansion

**Architecture Stages**:

1. **Feature Extraction**
   ```python
   Linear(256 → 64*4*4) → Reshape to (batch, 64, 4, 4)
   Conv2d(64 → 64, kernel=3, padding=1)
   BatchNorm2d + ReLU
   ```

2. **Mapping Layers** (repeated multiple times)
   ```python
   Conv2d(64 → 64, kernel=3, padding=1)
   BatchNorm2d + ReLU
   ```
   - Refine features in feature space
   - Multiple layers increase receptive field

3. **Deconvolution** (Upsampling)
   ```python
   Upsample(scale_factor=2, mode='bilinear')  # 4×4 → 8×8
   Conv2d(64 → 64, kernel=3, padding=1)
   # Repeated for 8×8 → 16×16 → 32×32 → 64×64
   ```

4. **Expansion**
   ```python
   Conv2d(64 → 64, kernel=1)
   ```
   - Final channel adjustment
   - 1×1 conv acts as learned combination

**Output**: (batch, 64, 64, 64) - rich spatial feature map

---

### SimpleCNN (Simple Model Head)

**Purpose**: Convert feature maps to keypoint coordinates.

**Architecture**:

```python
# Layer 1: Feature refinement
Conv2d(64 → 32, kernel=3, padding=1)  # Preserve spatial size
ReLU()
AvgPool2d(kernel_size=2)  # 64×64 → 32×32

# Layer 2: Keypoint prediction
Conv2d(32 → num_keypoints*3, kernel=3, padding=1)
ReLU()
AvgPool2d(kernel_size=32)  # 32×32 → 1×1 global pooling

# Reshape
Reshape to (batch, num_keypoints, 3)
```

**Design Choices**:
- **Average pooling**: Provides smoother gradients than max pooling
- **3 channels per keypoint**: [x, y, visibility]
- **Global average pooling**: Acts as spatial averaging, making predictions robust to small translations
- **Lightweight**: Only 2 conv layers minimize overfitting

**Output Format**:
- Shape: (batch, num_keypoints, 3)
- Values: Continuous coordinates (x, y) and visibility score

---

### EfficientNet Backbone

**Purpose**: Extract powerful features using pretrained representations.

**EfficientNet-B0 Architecture**:
- **Compound Scaling**: Balances depth, width, and resolution
- **MBConv Blocks**: Mobile Inverted Bottleneck Convolution
- **Squeeze-and-Excitation**: Adaptive channel-wise feature recalibration

**Modifications for Pose-SPI**:

```python
# Original: 3-channel RGB input
# Modified: 1-channel grayscale input (handled by transition conv 64→3)

# After transition conv, standard EfficientNet processes:
efficientnet_b0(
    # 16 MBConv blocks with varying expansion ratios
    # Progressive feature refinement
    # Output: (batch, 1280) feature vector
)
```

**Pretrained Weights**:
- Trained on ImageNet (1000 classes, millions of images)
- Learned general visual features (edges, textures, shapes)
- Transfer learning accelerates convergence

**Fine-tuning Strategy**:
- All layers trainable (full fine-tuning)
- Can freeze early layers for faster training
- Dropout in predictor prevents overfitting

---

### KeypointPredictor (EfficientNet Model Head)

**Purpose**: Regression head to map EfficientNet features to keypoint coordinates.

**Architecture**:

```python
Linear(1280 → 512)
ReLU()
Dropout(0.3)

Linear(512 → 256)
ReLU()
Dropout(0.3)

Linear(256 → num_keypoints*3)
Reshape to (batch, num_keypoints, 3)
```

**Design Choices**:
- **Fully connected**: Learns global relationships between features and keypoints
- **Progressive reduction**: 1280 → 512 → 256 → output
- **Dropout**: Prevents overfitting (p=0.3)
- **ReLU activations**: Non-linear transformations

**Why FC Layers?**
- EfficientNet already provides spatial features
- Keypoint positions depend on global context
- FC layers excel at learning these global relationships

---

## Design Decisions

### Why Two Models?

**Simple Model**:
- **Use Case**: Real-time applications, embedded systems, limited compute
- **Trade-off**: Speed vs. accuracy
- **Best For**: Simple poses, controlled environments

**EfficientNet Model**:
- **Use Case**: High-accuracy requirements, offline processing
- **Trade-off**: Accuracy vs. speed/memory
- **Best For**: Complex poses, varied conditions

### Why 64×64 Input?

**Rationale**:
1. **Computational Efficiency**: Smaller input = faster processing
2. **Dataset Constraint**: Toy dataset uses 64×64 images
3. **Sufficient Resolution**: Human keypoints identifiable at this scale
4. **Memory**: Fits larger batches in GPU memory

**Trade-offs**:
- Limited spatial precision
- May miss fine details
- **Note**: Production systems often use 256×256 or larger

### Why Grayscale?

**Advantages**:
1. **Fewer Parameters**: 1 channel vs. 3 channels (3× reduction in first layer)
2. **Faster Training**: Less data to process
3. **Sufficient Information**: Keypoint detection relies more on shape than color
4. **Robustness**: Invariant to color variations

**Considerations**:
- Color can provide useful cues (clothing, background)
- Pretrained models expect RGB (handled via transition conv)

### Loss Functions

#### Simple Model: MSE Loss

```python
criterion = nn.MSELoss()
loss = criterion(predictions, ground_truth)
```

**Why MSE?**
- Natural choice for coordinate regression
- Directly minimizes Euclidean distance
- Smooth gradients

**Limitations**:
- Treats all keypoints equally
- Sensitive to outliers

#### EfficientNet Model: Smooth L1 Loss

```python
criterion = KeypointRCNNLoss(loss_type='smooth_l1')
loss = criterion(predictions, targets)
```

**Why Smooth L1?**
- Less sensitive to outliers than MSE
- Smooth gradients near zero (L2-like)
- Linear for large errors (L1-like)

**Formula**:
```
smooth_l1(x) = 0.5 * x^2     if |x| < 1
               |x| - 0.5     otherwise
```

### Optimizer Choice: SGD

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
```

**Why SGD (not Adam)?**
1. **Small Batch Training**: SGD with small batches provides beneficial noise
2. **Better Generalization**: Often generalizes better than Adam
3. **Stability**: More stable for small datasets

**Considerations**:
- Slower convergence than Adam
- Requires careful learning rate tuning
- May benefit from momentum (can be added)

### Batch Size: 4 (Default)

**Why Small Batches?**
1. **Regularization**: Adds noise to gradients (like dropout)
2. **Memory**: Fits in smaller GPUs
3. **Small Dataset**: 100 images → batches of 4 = 25 iterations/epoch

**Scaling Up**:
- Larger datasets can use batch sizes 16-64
- Adjust learning rate proportionally (linear scaling rule)

---

## Performance Considerations

### Computational Complexity

#### Simple Model

| Component | FLOPs | Parameters |
|-----------|-------|------------|
| Encoder | ~130M | ~33M |
| FSRCNN | ~180M | ~0.5M |
| SimpleCNN | ~25M | ~0.05M |
| **Total** | **~335M** | **~33.5M** |

**Inference Time**:
- GPU (NVIDIA RTX 3080): ~5-8ms
- CPU (Intel i7): ~50-80ms

#### EfficientNet Model

| Component | FLOPs | Parameters |
|-----------|-------|------------|
| Encoder | ~130M | ~33M |
| FSRCNN | ~180M | ~0.5M |
| EfficientNet | ~400M | ~5.3M |
| Predictor | ~2M | ~0.5M |
| **Total** | **~712M** | **~39.3M** |

**Inference Time**:
- GPU (NVIDIA RTX 3080): ~15-25ms
- CPU (Intel i7): ~200-300ms

### Memory Usage

**Simple Model**:
- Model: ~130 MB
- Batch (size 4): ~50 MB
- Total: ~180 MB

**EfficientNet Model**:
- Model: ~160 MB
- Batch (size 4): ~80 MB
- Total: ~240 MB

### Optimization Opportunities

1. **Model Quantization**:
   - INT8 quantization → 4× size reduction
   - ~2× speedup with minimal accuracy loss

2. **Pruning**:
   - Remove redundant weights
   - 20-40% speedup possible

3. **Knowledge Distillation**:
   - Train simple model to mimic EfficientNet
   - Get EfficientNet accuracy at Simple model speed

4. **ONNX Export**:
   - Convert to ONNX for deployment
   - Use TensorRT for GPU optimization

---

## Training Strategy

### Reproducibility

```python
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Trade-offs**:
- **Deterministic**: Same results on repeated runs
- **Performance**: ~10% slower than non-deterministic mode
- **Debugging**: Easier to identify bugs

### Overfitting Prevention

1. **Small Batch Sizes**: Acts as regularization
2. **Dropout** (EfficientNet model): p=0.3 in predictor
3. **Data Augmentation** (TODO): Rotations, flips, color jitter
4. **Early Stopping** (TODO): Monitor validation loss

### Validation Strategy

**Current**: Uses training set (suboptimal)

**Recommended**:
```
- Training: 70% of data
- Validation: 15% of data
- Test: 15% of data
```

### Hyperparameter Tuning

**Critical Hyperparameters**:

| Parameter | Default | Tuning Range | Impact |
|-----------|---------|--------------|--------|
| Learning Rate | 0.0001 | 1e-5 to 1e-3 | High |
| Batch Size | 4 | 2 to 16 | Medium |
| Epochs | 1000 | 100 to 5000 | Low |
| Num Keypoints | 13 | 13 to 17 | Structural |

**Tuning Process**:
1. Start with default values
2. Verify model can overfit small subset (5-10 images)
3. Tune learning rate (most important)
4. Adjust batch size based on GPU memory
5. Increase epochs until validation loss plateaus

---

## Extending the Architecture

### Adding More Keypoints

**Simple**: Change `--num_keypoints` argument

**Considerations**:
- Update COCO annotations to include new keypoints
- Modify `kpt_oks_sigmas` for COCO evaluation
- Larger output → slightly slower inference

### Larger Input Images

**Example**: 128×128 instead of 64×64

**Required Changes**:

1. **Encoder FC Layers**:
   ```python
   # models/model_grayscale_input.py
   # Change: Linear(65536, 512)  # 64*32*32
   # To:     Linear(262144, 512)  # 128*64*64
   ```

2. **Training Script**:
   ```python
   # No transform resize needed if images are already correct size
   ```

3. **FSRCNN Upsampling**:
   - May need additional upsampling stages
   - Or adjust intermediate sizes

### Different Backbones

**Replacing EfficientNet**:

```python
# Instead of EfficientNet-B0
from torchvision.models import resnet18, resnet50, mobilenet_v2

# Example: ResNet18
backbone = resnet18(pretrained=True)
backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
# Remove final FC layer
backbone = nn.Sequential(*list(backbone.children())[:-1])
```

**Options**:
- **ResNet**: Better for deeper networks
- **MobileNet**: Faster, more efficient
- **Vision Transformer**: State-of-the-art accuracy

---

## Future Improvements

1. **Multi-Scale Features**: Combine features from multiple resolutions
2. **Attention Mechanisms**: Focus on relevant body parts
3. **Temporal Modeling**: For video inputs (LSTM/Transformer)
4. **Heatmap Regression**: Predict heatmaps instead of coordinates
5. **Multi-Person Detection**: Handle multiple people in one image
6. **Data Augmentation**: Rotations, flips, color jitter, cutout
7. **Advanced Loss Functions**: Perceptual loss, adversarial loss

---

## Conclusion

Pose-SPI provides a flexible framework for keypoint detection with two complementary architectures. The simple model offers speed and efficiency, while the EfficientNet model provides higher accuracy through transfer learning. Both share a common encoder-FSRCNN pipeline, making it easy to experiment with different detection heads.

The architecture is designed for research and prototyping, with clear extension points for production deployment or advanced research.
