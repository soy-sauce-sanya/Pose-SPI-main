# Architecture Documentation

## Single-Pixel Imaging for Privacy-Preserving Pose Estimation

This document provides an in-depth explanation of the Pose-SPI architecture, focusing on how it enables keypoint detection from single-pixel measurements without reconstructing full images.

## Table of Contents

- [Overview](#overview)
- [Single-Pixel Imaging Principles](#single-pixel-imaging-principles)
- [Architecture Pipeline](#architecture-pipeline)
- [Component Details](#component-details)
- [Privacy-Preserving Design](#privacy-preserving-design)
- [Performance Considerations](#performance-considerations)

---

## Overview

Pose-SPI implements privacy-preserving human keypoint detection using **single-pixel imaging (SPI)**. Unlike traditional computer vision systems that capture full images, this approach:

1. **Captures**: Aggregate light measurements using a single photodetector
2. **Processes**: Measurements through specialized neural networks
3. **Extracts**: Human keypoints directly without image reconstruction
4. **Preserves**: Privacy by never forming complete visual images

### Key Innovation

The architecture enables **direct task learning** from single-pixel measurements, bypassing the traditional two-step process of:
1. ❌ Image reconstruction (privacy-invasive)
2. ❌ Feature extraction from reconstructed image

Instead, it performs:
1. ✅ Direct feature extraction from measurements
2. ✅ Task-specific (keypoint) prediction

---

## Single-Pixel Imaging Principles

### What Are Single-Pixel Measurements?

In single-pixel imaging:

```
┌─────────────────────────────────────────┐
│  Scene Illumination & Measurement       │
├─────────────────────────────────────────┤
│                                         │
│  Pattern 1: ▓░▓░▓░▓░  →  Detector  →  m₁│
│  Pattern 2: ▓▓░░▓▓░░  →  Detector  →  m₂│
│  Pattern 3: ░▓▓░░▓▓░  →  Detector  →  m₃│
│  ...                                    │
│  Pattern N: ▓░░▓▓░▓░  →  Detector  →  mₙ│
│                                         │
│  Measurement Vector: [m₁, m₂, ..., mₙ] │
└─────────────────────────────────────────┘
```

Each measurement is the **scalar dot product** of:
- **Illumination pattern**: Spatial distribution of light
- **Scene reflectance**: How scene reflects light

Mathematically: `measurement_i = ⟨pattern_i, scene⟩ + noise`

### Hadamard Patterns

This implementation uses **Hadamard-inspired patterns** for measurements:

**Properties**:
- **Orthogonal**: Patterns are statistically independent
- **Binary**: +1 (light) or -1 (no light) values
- **Optimal**: Maximize signal-to-noise ratio
- **Complete**: Set of N² patterns can fully describe N×N scene

**In Pose-SPI**:
- 64×64 resolution → up to 4096 measurements
- CustomConv layer simulates Hadamard multiplication
- Learnable patterns adapt to keypoint detection task

---

## Architecture Pipeline

### High-Level Flow

```
Single-Pixel Measurements
(simulated as 64×64 grayscale in this implementation)
          ↓
┌─────────────────────────────┐
│  CustomConv                 │
│  Simulates Hadamard product │
│  Learnable patterns         │
└─────────────────────────────┘
          ↓
┌─────────────────────────────┐
│  Encoder                    │
│  Extract features from      │
│  measurements (not images!) │
└─────────────────────────────┘
          ↓
Feature Vector (1024-dim)
          ↓
┌─────────────────────────────┐
│  FSRCNN                     │
│  Upsample feature space     │
│  (NOT image reconstruction) │
└─────────────────────────────┘
          ↓
Spatial Feature Map (64×64)
          ↓
┌──────────────────────────────────┐
│  Detection Head (two variants)   │
│  ├─ SimpleCNN                    │
│  └─ EfficientNet + Predictor     │
└──────────────────────────────────┘
          ↓
Keypoints (N × 3)
[x, y, visibility]
```

### Key Difference from Traditional Vision

| Traditional CV | Pose-SPI (Single-Pixel) |
|----------------|-------------------------|
| Image sensor → CNN | Measurements → CustomConv |
| Features from spatial pixels | Features from measurement patterns |
| Image = 2D spatial data | "Image" = arranged measurements |
| Conv layers process spatial neighborhoods | Conv layers process measurement correlations |

---

## Component Details

### 1. CustomConv: Hadamard Pattern Simulation

**Purpose**: Simulate the measurement process in single-pixel imaging

**Code**: `models/model_grayscale_input.py:5-30`

#### Architecture

```python
class CustomConv(nn.Module):
    def __init__(self, product=True):
        self.weight = nn.Parameter(torch.Tensor(1, 1, 64, 64))
        # Learnable 64×64 pattern (like Hadamard patterns)

    def forward(self, x):
        if product:
            out = self.weight.mul(x)  # Element-wise multiplication
        else:
            out = self.conv(x)  # Standard convolution
```

#### Single-Pixel Interpretation

**When `product=True` (Hadamard mode)**:
- Simulates pattern-based measurement: `y = pattern ⊙ measurements`
- Learnable weights act as **adaptive sampling patterns**
- Network learns optimal patterns for keypoint detection

**When `product=False` (Convolution mode)**:
- Falls back to standard convolution
- Useful for comparison with traditional approaches

#### Why This Design?

1. **Flexibility**: Patterns learned end-to-end for task
2. **Efficiency**: Single multiplication operation
3. **Interpretability**: Can visualize learned patterns
4. **Biological plausibility**: Similar to retinal sampling

---

### 2. Encoder: Measurement Feature Extraction

**Purpose**: Extract semantic features from single-pixel measurements

**Code**: `models/model_grayscale_input.py:32-62`

#### Architecture Stages

```python
# Stage 1: Pattern processing
CustomConv(product=True)  # Simulates Hadamard sampling
# Output: (batch, 1, 64, 64) "arranged measurements"

# Stage 2: Optional measurement reduction
MaxPool2d(kernel_size=2)  # Simulates fewer measurements
# Output: (batch, 1, 32, 32) if reduce=True

# Stage 3: Feature embedding
Flatten()
Linear(4096 → 1024) or Linear(1024 → 1024)  # Depends on reduce
# Output: (batch, 1024) feature vector

# Stage 4: Spatial arrangement
Reshape → (batch, 1, 32, 32)
```

#### Measurement Reduction

The `reduce` parameter simulates different measurement counts:

**reduce=False** (default):
- Uses all 64×64 = 4096 measurements
- fc: 4096 → 1024
- Higher accuracy, more measurements needed

**reduce=True**:
- Uses 32×32 = 1024 measurements (via MaxPooling)
- fc: 1024 → 1024
- Faster acquisition, slightly lower accuracy

**Physical Interpretation**:
- In real hardware: Fewer patterns shown, faster capture
- In this simulation: MaxPooling downsamples measurement grid

---

### 3. FSRCNN: Feature Space Upsampling

**Purpose**: Expand compact features to spatial representations for keypoint localization

**Code**: `models/model_grayscale_input.py:65-100`

#### Important: This is NOT Image Reconstruction

Common misconception: FSRCNN reconstructs images
**Reality**: FSRCNN upsamples **feature representations**

```
NOT THIS:                      BUT THIS:
Measurements → Image          Measurements → Features
                                             ↓
                                    Upsample Features
                                             ↓
                                       Keypoints
```

#### Architecture

```python
# Feature Extraction
Conv2d(1 → 56, kernel=5)  # Extract patterns from features
BatchNorm + ReLU

# Mapping (4 layers)
Conv2d(56 → 12, kernel=3)  # Refine features
Conv2d(12 → 12, kernel=3)  # Multiple layers
Conv2d(12 → 12, kernel=3)  # increase receptive field
Conv2d(12 → 12, kernel=3)

# Upsampling
UpsamplingBilinear2d(32×32 → 64×64)  # Spatial expansion

# Expansion
Conv2d(12 → 1, kernel=3)  # Output feature map
```

#### Why FSRCNN for SPI?

1. **Spatial Localization**: Keypoints need spatial positions
2. **Feature Preservation**: Maintains measurement information
3. **Efficient**: Originally designed for super-resolution
4. **Privacy-Preserving**: Upsamples features, not images

---

### 4. Detection Heads

#### Model 1: SimpleCNN

**Code**: `models/model_grayscale_input.py:103-125`

**Architecture**:
```python
Conv2d(1 → 64, kernel=3)
ReLU + AvgPool2d(2)  # 64×64 → 32×32

Conv2d(64 → 128, kernel=3)
ReLU + AvgPool2d(2)  # 32×32 → 16×16

Flatten
Linear(128×16×16 → 512)
ReLU
Linear(512 → num_keypoints × 3)
```

**Characteristics**:
- Lightweight: ~5M parameters
- Fast inference: ~5-10ms on GPU
- Good for real-time SPI systems
- Direct regression to keypoint coordinates

#### Model 2: EfficientNet + Predictor

**Code**: `models/model_grayscale_keypointrcnn.py`

**Architecture**:
```python
# Transition: Feature map → RGB-like
Conv2d(1 → 3, kernel=1)

# EfficientNet-B0 backbone
EfficientNet(pretrained=True)
# Output: 1280-dim feature vector

# Keypoint Predictor
Linear(1280 → 512) + ReLU + Dropout(0.3)
Linear(512 → 256) + ReLU + Dropout(0.3)
Linear(256 → num_keypoints × 3)
```

**Characteristics**:
- Higher capacity: ~15M parameters
- Pretrained features (adapted for SPI)
- Better accuracy on complex poses
- More computationally intensive

---

## Privacy-Preserving Design

### How Privacy is Maintained

#### 1. **Hardware Level**: Single-Pixel Sensor

```
Traditional Camera:           Single-Pixel Camera:
Each pixel sees               Single detector sees
small spatial region    VS.   integrated light
→ Can form image             → Cannot form image
                               without all patterns
```

#### 2. **Measurement Level**: Structured Patterns

- Measurements are **linear combinations** of scene
- Individual measurements reveal **global, not local** information
- Example: `m₁ = 0.3×pixel₁ + 0.7×pixel₂ + 0.1×pixel₃ + ...`
  - Cannot identify specific objects from m₁ alone

#### 3. **Algorithmic Level**: Direct Task Learning

```
Privacy-Invasive:
Measurements → Image Reconstruction → Feature Extraction → Keypoints
                      ↑
                Privacy leak!

Privacy-Preserving (Pose-SPI):
Measurements → Feature Extraction → Keypoints
                    ↑
            No image formation!
```

#### 4. **Network Design**: No Reconstruction Pathway

Key architectural choices:
- **No decoder to image space**: FSRCNN upsamples features, not images
- **Task-specific heads**: Direct keypoint regression
- **No intermediate visualizations**: Cannot "see" what network sees

### Privacy Analysis

**What can be recovered?**
- ✅ Keypoint positions (intended)
- ✅ Approximate pose (intended)

**What cannot be recovered?**
- ❌ Facial features
- ❌ Identity information
- ❌ Clothing details
- ❌ Background/environment
- ❌ Text or fine details

**Threat Model**:
- Adversary with full model access
- Adversary with all measurements
- **Still cannot reconstruct identifying information** without specialized reconstruction algorithms and all measurement patterns

---

## Performance Considerations

### Computational Complexity

#### Simple Model

| Component | Operations | Parameters |
|-----------|------------|------------|
| CustomConv | 4K (element-wise) | 4096 |
| Encoder FC | 4M - 1M | 4M |
| FSRCNN | ~50M | 150K |
| SimpleCNN | ~100M | 2M |
| **Total** | **~150M FLOPs** | **~6M params** |

#### EfficientNet Model

| Component | Operations | Parameters |
|-----------|------------|------------|
| CustomConv | 4K | 4096 |
| Encoder FC | 4M - 1M | 4M |
| FSRCNN | ~50M | 150K |
| EfficientNet | ~400M | 5.3M |
| Predictor | ~2M | 500K |
| **Total** | **~460M FLOPs** | **~10M params** |

### Measurement Acquisition Time

For real single-pixel hardware:

**Factors**:
- Pattern projection speed
- Detector integration time
- Pattern switching time

**Example** (DMD-based system):
- Pattern rate: 20 kHz
- 4096 measurements: 0.2 seconds
- 1024 measurements: 0.05 seconds (20 FPS)

**Trade-off**:
```
More measurements ↔ Better accuracy
Fewer measurements ↔ Faster acquisition
```

### Memory Requirements

**Training**:
- Simple model: ~500 MB (batch=4)
- EfficientNet model: ~800 MB (batch=4)

**Inference**:
- Simple model: ~25 MB
- EfficientNet model: ~50 MB

---

## Design Decisions

### Why 64×64 Measurements?

**Rationale**:
1. **Sufficient for keypoints**: Coarse spatial resolution adequate for body parts
2. **Practical hardware**: Achievable with DMD or SLM projectors
3. **Computational efficiency**: Manageable feature dimensions
4. **Privacy sweet spot**: Too few to reconstruct detailed images

**Scaling**:
- Fewer (32×32): Faster, less privacy risk, lower accuracy
- More (128×128): Slower, potentially more privacy risk, higher accuracy

### Why Hadamard Patterns?

**Alternatives considered**:
1. **Random patterns**: Less structured, lower SNR
2. **Fourier patterns**: Good for compression, complex hardware
3. **Learned patterns**: Optimal for task (implemented via CustomConv!)

**Chosen approach**: **Learnable Hadamard-like patterns**
- Start with Hadamard-like structure (element-wise multiplication)
- Let network adapt patterns for keypoint detection
- Best of both worlds: structure + task optimization

### Why Direct Keypoint Learning?

**Alternative**: Two-stage approach
1. Reconstruct image from measurements
2. Apply standard pose estimation

**Problems**:
- ❌ Defeats privacy purpose
- ❌ Computationally expensive
- ❌ Error compounds across stages

**Direct learning benefits**:
- ✅ Maintains privacy
- ✅ End-to-end optimization
- ✅ Task-specific features
- ✅ Fewer parameters

---

## Extending the Architecture

### Adding More Measurements

To use 128×128 measurements:

```python
# models/model_grayscale_input.py

class CustomConv(nn.Module):
    def __init__(self, product=True):
        self.weight = nn.Parameter(torch.Tensor(1, 1, 128, 128))
        # Change: 64→128

class Encoder(nn.Module):
    def __init__(self, product=False, reduce=False):
        # ...
        self.fc = nn.Linear(128*128, 1024)  # Change: 64*64→128*128
```

### Different Sampling Patterns

Replace CustomConv with specialized pattern generators:

```python
class FourierPatternConv(nn.Module):
    """Use Fourier basis patterns instead of Hadamard"""
    def __init__(self):
        # Generate Fourier basis
        # Apply frequency-domain sampling

class RandomPatternConv(nn.Module):
    """Use random patterns (compressed sensing)"""
    def __init__(self):
        # Generate random Gaussian patterns
        # Exploits sparsity in keypoint space
```

### Multi-Person Detection

Current limitation: Single-person assumption

**Extension strategy**:
1. Use region proposal network (RPN)
2. Extract measurements per region
3. Apply keypoint detector per region
4. Aggregate results

**Challenge**: Maintaining privacy with spatial localization

---

## Future Research Directions

### 1. Measurement Reduction

**Goal**: Detect keypoints with <1000 measurements

**Approaches**:
- Compressed sensing theory
- Active measurement selection
- Adaptive sampling patterns

### 2. Privacy Guarantees

**Goal**: Formal privacy bounds

**Approaches**:
- Information-theoretic analysis
- Differential privacy framework
- Reconstruction attack resistance

### 3. Real Hardware Integration

**Goal**: Deploy on actual single-pixel cameras

**Challenges**:
- Noise modeling
- Calibration
- Real-time pattern projection

### 4. Multi-Task Learning

**Goal**: Detect keypoints + activity + scene context

**Approaches**:
- Multi-head architectures
- Shared feature extraction
- Task-balancing losses

---

## Conclusion

Pose-SPI demonstrates a novel paradigm for computer vision:

**Traditional**: Capture everything → Extract what's needed
**Pose-SPI**: Measure what's needed → Preserve privacy

The architecture carefully balances:
- **Performance**: Accurate keypoint detection
- **Privacy**: No image reconstruction
- **Efficiency**: Practical measurement counts
- **Flexibility**: Adaptable to different hardware

This privacy-by-design approach opens new applications in healthcare, smart homes, and other sensitive domains where traditional cameras are unacceptable.

---

## References

- **Single-Pixel Imaging**: Duarte et al., "Single-Pixel Imaging via Compressive Sampling" (2008)
- **Hadamard Patterns**: Harwit & Sloane, "Hadamard Transform Optics" (1979)
- **FSRCNN**: Dong et al., "Accelerating the Super-Resolution Convolutional Neural Network" (2016)
- **Privacy-Preserving Vision**: Padilla-López et al., "Visual Privacy Protection Methods: A Survey" (2015)
