# API Documentation

This document provides detailed API documentation for the Pose-SPI project's core modules and functions.

## Table of Contents

- [Data Loading](#data-loading)
- [Models](#models)
- [Utilities](#utilities)
- [COCO Evaluation](#coco-evaluation)

---

## Data Loading

### `data_loader.py`

#### `CocoKeypointsDataset`

PyTorch Dataset class for loading COCO-format keypoint annotations.

**Class Definition:**
```python
class CocoKeypointsDataset(torch.utils.data.Dataset)
```

**Parameters:**
- `coco` (COCO): Initialized COCO object from `pycocotools.coco.COCO`
- `root_dir` (str): Root directory containing images
- `transform` (callable, optional): Optional transform to apply to images

**Methods:**

##### `__init__(self, coco, root_dir, transform=None)`
Initialize the dataset.

**Example:**
```python
from pycocotools.coco import COCO
from data_loader import CocoKeypointsDataset
import torchvision.transforms as transforms

coco = COCO('annotations.json')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
dataset = CocoKeypointsDataset(coco, './images/', transform)
```

##### `__len__(self)`
Returns the total number of images in the dataset.

**Returns:**
- `int`: Number of images

##### `__getitem__(self, idx)`
Retrieves an image and its keypoint annotations.

**Parameters:**
- `idx` (int): Index of the sample

**Returns:**
- `tuple`: (image, keypoints, image_id)
  - `image` (Tensor): Transformed image tensor
  - `keypoints` (Tensor): Keypoint tensor of shape (num_keypoints, 3) containing [x, y, visibility]
  - `image_id` (int): COCO image ID

**Example:**
```python
image, keypoints, img_id = dataset[0]
# image: torch.Tensor of shape (1, 64, 64) for grayscale
# keypoints: torch.Tensor of shape (num_keypoints, 3)
# img_id: int
```

---

## Models

### `models/model_grayscale_input.py`

Simple CNN-based keypoint detection model.

#### `CustomConv`

Learnable weight multiplication or convolution with optional upsampling.

**Class Definition:**
```python
class CustomConv(nn.Module)
```

**Parameters:**
- `input_channels` (int): Number of input channels
- `output_channels` (int): Number of output channels
- `upsample` (bool, optional): Whether to apply bilinear upsampling (default: False)

**Forward:**
- **Input**: `(batch_size, input_channels, H, W)`
- **Output**: `(batch_size, output_channels, H*2, W*2)` if upsample=True, else `(batch_size, output_channels, H, W)`

---

#### `Encoder`

Spatial dimension reduction using convolution, max pooling, and fully connected layers.

**Class Definition:**
```python
class Encoder(nn.Module)
```

**Parameters:**
- `input_channels` (int): Number of input channels (default: 1 for grayscale)

**Architecture:**
- CustomConv: 1 → 64 channels
- MaxPool: 64×64 → 32×32
- Fully connected: Flattened features → 512 → 256

**Forward:**
- **Input**: `(batch_size, 1, 64, 64)`
- **Output**: `(batch_size, 256)`

---

#### `FSRCNN`

Fast Super-Resolution CNN for upsampling feature maps.

**Class Definition:**
```python
class FSRCNN(nn.Module)
```

**Parameters:**
- `input_dim` (int): Input feature dimension (default: 256)
- `scale_factor` (int): Upsampling scale factor (default: 2)

**Architecture:**
1. **Feature Extraction**: Conv(3×3) → BatchNorm → ReLU
2. **Mapping Layers**: Multiple Conv(3×3) → BatchNorm → ReLU blocks
3. **Deconvolution**: Bilinear upsampling with learnable weights
4. **Expansion**: Conv(1×1) to expand channels

**Forward:**
- **Input**: `(batch_size, 256)`
- **Output**: `(batch_size, 64, 64, 64)` feature map

---

#### `SimpleCNN`

Lightweight keypoint detection head.

**Class Definition:**
```python
class SimpleCNN(nn.Module)
```

**Parameters:**
- `num_keypoints` (int): Number of keypoints to detect (default: 13)

**Architecture:**
- Conv1: 64 → 32 channels (3×3 kernel)
- AvgPool: 64×64 → 32×32
- Conv2: 32 → num_keypoints×3 channels (3×3 kernel)
- AvgPool: 32×32 → 1×1

**Forward:**
- **Input**: `(batch_size, 64, 64, 64)`
- **Output**: `(batch_size, num_keypoints, 3)` containing [x, y, visibility] for each keypoint

---

#### `KeypointDetection`

Complete keypoint detection model combining all components.

**Class Definition:**
```python
class KeypointDetection(nn.Module)
```

**Parameters:**
- `input_channels` (int): Number of input image channels (default: 1)
- `num_keypoints` (int): Number of keypoints to predict (default: 13)

**Forward:**
- **Input**: `(batch_size, 1, 64, 64)` grayscale images
- **Output**: `(batch_size, num_keypoints, 3)` keypoint predictions

**Example:**
```python
from models.model_grayscale_input import KeypointDetection

model = KeypointDetection(num_keypoints=13)
images = torch.randn(4, 1, 64, 64)  # Batch of 4 grayscale images
predictions = model(images)  # Shape: (4, 13, 3)
```

---

### `models/model_grayscale_keypointrcnn.py`

EfficientNet-based keypoint detection model with RCNN-style head.

#### `KeypointRCNN`

EfficientNet backbone feature extractor.

**Class Definition:**
```python
class KeypointRCNN(nn.Module)
```

**Parameters:**
- `num_classes` (int): Number of output classes
- `pretrained` (bool): Whether to use pretrained EfficientNet weights (default: True)

**Architecture:**
- Pretrained EfficientNet-B0 backbone
- Modified for single-channel (grayscale) input
- Feature extraction without classification head

**Forward:**
- **Input**: `(batch_size, 1, 64, 64)` grayscale images
- **Output**: `(batch_size, feature_dim)` feature vector

---

#### `KeypointPredictor`

RCNN-style regression head for keypoint prediction.

**Class Definition:**
```python
class KeypointPredictor(nn.Module)
```

**Parameters:**
- `in_features` (int): Input feature dimension
- `num_keypoints` (int): Number of keypoints to predict

**Architecture:**
- Fully connected layers with ReLU activation
- Dropout for regularization
- Output layer: `num_keypoints × 3` for [x, y, visibility]

**Forward:**
- **Input**: `(batch_size, in_features)`
- **Output**: `(batch_size, num_keypoints, 3)`

---

#### `KeypointRCNNLoss`

Custom loss function for keypoint detection.

**Class Definition:**
```python
class KeypointRCNNLoss(nn.Module)
```

**Parameters:**
- `loss_type` (str): Type of loss - 'l1' or 'smooth_l1' (default: 'smooth_l1')

**Forward:**
- **Input**:
  - `predictions` (Tensor): Predicted keypoints `(batch_size, num_keypoints, 3)`
  - `targets` (Tensor): Ground truth keypoints `(batch_size, num_keypoints, 3)`
- **Output**: `(Tensor)` Scalar loss value

**Example:**
```python
criterion = KeypointRCNNLoss(loss_type='smooth_l1')
loss = criterion(predictions, ground_truth)
```

---

#### `KeypointDetection`

Complete EfficientNet-based keypoint detection model.

**Class Definition:**
```python
class KeypointDetection(nn.Module)
```

**Parameters:**
- `num_keypoints` (int): Number of keypoints to detect (default: 13)
- `pretrained` (bool): Use pretrained EfficientNet (default: True)

**Architecture:**
1. Encoder (same as simple model)
2. FSRCNN upsampling
3. Transition convolution
4. EfficientNet backbone
5. KeypointPredictor head

**Forward:**
- **Input**: `(batch_size, 1, 64, 64)` grayscale images
- **Output**: `(batch_size, num_keypoints, 3)` keypoint predictions

**Example:**
```python
from models.model_grayscale_keypointrcnn import KeypointDetection

model = KeypointDetection(num_keypoints=13, pretrained=True)
images = torch.randn(4, 1, 64, 64)
predictions = model(images)  # Shape: (4, 13, 3)
```

---

## Utilities

### `utils.py`

#### `AverageMeter`

Tracks and computes average values over iterations.

**Class Definition:**
```python
class AverageMeter:
```

**Methods:**

##### `__init__(self)`
Initialize the meter.

##### `reset(self)`
Reset all statistics.

##### `update(self, val, n=1)`
Update the meter with a new value.

**Parameters:**
- `val` (float): Value to add
- `n` (int): Number of items represented by this value (for weighted average)

**Attributes:**
- `avg`: Current average
- `sum`: Running sum
- `count`: Total count

**Example:**
```python
loss_meter = AverageMeter()
for batch in dataloader:
    loss = compute_loss(batch)
    loss_meter.update(loss.item(), batch_size)
print(f"Average loss: {loss_meter.avg}")
```

---

#### `draw_keypoints()`

Visualizes predicted and ground truth keypoints on images.

**Function Signature:**
```python
def draw_keypoints(pred_keypoints, gt_keypoints, images)
```

**Parameters:**
- `pred_keypoints` (Tensor): Predicted keypoints `(num_keypoints, 3)` or `(num_keypoints,)` for flattened
- `gt_keypoints` (Tensor): Ground truth keypoints `(num_keypoints, 3)` or `(num_keypoints,)` for flattened
- `images` (Tensor): Image tensor `(C, H, W)`

**Returns:**
- `np.ndarray`: Image array with drawn keypoints and skeleton, shape `(3, H, W)`

**Visualization:**
- Red circles: Ground truth keypoints
- Green circles: Predicted keypoints
- Blue lines: Ground truth skeleton connections
- Cyan lines: Predicted skeleton connections

**Example:**
```python
from utils import draw_keypoints

result_img = draw_keypoints(
    pred_keypoints=model_output[0],
    gt_keypoints=ground_truth[0],
    images=images[0]
)
# Use with TensorBoard
writer.add_image('keypoints', result_img, global_step=step)
```

---

#### `evaluate_keypoint_detection()`

Performs COCO-style evaluation for keypoint detection.

**Function Signature:**
```python
def evaluate_keypoint_detection(model, device, ann_file, data_loader, num_keypoints)
```

**Parameters:**
- `model` (nn.Module): Trained keypoint detection model
- `device` (torch.device): Device to run evaluation on (CPU/GPU)
- `ann_file` (str): Path to COCO annotations JSON file
- `data_loader` (DataLoader): DataLoader for evaluation dataset
- `num_keypoints` (int): Number of keypoints in the model

**Process:**
1. Runs inference on all images in data_loader
2. Converts predictions to COCO format
3. Computes COCO evaluation metrics (AP, AR at various thresholds)

**Prints:**
- Average Precision (AP) @ IoU=0.50:0.95
- AP @ IoU=0.50
- AP @ IoU=0.75
- AP for medium objects
- AP for large objects
- Average Recall (AR) @ IoU=0.50:0.95
- AR @ maxDets=20
- AR @ maxDets=50
- AR for medium objects
- AR for large objects

**Example:**
```python
from utils import evaluate_keypoint_detection

evaluate_keypoint_detection(
    model=trained_model,
    device=torch.device('cuda'),
    ann_file='annotations.json',
    data_loader=val_loader,
    num_keypoints=13
)
```

---

## COCO Evaluation

### `coco_eval/coco_eval.py`

Implements COCO evaluation metrics for keypoint detection using Object Keypoint Similarity (OKS).

#### `COCOeval`

Main evaluation class for keypoint detection.

**Key Methods:**
- `evaluate()`: Run evaluation on all images
- `accumulate()`: Accumulate evaluation results
- `summarize()`: Print summary statistics

**Keypoint-Specific Features:**
- Uses OKS (Object Keypoint Similarity) instead of IoU
- Configurable `kpt_oks_sigmas` for different keypoint types
- Empirical sigma values for standard human keypoints

**OKS Formula:**
```
OKS = Σ exp(-di²/(2*s²*κi²)) * δ(vi>0) / Σ δ(vi>0)
```
Where:
- `di`: Euclidean distance between predicted and ground truth keypoint
- `s`: Object scale (√(area))
- `κi`: Per-keypoint constant (kpt_oks_sigmas)
- `vi`: Visibility flag

---

### `coco_eval/mask_cocoeval.py`

Utilities for Run-Length Encoding (RLE) masks used in COCO evaluation.

#### Key Functions:

##### `encode(mask)`
Converts binary mask to RLE format.

##### `decode(rle)`
Converts RLE format back to binary mask.

##### `area(rle)`
Computes area of RLE-encoded mask.

##### `iou(rle1, rle2, pyiscrowd)`
Computes Intersection over Union between two RLE masks.

---

## Training Scripts

### `train.py`

Main training script for the simple keypoint detector.

**Command-line Arguments:**
```bash
python train.py [OPTIONS]
```

**Options:**
- `--data_path`: Path to images directory (default: './toydataset/images/train2017')
- `--annotation_path`: Path to annotations JSON (default: './toydataset/annotations/annotations.json')
- `--epochs`: Number of training epochs (default: 1000)
- `--batch`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--output_dir`: Output directory for weights (default: 'weights')
- `--num_keypoints`: Number of keypoints (default: 13)

**Example:**
```bash
python train.py --epochs 500 --batch 8 --lr 0.001
```

---

### `train_keypointrcnn.py`

Training script for EfficientNet-based model.

**Command-line Arguments:** Same as `train.py`

**Differences from `train.py`:**
- Uses `KeypointRCNNLoss` for training
- Loads pretrained EfficientNet backbone
- May require more GPU memory

**Example:**
```bash
python train_keypointrcnn.py --epochs 500 --num_keypoints 17
```

---

## Usage Examples

### Complete Training Pipeline

```python
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pycocotools.coco import COCO

from data_loader import CocoKeypointsDataset
from models.model_grayscale_input import KeypointDetection
from utils import AverageMeter, evaluate_keypoint_detection

# Initialize COCO
coco = COCO('annotations.json')

# Create dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
dataset = CocoKeypointsDataset(coco, './images/', transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize model
model = KeypointDetection(num_keypoints=13)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(100):
    model.train()
    loss_meter = AverageMeter()

    for images, keypoints, _ in dataloader:
        images, keypoints = images.to(device), keypoints.to(device)

        outputs = model(images)
        loss = criterion(outputs, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))

    print(f"Epoch {epoch}, Loss: {loss_meter.avg}")

# Evaluation
evaluate_keypoint_detection(model, device, 'annotations.json', dataloader, 13)
```

---

## Notes

- All models expect input images of size **64×64 pixels**
- Keypoint format is **[x, y, visibility]** per keypoint
- Visibility values: 0 (not labeled), 1 (labeled but not visible), 2 (labeled and visible)
- Training uses deterministic settings for reproducibility
- GPU is automatically detected and used if available
