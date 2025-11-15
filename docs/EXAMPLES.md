# Usage Examples and Tutorials

**Privacy-Preserving Keypoint Detection via Single-Pixel Imaging**

This document provides practical examples and tutorials for using Pose-SPI for privacy-preserving human keypoint detection.

> **Important Context**: Pose-SPI uses **single-pixel imaging (SPI)** to detect human keypoints without capturing full images. The examples below use grayscale images to **simulate** single-pixel measurements. In real hardware deployment, you would replace image inputs with measurement vectors from a single-pixel camera. This approach preserves privacy by extracting only pose information without reconstructing identifiable visual details. See [README.md](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Training](#basic-training)
- [Custom Dataset](#custom-dataset)
- [Inference](#inference)
- [Visualization](#visualization)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Pose-SPI-main

# Install dependencies
pip install -r requirements.txt
pip install tensorboard  # If not included

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

### 2. Train on Toy Dataset (5 minutes)

```bash
# Train simple model
python train.py --epochs 100 --batch 4

# Monitor training (in separate terminal)
tensorboard --logdir tensorboard_logs

# View results at http://localhost:6006
```

### 3. Train EfficientNet Model

```bash
# Train with pretrained backbone
python train_keypointrcnn.py --epochs 100 --batch 4 --lr 0.0001
```

---

## Basic Training

### Example 1: Training with Default Settings

```bash
python train.py
```

**What happens**:
- Loads toy dataset (100 images)
- Trains for 1000 epochs
- Batch size: 4
- Learning rate: 0.0001
- Saves logs to `tensorboard_logs/`

**Expected output**:
```
Namespace(data_path='./toydataset/images/train2017', annotation_path='./toydataset/annotations/annotations.json', epochs=1000, batch=4, lr=0.0001, output_dir='weights', num_keypoints=13)
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
Epoch [1/1000], Loss: 0.4523
Epoch [1/1000], Loss: 0.3891
...
```

### Example 2: Custom Training Parameters

```bash
python train.py \
    --epochs 500 \
    --batch 8 \
    --lr 0.001 \
    --num_keypoints 17 \
    --output_dir ./my_weights
```

**Explanation**:
- `--epochs 500`: Train for 500 epochs instead of 1000
- `--batch 8`: Larger batch size (requires more GPU memory)
- `--lr 0.001`: Higher learning rate (10× default)
- `--num_keypoints 17`: Detect 17 keypoints (COCO standard)
- `--output_dir ./my_weights`: Save weights to custom directory

---

## Custom Dataset

### Example 3: Preparing Your Own Dataset

#### Step 1: Organize Your Images

```bash
my_dataset/
├── images/
│   ├── train2017/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   └── val2017/
│       ├── 100001.jpg
│       ├── 100002.jpg
│       └── ...
└── annotations/
    ├── train_annotations.json
    └── val_annotations.json
```

#### Step 2: Create COCO-Format Annotations

```python
# create_annotations.py
import json

# COCO annotation structure
annotations = {
    "images": [
        {
            "id": 1,
            "file_name": "000001.jpg",
            "height": 64,
            "width": 64
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "keypoints": [
                # [x, y, visibility] for each keypoint
                32, 20, 2,  # Nose
                30, 18, 2,  # Left eye
                34, 18, 2,  # Right eye
                # ... (13 or 17 keypoints total)
            ],
            "num_keypoints": 13,
            "bbox": [10, 5, 44, 55]  # [x, y, width, height]
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip"
            ],
            "skeleton": [
                [0, 1], [0, 2], [1, 3], [2, 4],  # Head
                [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
                [5, 6], [5, 11], [6, 12], [11, 12]  # Torso
            ]
        }
    ]
}

# Save to JSON
with open('my_dataset/annotations/train_annotations.json', 'w') as f:
    json.dump(annotations, f)
```

#### Step 3: Train on Custom Dataset

```bash
python train.py \
    --data_path ./my_dataset/images/train2017 \
    --annotation_path ./my_dataset/annotations/train_annotations.json \
    --epochs 1000 \
    --batch 4 \
    --num_keypoints 13
```

---

## Inference

### Example 4: Single Image Inference

```python
# inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.model_grayscale_input import KeypointDetection

# Load model
model = KeypointDetection(num_keypoints=13)
model.eval()

# Load weights (if available)
# checkpoint = torch.load('weights/100ckpt.pth')
# model.load_state_dict(checkpoint['model'])

# Prepare image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

image = Image.open('test_image.jpg')
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
image_tensor = image_tensor.to(device)

with torch.no_grad():
    predictions = model(image_tensor)

# Output shape: (1, 13, 3)
print(f"Predictions shape: {predictions.shape}")
print(f"First keypoint (x, y, vis): {predictions[0, 0].cpu().numpy()}")
```

### Example 5: Batch Inference

```python
# batch_inference.py
import torch
from torch.utils.data import DataLoader
from data_loader import CocoKeypointsDataset
from models.model_grayscale_input import KeypointDetection
from pycocotools.coco import COCO
import torchvision.transforms as transforms

# Setup
coco = COCO('annotations.json')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

dataset = CocoKeypointsDataset(coco, './images/', transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Load model
model = KeypointDetection(num_keypoints=13)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inference on all images
all_predictions = []
all_image_ids = []

with torch.no_grad():
    for images, _, image_ids in dataloader:
        images = images.to(device)
        predictions = model(images)

        all_predictions.append(predictions.cpu())
        all_image_ids.extend(image_ids)

# Concatenate results
all_predictions = torch.cat(all_predictions, dim=0)
print(f"Total predictions: {all_predictions.shape}")  # (num_images, 13, 3)
```

---

## Visualization

### Example 6: Visualizing Predictions

```python
# visualize.py
import torch
import matplotlib.pyplot as plt
from utils import draw_keypoints
from data_loader import CocoKeypointsDataset
from models.model_grayscale_input import KeypointDetection
from pycocotools.coco import COCO
import torchvision.transforms as transforms

# Load data
coco = COCO('toydataset/annotations/annotations.json')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
dataset = CocoKeypointsDataset(coco, 'toydataset/images/train2017/', transform)

# Load model
model = KeypointDetection(num_keypoints=13)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Get a sample
image, gt_keypoints, img_id = dataset[0]
image = image.unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    predictions = model(image)

# Visualize
result_img = draw_keypoints(
    pred_keypoints=predictions[0],
    gt_keypoints=gt_keypoints,
    images=image[0]
)

# Display
plt.figure(figsize=(10, 10))
plt.imshow(result_img.transpose(1, 2, 0))
plt.title('Red: Ground Truth, Green: Predictions')
plt.axis('off')
plt.savefig('visualization.png')
plt.show()
```

### Example 7: Creating a Video Visualization

```python
# video_visualization.py
import torch
import cv2
import numpy as np
from models.model_grayscale_input import KeypointDetection
from utils import draw_keypoints
import torchvision.transforms as transforms

# Load model
model = KeypointDetection(num_keypoints=13)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Open video
cap = cv2.VideoCapture('input_video.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
out = cv2.VideoWriter('output_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    frame_tensor = transform(frame).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        predictions = model(frame_tensor)

    # Draw keypoints on original frame
    # (Note: Need to scale predictions from 64×64 back to original size)
    scale_x = width / 64
    scale_y = height / 64

    for kpt in predictions[0]:
        x, y, vis = kpt.cpu().numpy()
        x_scaled = int(x * scale_x)
        y_scaled = int(y * scale_y)
        cv2.circle(frame, (x_scaled, y_scaled), 5, (0, 255, 0), -1)

    out.write(frame)

cap.release()
out.release()
print("Video processing complete!")
```

---

## Advanced Usage

### Example 8: Custom Loss Function

```python
# custom_training.py
import torch
import torch.nn as nn

class WeightedKeypointLoss(nn.Module):
    """Custom loss that weights important keypoints more heavily"""

    def __init__(self, keypoint_weights):
        super().__init__()
        self.weights = torch.tensor(keypoint_weights).float()

    def forward(self, predictions, targets):
        # predictions, targets: (batch, num_keypoints, 3)
        diff = (predictions - targets) ** 2  # MSE
        weighted_diff = diff * self.weights.view(1, -1, 1).to(diff.device)
        return weighted_diff.mean()

# Usage
keypoint_weights = [
    2.0,  # Nose (important)
    1.5, 1.5,  # Eyes
    1.0, 1.0,  # Ears
    2.0, 2.0,  # Shoulders (important)
    1.5, 1.5,  # Elbows
    1.5, 1.5,  # Wrists
    2.0, 2.0   # Hips (important)
]

criterion = WeightedKeypointLoss(keypoint_weights)

# In training loop
loss = criterion(predictions, targets)
```

### Example 9: Learning Rate Scheduling

```python
# scheduled_training.py
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Option 1: Step decay at specific epochs
scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)

# Option 2: Reduce on plateau
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# Training loop
for epoch in range(epochs):
    # ... training code ...

    # Step scheduler (Option 1)
    scheduler.step()

    # OR Step with validation loss (Option 2)
    # scheduler.step(val_loss)

    print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
```

### Example 10: Data Augmentation

```python
# augmented_training.py
import torchvision.transforms as transforms

# Enhanced transform with augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Note: When augmenting, you must also transform keypoints accordingly
# This requires custom dataset class
```

### Example 11: Model Ensemble

```python
# ensemble.py
import torch
from models.model_grayscale_input import KeypointDetection as SimpleModel
from models.model_grayscale_keypointrcnn import KeypointDetection as EfficientNetModel

# Load multiple models
model1 = SimpleModel(num_keypoints=13)
model2 = EfficientNetModel(num_keypoints=13)

# Load weights
model1.load_state_dict(torch.load('simple_weights.pth')['model'])
model2.load_state_dict(torch.load('efficientnet_weights.pth')['model'])

model1.eval()
model2.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.to(device)
model2.to(device)

# Ensemble inference
def ensemble_predict(image):
    with torch.no_grad():
        pred1 = model1(image)
        pred2 = model2(image)

        # Average predictions
        ensemble_pred = (pred1 + pred2) / 2

        return ensemble_pred

# Usage
image = torch.randn(1, 1, 64, 64).to(device)
predictions = ensemble_predict(image)
```

### Example 12: Export to ONNX

```python
# export_onnx.py
import torch
from models.model_grayscale_input import KeypointDetection

# Load model
model = KeypointDetection(num_keypoints=13)
model.eval()

# Dummy input
dummy_input = torch.randn(1, 1, 64, 64)

# Export
torch.onnx.export(
    model,
    dummy_input,
    "keypoint_detector.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("Model exported to keypoint_detector.onnx")

# Usage with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession("keypoint_detector.onnx")
input_name = session.get_inputs()[0].name

predictions = session.run(None, {input_name: dummy_input.numpy()})
print(f"ONNX predictions shape: {predictions[0].shape}")
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solution**:
```bash
# Reduce batch size
python train.py --batch 2

# Or use CPU
python train.py --device cpu
```

### Issue 2: Slow Training

**Problem**: Training takes too long

**Solutions**:

1. **Use GPU**:
   ```python
   # Verify GPU is being used
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Increase batch size**:
   ```bash
   python train.py --batch 8  # If GPU memory allows
   ```

3. **Reduce epochs for testing**:
   ```bash
   python train.py --epochs 100
   ```

### Issue 3: Poor Predictions

**Problem**: Model predictions are inaccurate

**Debugging Steps**:

1. **Check overfitting capability**:
   ```python
   # Train on just 5 images - model should overfit
   # Modify train.py to use subset
   train_dataset = torch.utils.data.Subset(train_dataset, range(5))
   ```

2. **Visualize predictions**:
   ```python
   # Use utils.draw_keypoints to see what model predicts
   ```

3. **Check learning rate**:
   ```bash
   # Try different learning rates
   python train.py --lr 0.001  # Higher
   python train.py --lr 0.00001  # Lower
   ```

4. **Verify data**:
   ```python
   # Check annotations are correct
   from pycocotools.coco import COCO
   coco = COCO('annotations.json')
   ann_ids = coco.getAnnIds(imgIds=[1])
   anns = coco.loadAnns(ann_ids)
   print(anns)
   ```

### Issue 4: Dimension Mismatch

**Error**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

**Cause**: Input image size doesn't match expected 64×64

**Solution**:
```python
# Ensure images are resized
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Add this
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])
```

### Issue 5: TensorBoard Not Showing Logs

**Problem**: TensorBoard shows no data

**Solutions**:

1. **Check log directory**:
   ```bash
   ls tensorboard_logs/
   # Should show event files
   ```

2. **Specify correct directory**:
   ```bash
   tensorboard --logdir tensorboard_logs
   ```

3. **Refresh browser**: Press Ctrl+R in browser

4. **Check port conflicts**:
   ```bash
   tensorboard --logdir tensorboard_logs --port 6007
   ```

---

## Best Practices

### 1. Development Workflow

```bash
# 1. Start with small overfitting test
python train.py --epochs 50 --batch 4
# Verify loss decreases to near zero

# 2. Train on full dataset
python train.py --epochs 1000 --batch 4

# 3. Monitor with TensorBoard
tensorboard --logdir tensorboard_logs

# 4. Evaluate
# Uncomment evaluation code in train.py

# 5. Visualize results
# Use draw_keypoints on validation set
```

### 2. Hyperparameter Tuning

```python
# Systematic grid search
learning_rates = [1e-5, 1e-4, 1e-3]
batch_sizes = [2, 4, 8]

for lr in learning_rates:
    for batch in batch_sizes:
        # Train with combination
        # Log results
        # Compare performance
```

### 3. Reproducibility

```python
# Always set seeds
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **COCO Dataset**: https://cocodataset.org/
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946
- **Human Pose Estimation**: https://github.com/cbsudux/awesome-human-pose-estimation

---

## Next Steps

After working through these examples, you should be able to:

1. Train models on custom datasets
2. Perform inference on new images
3. Visualize and debug predictions
4. Optimize hyperparameters
5. Deploy models in production

For more advanced topics, see:
- `docs/ARCHITECTURE.md` - Detailed architecture explanations
- `docs/API.md` - Complete API reference
