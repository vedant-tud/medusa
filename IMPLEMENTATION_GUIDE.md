# Medusa Implementation Guide

This branch contains the complete folder structure and setup instructions for testing the emotion recognition pipeline. Follow this guide to set up and run the system on your local machine.

## Project Overview

**Medusa** is an emotion recognition system that analyzes micro-expressions in videos. The pipeline processes video frames to detect emotional states using:
- **TVL1 Optical Flow** (5x amplification) for motion encoding
- **DualStreamModel** (SwinV2 Tiny + ResNet50) for multi-stream emotion classification
- **MTCNN** for face detection and landmark-based alignment
- **ApexFrameSpotter** (3D-FFT + LBP) for apex frame detection

## System Requirements

- **Python**: 3.10+ (tested with 3.13)
- **CUDA**: 11.8+ (recommended for GPU acceleration)
- **RAM**: 8GB+ (16GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

## Folder Structure

```
medusa/
├── backend/                          # Backend inference module
│   ├── __init__.py
│   ├── main.py                       # FastAPI endpoints
│   ├── pipeline.py                   # Main inference pipeline (TVL1 + DualStreamModel)
│   ├── test_imports.py               # Dependency check script
│   └── test_inference.py             # Single-video test script
├── exp_02_emotion_transformer/       # Training reference — DO NOT MODIFY
│   ├── models.py                     # DualStreamModel architecture
│   ├── utils.py                      # Utilities (flow_to_rgb_heatmap, etc.)
│   ├── dataset.py                    # Training dataset
│   ├── best_model_fold4.pth          # Pre-trained checkpoint
│   └── ...
├── apex_frame_detection.py           # Apex frame detection (3D-FFT + LBP)
├── generate_tvl1_flows_amp5.py       # Reference: TVL1 flow generation
├── trainKfold_single_stream.py       # Reference: training pipeline
├── IMPLEMENTATION_GUIDE.md           # This file
├── PIPELINE_UPGRADE_PLAN.md          # Technical specifications (archived)
├── requirements.txt                  # Python dependencies
├── setup.sh                          # Environment setup script
└── README.md                         # Project overview
```

## Setup Instructions

### 1. Clone & Environment Setup

```bash
# Navigate to the repository
cd medusa

# Make setup script executable
chmod +x setup.sh

# Run setup script (creates venv and installs dependencies)
./setup.sh
```

Or manually:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Activate environment
source venv/bin/activate

# Check all dependencies
python backend/test_imports.py
```

Expected output:
```
=== Medusa Dependency Check ===

Required Dependencies:
✓ PyTorch              (v2.0.0)
✓ TorchVision         (v0.15.0)
✓ NumPy                (v1.24.0)
✓ OpenCV              (v4.8.0)
✓ scikit-image        (v0.21.0)
✓ timm (Vision Models) (v0.9.0)
✓ FastAPI            (v0.104.0)
✓ Uvicorn            (v0.24.0)

Optional Dependencies:
✓ facenet_pytorch (MTCNN) (v2.5.0)
✓ Jupyter              (v1.0.0)
✓ Matplotlib          (v3.7.0)

=== Summary ===
Required: 8/8 ✓
Optional: 3/3 ✓

✓ All required dependencies installed!
```

### 3. Prepare Model Checkpoint

The inference pipeline requires `best_model_fold4.pth` from `exp_02_emotion_transformer/`:

```bash
# The checkpoint should already be in the repo
# Verify it exists:
ls -lh exp_02_emotion_transformer/best_model_fold4.pth
```

The checkpoint contains the trained DualStreamModel weights.

## Running Inference

### Quick Test (Single Video)

```bash
python backend/test_inference.py \
    --video path/to/test_video.mp4 \
    --model exp_02_emotion_transformer/best_model_fold4.pth
```

Example with sample video:
```bash
python backend/test_inference.py \
    --video test_video.mp4 \
    --model exp_02_emotion_transformer/best_model_fold4.pth \
    --device cuda
```

### Using the Backend API

Start the FastAPI server:

```bash
uvicorn backend.main:app --reload --port 8000
```

Send a video for processing:

```bash
curl -X POST -F "file=@video.mp4" http://localhost:8000/process
```

Response example:
```json
{
  "emotion": "positive",
  "group": "positive",
  "confidence": 0.87,
  "group_probs": {
    "positive": 0.87,
    "negative": 0.08,
    "surprise": 0.05
  },
  "all_probs": {
    "positive": 0.87,
    "negative": 0.08,
    "surprise": 0.05
  },
  "apex_frame_index": 42,
  "onset_frame_index": 15,
  "total_frames": 120,
  "fps": 30.0,
  "images": {
    "onset": "data:image/jpeg;base64,...",
    "apex": "data:image/jpeg;base64,...",
    "onset_face": "data:image/jpeg;base64,...",
    "apex_face": "data:image/jpeg;base64,...",
    "optical_flow": "data:image/jpeg;base64,..."
  }
}
```

## Pipeline Flow

For each video:

1. **Load Frames** (line 271)
   - Read all frames via `cv2.VideoCapture`
   - Convert BGR → RGB
   - Output: list of `(H, W, 3)` uint8 arrays

2. **Apex Frame Detection** (line 284)
   - 3D-FFT + LBP temporal analysis
   - Finds frame with maximum micro-expression intensity
   - Output: `apex_idx`, `apex_score`

3. **Onset Frame Detection** (line 290)
   - Sliding window (default: 15 frames) before apex
   - Finds most temporally stable segment (lowest inter-frame variance)
   - Returns last frame of that window as onset
   - Output: `onset_idx`

4. **Face Detection & Crop** (line 296)
   - MTCNN detection on both onset and apex frames
   - Center 90% crop + resize to 256×256
   - Output: `(256, 256, 3)` uint8 arrays for both frames

5. **Optional: Face Alignment** (line 301)
   - MTCNN landmarks-based affine alignment
   - Aligns apex onto onset coordinate frame
   - Improves flow consistency

6. **TVL1 Optical Flow** (line 307)
   - Computes flow: onset_crop → apex_crop
   - 5x amplification factor
   - Output: `(2, 256, 256)` float32 array `[U, V]`

7. **Model Inference** (line 315)
   - **Flow Stream**: `(2, H, W)` → `flow_to_rgb_heatmap()` → `(3, H, W)` → ImageNet normalize
   - **Spatial Stream**: `(H, W, 3)` apex_crop → resize 256×256 → ImageNet normalize
   - **DualStreamModel**: combines both streams → logits → softmax
   - Output: emotion class + probabilities

## Model Architecture

### DualStreamModel

From `exp_02_emotion_transformer/models.py`:

- **Flow Branch**: SwinV2 Tiny backbone on flow RGB heatmap
- **Spatial Branch**: ResNet50 backbone on RGB frame
- **Fusion**: Concatenate features + FC layers
- **Output**: 3-class logits (positive, negative, surprise)

### Input Preprocessing

**Flow Stream**:
```python
# (2, H, W) TVL1 flow → (3, H, W) RGB heatmap
flow_heatmap = flow_to_rgb_heatmap(flow)  # From exp_02/utils.py

# Normalize: ImageNet statistics
tensor_flow = (rgb_heatmap - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

**Spatial Stream**:
```python
# (H, W, 3) uint8 → (H, W, 3) float [0, 1] → ImageNet normalize
apex_resized = cv2.resize(apex_rgb, (256, 256))
tensor_spatial = (apex_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

## Testing Checklist

Before running on production:

- [ ] `python backend/test_imports.py` passes all checks
- [ ] Checkpoint `exp_02_emotion_transformer/best_model_fold4.pth` exists
- [ ] Test video loads (check frame count, resolution)
- [ ] Apex frame detection returns valid index (0 ≤ apex_idx < num_frames)
- [ ] Onset frame detection returns valid index (0 ≤ onset_idx < apex_idx)
- [ ] Face crops are valid `(256, 256, 3)` uint8 arrays
- [ ] TVL1 flow output is `(2, 256, 256)` float32
- [ ] Model inference returns valid probabilities (sum to 1.0, all ≥ 0)
- [ ] API endpoint responds with emotion + confidence
- [ ] Base64 images decode correctly

## Performance Expectations

On NVIDIA GPU (RTX 3080, 1080p input):

| Step | Time | Notes |
|------|------|-------|
| Video load + RGB conversion | 0.5-1.5s | Depends on codec, duration |
| Apex detection (3D-FFT) | 0.5-2s | Window size based on video length |
| Onset detection (sliding window) | 0.3-1s | Window=15, typically 1 pass |
| MTCNN face detection + crop | 0.2-0.5s | Per frame, cached detector |
| TVL1 optical flow | 0.5-1.5s | 256×256 images |
| DualStreamModel inference | 0.1-0.2s | Batch size 1 |
| **Total** | **2-6s** | Typical end-to-end |

**CPU inference**: 5-10x slower (15-60s total)

## Troubleshooting

### ImportError: No module named 'facenet_pytorch'

**Issue**: MTCNN face detection not available

**Solution**: 
```bash
pip install facenet-pytorch
```

If still fails, face crops will use center 90% crop without landmark alignment (still works, slightly lower quality).

### RuntimeError: CUDA out of memory

**Issue**: GPU memory exhausted

**Solutions**:
1. Reduce crop size in `pipeline.py` line 303/304: change 256 → 192
2. Use CPU: `--device cpu`
3. Close other GPU programs

### FileNotFoundError: Video codec issue

**Issue**: `cv2.VideoCapture` can't read video format

**Solutions**:
1. Re-encode video: `ffmpeg -i video.mp4 -c:v libx264 -c:a aac output.mp4`
2. Check codec: `ffprobe video.mp4`
3. Ensure file exists and path is correct

### Model accuracy seems low

**Common causes**:
1. Video too short (< 30 frames) — onset/apex detection may fail
2. Face not visible or partially occluded — MTCNN won't detect
3. Wrong checkpoint loaded — verify `best_model_fold4.pth` path
4. Preprocessing mismatch — check ImageNet normalization in `run_inference()`

## Development Workflow

### Testing a Single Step

```python
import cv2
import torch
from backend.pipeline import load_model, crop_face_mtcnn, get_tvl1_flow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, device = load_model("exp_02_emotion_transformer/best_model_fold4.pth", device=device)

# Load test frames
cap = cv2.VideoCapture("test_video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Test individual steps
onset_crop = crop_face_mtcnn(frames[10], target_size=256)
apex_crop = crop_face_mtcnn(frames[50], target_size=256)
flow = get_tvl1_flow(onset_crop, apex_crop, amp_factor=5.0)
print(f"Flow shape: {flow.shape}, dtype: {flow.dtype}")
```

### Visualizing Flow Output

```python
import matplotlib.pyplot as plt
from backend.pipeline import _flow_to_vis

# Assuming 'flow' is (2, H, W) numpy array from get_tvl1_flow()
vis = _flow_to_vis(flow)
plt.imshow(vis)
plt.title("Optical Flow Visualization")
plt.show()
```

### Adding Custom Preprocessing

Edit `backend/pipeline.py`:
- Lines 303-304: Face crop size
- Line 314: TVL1 amplification factor
- Line 194: Image preprocessing size
- Lines 204-217: Normalization constants

## File References

| File | Purpose | Key Lines |
|------|---------|-----------|
| `backend/pipeline.py` | Main inference pipeline | 252-364 |
| `backend/main.py` | FastAPI endpoints | — |
| `exp_02_emotion_transformer/models.py` | DualStreamModel class | — |
| `exp_02_emotion_transformer/utils.py` | `flow_to_rgb_heatmap()` | — |
| `apex_frame_detection.py` | Apex detection | — |

## References

- **TVL1 Optical Flow**: OpenCV documentation on DualTVL1OpticalFlow
- **DualStreamModel**: Training code in `exp_02_emotion_transformer/`
- **MTCNN**: [Joint Face Detection and Alignment using Multi-task Cascaded CNNs](https://arxiv.org/abs/1604.02878)
- **Swin Transformer**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

---

**Status**: Ready for testing  
**Last Updated**: 2026-04-04  
**Maintained By**: Team
