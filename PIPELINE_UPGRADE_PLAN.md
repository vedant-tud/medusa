# Pipeline Upgrade Plan: RAFT + MagNet Integration

## Status: Pending — waiting for final model weights from team

## Context

The existing `backend/pipeline.py` uses:
- Eulerian video magnification
- Farneback optical flow
- 6-channel RGB (onset + apex) input to Swin model

The new training pipeline (`trainKfold_single_stream.py` + `generate_raft_flows.py`) uses:
- MagNet motion amplification
- RAFT Large optical flow
- 3-channel `[U_flow, V_flow, onset_gray]` input

The backend must match the training pipeline exactly to work with the new model weights.

> **Note on future model:** Team is moving to dual CNN or dual Swin. The inference input format (`[U_flow, V_flow, onset_gray]`) stays the same regardless — only the model class needs swapping.

---

## New Pipeline Flow (per user video)

```
User uploads video
  → Load frames
  → Detect apex frame       (existing 3D-FFT + LBP, no change)
  → Detect onset frame      (NEW: sliding window approach)
  → Face detect & crop      (224×224, same Haar cascade logic)
  → MagNet amplification    (onset_crop → amplified_apex_crop, amp_factor=10)
  → RAFT optical flow       (onset_crop vs amplified_apex_crop → (2,H,W) array)
  → Build model input       ([U_norm, V_norm, onset_gray_norm] 3-channel)
  → Model inference         (3-class: positive / negative / surprise)
```

---

## Changes to `backend/pipeline.py`

### A. Model class: 6-channel → 3-channel

```python
class BaselineSwinMER(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        # No patch embedding adaptation needed — native 3-channel input
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
```

Update label maps to match training:
```python
ID_TO_LABEL = {0: "positive", 1: "negative", 2: "surprise"}
NORM_MEAN = NORM_STD = [0.5, 0.5, 0.5]   # training uses this, not ImageNet stats
```

### B. `load_model()` — load RAFT + MagNet alongside Swin

```python
def load_model(checkpoint_path, num_classes=3, device=None):
    # ... device detection (unchanged) ...

    # Swin model
    model = BaselineSwinMER(num_classes=num_classes, pretrained=False)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=True)
    model.to(device).eval()

    # RAFT Large
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()

    # MagNet
    motion_amp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "motion_amp")
    sys.path.insert(0, motion_amp_dir)
    from magnet import MagNet
    from callbacks import gen_state_dict
    magnet_weights = os.path.join(motion_amp_dir, "magnet_epoch12_loss7.28e-02.pth")
    magnet_model = MagNet().to(device).eval()
    magnet_model.load_state_dict(gen_state_dict(magnet_weights))

    return model, raft_model, magnet_model, device
```

### C. Onset frame detection (sliding window)

```python
def detect_onset_frame(frames, apex_idx, window_size=15):
    """
    Scan frames before the apex with a sliding window.
    Find the most temporally stable segment (lowest inter-frame variance).
    The last frame of that stable window = onset.
    Fallback to frame 0 if video is too short.
    """
    search_end = max(0, apex_idx - 5)
    if search_end < window_size:
        return 0

    best_onset = 0
    min_variance = float('inf')
    for start in range(0, search_end - window_size + 1):
        window = frames[start:start + window_size]
        diffs = [np.mean(np.abs(window[i+1].astype(float) - window[i].astype(float)))
                 for i in range(len(window) - 1)]
        variance = np.var(diffs)
        if variance < min_variance:
            min_variance = variance
            best_onset = start + window_size - 1
    return best_onset
```

### D. MagNet amplification helper

```python
def apply_magnet(onset_crop, apex_crop, magnet_model, device, amp_factor=10):
    """Returns amplified apex as uint8 RGB numpy array (H, W, 3)."""
    def to_tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

    onset_t = to_tensor(onset_crop)
    apex_t  = to_tensor(apex_crop)
    amp     = torch.tensor([amp_factor], device=device, dtype=torch.float32).view(1, 1, 1, 1)

    with torch.no_grad():
        amplified = magnet_model(onset_t, apex_t, 0, 0, amp, mode='evaluate')[0]

    return torch.clamp((amplified.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()
```

### E. RAFT optical flow helpers

```python
def _prepare_for_raft(img_np, device):
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float tensor scaled to [-1, 1]."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return 2.0 * (t / 255.0) - 1.0

def compute_raft_flow(onset_crop, apex_amplified, raft_model, device):
    """Returns raw optical flow as (2, H, W) float32 numpy array."""
    with torch.no_grad():
        preds = raft_model(_prepare_for_raft(onset_crop, device),
                           _prepare_for_raft(apex_amplified, device))
    return preds[-1][0].cpu().numpy()  # (2, H, W)
```

Keep a visualization helper for the UI:
```python
def flow_to_vis(flow):
    """(2, H, W) flow array → (H, W, 3) uint8 RGB for display."""
    u, v = flow[0], flow[1]
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

### F. `run_inference()` — new 3-channel input format

Matches `CasmeBaselineDataset.__getitem__()` from `trainKfold_single_stream.py` exactly:

```python
@torch.no_grad()
def run_inference(model, flow, onset_gray, device):
    """
    flow:       (2, H, W) float32 numpy — RAFT output
    onset_gray: (H, W)    uint8  numpy — grayscale onset crop
    """
    u, v = flow[0], flow[1]
    max_mag = np.max(np.sqrt(u**2 + v**2))
    if max_mag > 1e-5:
        u_norm = (u / max_mag + 1.0) / 2.0
        v_norm = (v / max_mag + 1.0) / 2.0
    else:
        u_norm = np.full_like(u, 0.5)
        v_norm = np.full_like(v, 0.5)
    gray_norm = onset_gray.astype(np.float32) / 255.0

    combined = np.stack([u_norm, v_norm, gray_norm], axis=-1)  # (H, W, 3)
    tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(
        torch.from_numpy(combined).permute(2, 0, 1)
    ).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_id = int(np.argmax(probs))
    # ... build and return result dict ...
```

### G. `process_video()` — updated step order

```python
def process_video(video_path, model, raft_model, magnet_model, device):
    # 1. Load frames
    # 2. Detect apex (existing 3D-FFT spotter on raw frames — Eulerian step removed)
    # 3. Detect onset (sliding window)
    # 4. Face detect & crop onset + apex (224×224)
    # 5. Apply MagNet: onset_crop → amplified_apex_crop
    # 6. Compute RAFT flow: (onset_crop, amplified_apex_crop) → (2, H, W)
    # 7. Build flow visualization for UI display
    # 8. Run inference: [U_flow, V_flow, onset_gray] → emotion
    # 9. Pack and return result dict with base64 images
```

---

## Changes to `backend/main.py`

```python
# Startup
_model = _raft_model = _magnet_model = _device = None

@app.on_event("startup")
async def startup():
    global _model, _raft_model, _magnet_model, _device
    _model, _raft_model, _magnet_model, _device = load_model(CHECKPOINT_PATH)

# Process endpoint
result = process_video(tmp_path, _model, _raft_model, _magnet_model, _device)
```

---

## Files to Edit

| File | Change |
|------|--------|
| `backend/pipeline.py` | Full rewrite of pipeline functions (see above) |
| `backend/main.py` | Update `load_model` unpack + `process_video` call |

## Files to Reference (read-only)

| File | Purpose |
|------|---------|
| `generate_raft_flows.py` | RAFT + MagNet usage patterns |
| `trainKfold_single_stream.py:292-355` | Exact preprocessing logic to replicate |
| `motion_amp/magnet.py` | MagNet architecture |
| `motion_amp/callbacks.py` | `gen_state_dict` helper |
| `motion_amp/magnet_epoch12_loss7.28e-02.pth` | MagNet weights |
