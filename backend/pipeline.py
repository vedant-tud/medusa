"""
pipeline.py — End-to-end inference pipeline
Video → Magnify → Apex Detection → Optical Flow → Model → Emotion
"""

import os
import sys
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional

# Add parent dir so we can import from train.py and apex_frame_detection.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm
from apex_frame_detection import ApexFrameSpotter

# ── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

ID_TO_LABEL = {
    0: "happiness",
    1: "disgust",
    2: "surprise",
    3: "anger",
    4: "fear",
    5: "sadness",
    6: "others",
}

# Map 7 fine-grained classes → 3 user-facing categories
EMOTION_GROUP = {
    "happiness": "positive",
    "disgust":   "negative",
    "surprise":  "surprised",
    "anger":     "negative",
    "fear":      "negative",
    "sadness":   "negative",
    "others":    "negative",
}

GROUP_COLORS = {
    "positive":  "#4CAF50",
    "negative":  "#F44336",
    "surprised": "#FF9800",
}

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Model ────────────────────────────────────────────────────────────────────

class BaselineSwinMER(nn.Module):
    """6-channel Swin Tiny for micro-expression recognition."""

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self._adapt_patch_embedding()
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def _adapt_patch_embedding(self):
        old = self.backbone.patch_embed.proj
        w   = old.weight.data
        new_conv = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                             old.padding, bias=(old.bias is not None))
        new_conv.weight.data = torch.cat([w, w], dim=1) * 0.5
        if old.bias is not None:
            new_conv.bias.data = old.bias.data.clone()
        self.backbone.patch_embed.proj = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ── Face helpers ─────────────────────────────────────────────────────────────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def align_and_crop_face(img_rgb: np.ndarray, target_size: int = 224) -> np.ndarray:
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        m  = int(0.15 * min(w, h))
        x1, y1 = max(0, x - m), max(0, y - m)
        x2 = min(img_rgb.shape[1], x + w + m)
        y2 = min(img_rgb.shape[0], y + h + m)
        crop = img_rgb[y1:y2, x1:x2]
    else:
        h, w = img_rgb.shape[:2]
        s    = min(h, w)
        crop = img_rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(crop, (target_size, target_size))


# ── Video magnification ───────────────────────────────────────────────────────

def magnify_video(frames: list[np.ndarray], alpha: float = 20.0,
                  freq_lo: float = 0.4, freq_hi: float = 3.0,
                  fps: float = 30.0) -> list[np.ndarray]:
    """
    Simplified Eulerian Video Magnification (color / temporal bandpass).
    Amplifies subtle colour changes (micro-expressions) by factor alpha.
    """
    if len(frames) < 2:
        return frames

    h, w = frames[0].shape[:2]
    # Pyramid level-1: downsample for efficiency
    ph, pw = max(h // 4, 1), max(w // 4, 1)

    small = np.stack(
        [cv2.resize(f.astype(np.float32), (pw, ph)) for f in frames], axis=0
    )  # (T, ph, pw, 3)

    # IIR temporal bandpass = difference of two low-pass EMA filters
    r_hi = max(0.0, 1.0 - 2 * np.pi * freq_hi / fps)
    r_lo = max(0.0, 1.0 - 2 * np.pi * freq_lo / fps)

    low_hi = small[0].copy()
    low_lo = small[0].copy()
    filtered = np.zeros_like(small)

    for i, s in enumerate(small):
        low_hi = r_hi * low_hi + (1 - r_hi) * s
        low_lo = r_lo * low_lo + (1 - r_lo) * s
        filtered[i] = low_hi - low_lo  # bandpass component

    magnified = []
    for i, frame in enumerate(frames):
        amp    = cv2.resize(filtered[i], (w, h))
        result = np.clip(frame.astype(np.float32) + alpha * amp, 0, 255).astype(np.uint8)
        magnified.append(result)

    return magnified


# ── Optical flow ──────────────────────────────────────────────────────────────

def compute_optical_flow_vis(onset_rgb: np.ndarray, apex_rgb: np.ndarray) -> np.ndarray:
    """Dense Farneback optical flow between onset and apex, visualised as HSV→RGB."""
    g1 = cv2.cvtColor(onset_rgb, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(apex_rgb,  cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*onset_rgb.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ── Encode image as base64 PNG ────────────────────────────────────────────────

def encode_b64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ── Inference ────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None, num_classes: int = 7,
               device: Optional[torch.device] = None) -> BaselineSwinMER:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = BaselineSwinMER(num_classes=num_classes, pretrained=False)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[pipeline] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[pipeline] No checkpoint — using untrained weights (demo mode).")

    model.to(device).eval()
    return model, device


@torch.no_grad()
def run_inference(model: BaselineSwinMER, onset_rgb: np.ndarray,
                  apex_rgb: np.ndarray, device: torch.device) -> dict:
    onset_t = _transform(onset_rgb)   # (3, H, W)
    apex_t  = _transform(apex_rgb)    # (3, H, W)
    x = torch.cat([onset_t, apex_t], dim=0).unsqueeze(0).to(device)  # (1, 6, H, W)

    logits = model(x)
    probs  = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_id    = int(np.argmax(probs))
    pred_label = ID_TO_LABEL[pred_id]
    pred_group = EMOTION_GROUP[pred_label]
    confidence = probs[pred_id]

    # Per-group probabilities
    group_probs: dict[str, float] = {"positive": 0.0, "negative": 0.0, "surprised": 0.0}
    for i, p in enumerate(probs):
        grp = EMOTION_GROUP[ID_TO_LABEL[i]]
        group_probs[grp] += p

    return {
        "emotion":       pred_label,
        "group":         pred_group,
        "confidence":    round(confidence, 4),
        "group_probs":   {k: round(v, 4) for k, v in group_probs.items()},
        "all_probs":     {ID_TO_LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

def _log(step: int, total: int, msg: str) -> None:
    print(f"  [{step}/{total}] {msg}", flush=True)


def process_video(video_path: str, model: BaselineSwinMER, device: torch.device,
                  magnify_alpha: float = 20.0) -> dict:
    """
    Full pipeline: video file → dict with results and base64 images.
    """
    TOTAL = 7
    print("\n── Pipeline started ─────────────────────────────", flush=True)

    # 1. Load frames
    _log(1, TOTAL, "Loading frames from video…")
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise ValueError("Could not read any frames from the video.")
    print(f"      → {len(frames)} frames at {fps:.1f} fps", flush=True)

    # 2. Magnify video
    _log(2, TOTAL, f"Eulerian video magnification (α={magnify_alpha})…")
    magnified_frames = magnify_video(frames, alpha=magnify_alpha, fps=fps)

    # 3. Apex frame detection on magnified frames
    _log(3, TOTAL, "Apex frame detection (LBP + 3D-FFT)…")
    spotter  = ApexFrameSpotter(t_window=min(61, len(magnified_frames)))
    apex_idx, apex_score, _ = spotter.find_apex_in_short_video(magnified_frames)
    onset_idx = 0
    print(f"      → apex frame: {apex_idx}  (score={apex_score:.2f})", flush=True)

    onset_raw = frames[onset_idx]
    apex_raw  = frames[apex_idx]
    apex_mag  = magnified_frames[apex_idx]

    # 4. Face detection & crop
    _log(4, TOTAL, "Face detection & cropping (onset + apex)…")
    onset_cropped = align_and_crop_face(onset_raw)
    apex_cropped  = align_and_crop_face(apex_raw)

    # 5. Optical flow
    _log(5, TOTAL, "Computing dense optical flow (Farneback)…")
    flow_vis = compute_optical_flow_vis(onset_cropped, apex_cropped)

    # 6. Model inference
    _log(6, TOTAL, "Running model inference (Swin Transformer)…")
    result = run_inference(model, onset_cropped, apex_cropped, device)
    print(f"      → {result['emotion']} ({result['group']}, conf={result['confidence']:.3f})", flush=True)

    # 7. Pack images
    _log(7, TOTAL, "Encoding result images…")
    result.update({
        "apex_frame_index": int(apex_idx),
        "total_frames":     len(frames),
        "fps":              round(fps, 2),
        "images": {
            "onset":        encode_b64(onset_raw),
            "apex":         encode_b64(apex_raw),
            "onset_face":   encode_b64(onset_cropped),
            "apex_face":    encode_b64(apex_cropped),
            "magnified":    encode_b64(apex_mag),
            "optical_flow": encode_b64(flow_vis),
        },
    })

    print("── Pipeline complete ────────────────────────────\n", flush=True)
    return result
