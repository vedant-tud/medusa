"""
pipeline.py — End-to-end inference pipeline
Video → Apex/Onset Detection → MagNet Amplification → RAFT Flow → DualCnnMER → Emotion
"""

import os
import sys
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Add parent dir so we can import apex_frame_detection and motion_amp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex_frame_detection import ApexFrameSpotter

# ── Constants ────────────────────────────────────────────────────────────────

ID_TO_LABEL = {0: "positive", 1: "negative", 2: "surprise"}

# ── Model architecture (verbatim from train_dual_cnn.py) ─────────────────────

class FlowCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNNStream(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.features = nn.Sequential(
            FlowCNNBlock(in_channels, 32,  pool=True),
            FlowCNNBlock(32,  64,  pool=True),
            FlowCNNBlock(64,  128, pool=True),
            FlowCNNBlock(128, 128, pool=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x):
        return self.features(x)


class ChannelAttention2D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.fc(self.avg_pool(x).view(b, c))
        y_max = self.fc(self.max_pool(x).view(b, c))
        weight = self.sigmoid(y_avg + y_max).view(b, c, 1, 1)
        return x * weight


class CNNFlowDecoder(nn.Module):
    def __init__(self, in_features=256, base_channels=128, out_channels=2):
        super().__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(in_features, base_channels * 4 * 4)
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.base_channels, 4, 4)
        return self.deconv(x)


class DualCnnMER(nn.Module):
    def __init__(self, num_classes: int = 3, spatial_chans: int = 3):
        super().__init__()
        self.stream_motion  = CustomCNNStream(in_channels=2)
        self.stream_spatial = CustomCNNStream(in_channels=spatial_chans)
        self.flow_decoder   = CNNFlowDecoder(in_features=256, base_channels=128, out_channels=2)
        self.cross_attention = ChannelAttention2D(in_channels=256, reduction=8)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(2 * 128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, flow_x: torch.Tensor, spatial_x: torch.Tensor,
                return_flow: bool = False, return_feat: bool = False):
        f_motion  = self.stream_motion(flow_x)
        f_spatial = self.stream_spatial(spatial_x)
        f_fused   = torch.cat((f_motion, f_spatial), dim=1)
        f_fused   = self.cross_attention(f_fused)
        embed     = self.encoder(f_fused)
        logits    = self.head(embed)

        if return_flow and return_feat:
            return logits, embed, self.flow_decoder(embed)
        elif return_flow:
            return logits, self.flow_decoder(embed)
        elif return_feat:
            return logits, embed
        return logits


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


# ── Onset detection ───────────────────────────────────────────────────────────

def detect_onset_frame(frames: list, apex_idx: int, window_size: int = 15) -> int:
    """
    Slide a window over frames[0:apex_idx] and find the most temporally stable
    segment (lowest inter-frame variance). The last frame of that window = onset.
    window_size default 15 (~0.5 s at 30 fps). Tune via process_video().
    """
    search_end = max(0, apex_idx - 5)
    if search_end < window_size:
        return 0
    best_onset, min_var = 0, float('inf')
    for start in range(0, search_end - window_size + 1):
        window = frames[start:start + window_size]
        diffs  = [np.mean(np.abs(window[i + 1].astype(float) - window[i].astype(float)))
                  for i in range(len(window) - 1)]
        v = np.var(diffs)
        if v < min_var:
            min_var    = v
            best_onset = start + window_size - 1
    return best_onset


# ── MagNet amplification ──────────────────────────────────────────────────────

def apply_magnet(onset_crop: np.ndarray, apex_crop: np.ndarray,
                 magnet_model, device: torch.device,
                 amp_factor: int = 10) -> np.ndarray:
    """Returns MagNet-amplified apex as uint8 RGB numpy array (H, W, 3)."""
    def to_t(img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

    amp = torch.tensor([amp_factor], device=device, dtype=torch.float32).view(1, 1, 1, 1)
    with torch.no_grad():
        out = magnet_model(to_t(onset_crop), to_t(apex_crop), 0, 0, amp, mode='evaluate')[0]
    return torch.clamp((out.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()


# ── RAFT optical flow ─────────────────────────────────────────────────────────

def _prep_raft(img_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) uint8 RGB → (1, 3, H, W) float in [-1, 1]."""
    t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return 2.0 * (t / 255.0) - 1.0


def compute_raft_flow(onset_crop: np.ndarray, apex_amplified: np.ndarray,
                      raft_model, device: torch.device) -> np.ndarray:
    """Returns raw optical flow as (2, H, W) float32 numpy array."""
    with torch.no_grad():
        preds = raft_model(_prep_raft(onset_crop, device), _prep_raft(apex_amplified, device))
    return preds[-1][0].cpu().numpy()  # (2, H, W)


def flow_to_vis(flow: np.ndarray) -> np.ndarray:
    """(2, H, W) flow → (H, W, 3) uint8 RGB for display."""
    u, v = flow[0], flow[1]
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# ── Encode image as base64 JPEG ───────────────────────────────────────────────

def encode_b64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None, num_classes: int = 3,
               device: Optional[torch.device] = None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # 1. DualCnnMER (spatial_chans=3 for RGB apex stream)
    model = DualCnnMER(num_classes=num_classes, spatial_chans=3)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[pipeline] Loaded DualCnnMER checkpoint: {checkpoint_path}")
    else:
        print("[pipeline] No checkpoint — using untrained weights (demo mode).")
    model.to(device).eval()

    # 2. RAFT Large
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    print("[pipeline] RAFT Large loaded.")

    # 3. MagNet
    motion_amp_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion_amp")
    sys.path.insert(0, motion_amp_dir)
    from magnet import MagNet
    from callbacks import gen_state_dict
    magnet_weights = os.path.join(motion_amp_dir, "magnet_epoch12_loss7.28e-02.pth")
    magnet_model   = MagNet().to(device).eval()
    magnet_model.load_state_dict(gen_state_dict(magnet_weights))
    print("[pipeline] MagNet loaded.")

    return model, raft_model, magnet_model, device


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: DualCnnMER, flow: np.ndarray,
                  apex_rgb: np.ndarray, device: torch.device) -> dict:
    """
    flow:     (2, H, W) float32 — RAFT output
    apex_rgb: (H, W, 3) uint8  — raw (non-amplified) apex crop

    Preprocessing matches CasmeDualCNNDataset.__getitem__() exactly.
    """
    IMAGE_SIZE = 112

    # Flow: resize → normalise magnitude → shift to [-1, 1]
    flow_resized = cv2.resize(
        flow.transpose(1, 2, 0), (IMAGE_SIZE, IMAGE_SIZE)
    ).transpose(2, 0, 1)
    u, v    = flow_resized[0], flow_resized[1]
    max_mag = np.max(np.sqrt(u ** 2 + v ** 2))
    if max_mag > 1e-5:
        u_norm = (u / max_mag + 1.0) / 2.0
        v_norm = (v / max_mag + 1.0) / 2.0
    else:
        u_norm = np.full_like(u, 0.5)
        v_norm = np.full_like(v, 0.5)
    tensor_flow = torch.from_numpy(np.stack([u_norm, v_norm])).float()
    tensor_flow = (tensor_flow - 0.5) / 0.5  # → [-1, 1]

    # Spatial (apex RGB): resize → [0, 1] → [-1, 1]
    apex_resized    = cv2.resize(apex_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor_spatial  = torch.from_numpy(apex_resized).permute(2, 0, 1).float() / 255.0
    tensor_spatial  = (tensor_spatial - 0.5) / 0.5

    # AdaptiveAvgPool2d on MPS requires input divisible by output size (unimplemented).
    # DualCnnMER is tiny so CPU fallback has negligible cost.
    if device.type == "mps":
        model.to("cpu")
        logits = model(tensor_flow.unsqueeze(0), tensor_spatial.unsqueeze(0))
        model.to(device)
    else:
        logits = model(tensor_flow.unsqueeze(0).to(device), tensor_spatial.unsqueeze(0).to(device))
    probs   = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_id = int(np.argmax(probs))

    return {
        "emotion":     ID_TO_LABEL[pred_id],
        "group":       ID_TO_LABEL[pred_id],
        "confidence":  round(probs[pred_id], 4),
        "group_probs": {ID_TO_LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
        "all_probs":   {ID_TO_LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ── Main pipeline ────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "Loading frames",
    "Apex frame detection",
    "Onset frame detection",
    "Face detection & crop",
    "MagNet amplification",
    "RAFT optical flow",
    "Running inference",
]


def process_video_stream(video_path: str, model: DualCnnMER, raft_model,
                         magnet_model, device: torch.device,
                         onset_window_size: int = 15):
    """
    Generator — yields progress dicts at each step, then a final result dict.

    Progress event:  {"step": int, "total": int, "label": str}
    Result event:    {"done": True, **result_fields}
    Error event:     {"error": str}
    """
    TOTAL = len(PIPELINE_STEPS)
    print("\n── Pipeline started ─────────────────────────────", flush=True)

    def progress(step: int, extra: str = ""):
        label = PIPELINE_STEPS[step - 1]
        print(f"  [{step}/{TOTAL}] {label}{(' — ' + extra) if extra else ''}…", flush=True)
        return {"step": step, "total": TOTAL, "label": label}

    try:
        # 1. Load frames
        yield progress(1)
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
            yield {"error": "Could not read any frames from the video."}
            return
        print(f"      → {len(frames)} frames at {fps:.1f} fps", flush=True)

        # 2. Apex detection
        yield progress(2)
        spotter  = ApexFrameSpotter(t_window=min(61, len(frames)))
        apex_idx, apex_score, _ = spotter.find_apex_in_short_video(frames)
        print(f"      → apex frame: {apex_idx}  (score={apex_score:.2f})", flush=True)

        # 3. Onset detection
        yield progress(3, f"window={onset_window_size}")
        onset_idx = detect_onset_frame(frames, apex_idx, window_size=onset_window_size)
        print(f"      → onset frame: {onset_idx}", flush=True)

        onset_raw = frames[onset_idx]
        apex_raw  = frames[apex_idx]

        # 4. Face crop
        yield progress(4)
        onset_crop = align_and_crop_face(onset_raw, target_size=224)
        apex_crop  = align_and_crop_face(apex_raw,  target_size=224)

        # 5. MagNet amplification
        yield progress(5)
        apex_amplified = apply_magnet(onset_crop, apex_crop, magnet_model, device)

        # 6. RAFT flow
        yield progress(6)
        flow     = compute_raft_flow(onset_crop, apex_amplified, raft_model, device)
        flow_vis = flow_to_vis(flow)
        print(f"      → flow shape: {flow.shape}", flush=True)

        # 7. Inference
        yield progress(7)
        result = run_inference(model, flow, apex_crop, device)
        print(f"      → {result['emotion']}  conf={result['confidence']:.3f}", flush=True)

        result.update({
            "apex_frame_index":  int(apex_idx),
            "onset_frame_index": int(onset_idx),
            "total_frames":      len(frames),
            "fps":               round(fps, 2),
            "images": {
                "onset":        encode_b64(onset_raw),
                "apex":         encode_b64(apex_raw),
                "onset_face":   encode_b64(onset_crop),
                "apex_face":    encode_b64(apex_crop),
                "magnified":    encode_b64(apex_amplified),
                "optical_flow": encode_b64(flow_vis),
            },
        })

        print("── Pipeline complete ────────────────────────────\n", flush=True)
        yield {"done": True, **result}

    except Exception as exc:
        import traceback
        traceback.print_exc()
        yield {"error": str(exc)}


def process_video(video_path: str, model: DualCnnMER, raft_model,
                  magnet_model, device: torch.device,
                  onset_window_size: int = 15) -> dict:
    """Blocking wrapper around process_video_stream for simple request/response use."""
    for event in process_video_stream(video_path, model, raft_model, magnet_model,
                                      device, onset_window_size=onset_window_size):
        if event.get("error"):
            raise RuntimeError(event["error"])
        if event.get("done"):
            event.pop("done")
            return event
