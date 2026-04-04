"""
pipeline.py — End-to-end inference pipeline (updated)
Video → Apex Detection → Onset Detection → Face Crop (MTCNN) → TVL1 Flow → DualStreamModel → Emotion
"""

import os
import sys
import base64
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

# Add parent dir so we can import apex_frame_detection
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from apex_frame_detection import ApexFrameSpotter
from exp_02_emotion_transformer.models import DualStreamModel
from exp_02_emotion_transformer.utils import flow_to_rgb_heatmap

# Try to import MTCNN for face detection
try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False
    print("[pipeline] WARNING: facenet_pytorch not available; face alignment will be skipped.")

# ── Constants ────────────────────────────────────────────────────────────────

ID_TO_LABEL = {0: "positive", 1: "negative", 2: "surprise"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── MTCNN Face Detection ─────────────────────────────────────────────────────

_mtcnn = None

def _get_mtcnn_detector(device):
    """Lazy-load MTCNN."""
    global _mtcnn
    if _mtcnn is None and _MTCNN_AVAILABLE:
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn

def get_landmarks(img_rgb: np.ndarray, device) -> Optional[np.ndarray]:
    """Detect face landmarks using MTCNN."""
    mtcnn = _get_mtcnn_detector(device)
    if mtcnn is None:
        return None

    try:
        boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
        if landmarks is None or len(landmarks) == 0:
            return None

        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = np.argmax(areas)
        else:
            best_idx = 0

        pts = landmarks[best_idx]
        return np.array([pts[0], pts[1], pts[2]], dtype=np.float32)
    except Exception:
        return None

def align_image(src_img: np.ndarray, target_img: np.ndarray, device) -> np.ndarray:
    """Align src_img to target_img using MTCNN landmarks."""
    src_pts = get_landmarks(src_img, device)
    dst_pts = get_landmarks(target_img, device)

    if src_pts is None or dst_pts is None:
        return src_img

    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        return src_img

    h, w = target_img.shape[:2]
    return cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def crop_face_mtcnn(img_rgb: np.ndarray, target_size: int = 256, device=None) -> np.ndarray:
    """
    Crop and resize face to target_size using MTCNN + center crop approach.
    Matches generate_tvl1_flows_amp5.py logic.
    """
    h, w = img_rgb.shape[:2]
    s = min(h, w)
    cy, cx = h // 2, w // 2
    crop_s = int(s * 0.9)  # Center 90% crop
    crop = img_rgb[cy - crop_s // 2 : cy + crop_s // 2, cx - crop_s // 2 : cx + crop_s // 2]
    return cv2.resize(crop, (target_size, target_size))

# ── TVL1 Optical Flow ────────────────────────────────────────────────────────

def get_tvl1_flow(on_img: np.ndarray, ap_img: np.ndarray, amp_factor: float = 5.0) -> np.ndarray:
    """
    Compute TVL1 optical flow from onset to apex, with amplification.
    Returns (2, H, W) float32 array.
    """
    on_gray = cv2.cvtColor(on_img, cv2.COLOR_RGB2GRAY)
    ap_gray = cv2.cvtColor(ap_img, cv2.COLOR_RGB2GRAY)

    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(medianFiltering=3)
        flow = tvl1.calc(on_gray, ap_gray, None)
    except AttributeError:
        # Fallback to Farneback
        flow = cv2.calcOpticalFlowFarneback(on_gray, ap_gray, None, 0.5, 3, 9, 3, 5, 1.2, 0)

    # Amplify
    flow *= amp_factor

    # Transpose to (2, H, W)
    return np.transpose(flow, (2, 0, 1)).astype(np.float32)

# ── Onset detection ───────────────────────────────────────────────────────────

def detect_onset_frame(frames: list, apex_idx: int, window_size: int = 15) -> int:
    """
    Slide a window over frames[0:apex_idx] and find the most temporally stable
    segment (lowest inter-frame variance). The last frame of that window = onset.
    """
    search_end = max(0, apex_idx - 5)
    if search_end < window_size:
        return 0

    best_onset, min_var = 0, float('inf')
    for start in range(0, search_end - window_size + 1):
        window = frames[start : start + window_size]
        diffs = [
            np.mean(np.abs(window[i + 1].astype(float) - window[i].astype(float)))
            for i in range(len(window) - 1)
        ]
        v = np.var(diffs)
        if v < min_var:
            min_var = v
            best_onset = start + window_size - 1

    return best_onset

# ── Encode image as base64 JPEG ───────────────────────────────────────────────

def encode_b64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Optional[str] = None, num_classes: int = 3,
               device: Optional[torch.device] = None):
    """Load DualStreamModel."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = DualStreamModel(num_classes=num_classes)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[pipeline] Loaded DualStreamModel checkpoint: {checkpoint_path}")
    else:
        print("[pipeline] No checkpoint — using untrained weights (demo mode).")

    model.to(device).eval()
    return model, device

# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: DualStreamModel, flow: np.ndarray,
                  apex_rgb: np.ndarray, device: torch.device) -> dict:
    """
    Run inference on flow (TVL1, RGB heatmap) and spatial (apex RGB).

    Preprocessing matches exp_02_emotion_transformer/dataset.py:
    - Flow: (2,H,W) float32 → flow_to_rgb_heatmap → (3,H,W) [0,1] → Normalize(ImageNet)
    - Spatial: (H,W,3) uint8 → resize to 256 → (3,H,W) [0,1] → Normalize(ImageNet)
    """
    IMAGE_SIZE = 256

    # Resize flow if needed
    if flow.shape[1] != IMAGE_SIZE or flow.shape[2] != IMAGE_SIZE:
        flow = cv2.resize(flow.transpose(1, 2, 0), (IMAGE_SIZE, IMAGE_SIZE)).transpose(2, 0, 1)

    # Flow: convert to RGB heatmap and normalize
    rgb_heatmap = flow_to_rgb_heatmap(flow)  # (3, H, W) float32 [0, 1]
    tensor_flow = torch.from_numpy(rgb_heatmap).float()
    tensor_flow = F.normalize(
        tensor_flow - torch.tensor(IMAGENET_MEAN).view(3, 1, 1),
        p=2,
        dim=0,
    )
    # Actually, use the standard normalization
    tensor_flow = torch.from_numpy(rgb_heatmap).float()
    tensor_flow = tensor_flow - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    tensor_flow = tensor_flow / torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # Spatial: resize and normalize
    apex_resized = cv2.resize(apex_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    tensor_spatial = torch.from_numpy(apex_resized).permute(2, 0, 1).float() / 255.0
    tensor_spatial = tensor_spatial - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    tensor_spatial = tensor_spatial / torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # Inference
    if device.type == "mps":
        model.to("cpu")
        logits = model(tensor_flow.unsqueeze(0), tensor_spatial.unsqueeze(0))
        model.to(device)
    else:
        logits = model(
            tensor_flow.unsqueeze(0).to(device),
            tensor_spatial.unsqueeze(0).to(device),
        )

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_id = int(np.argmax(probs))

    return {
        "emotion": ID_TO_LABEL[pred_id],
        "group": ID_TO_LABEL[pred_id],
        "confidence": round(probs[pred_id], 4),
        "group_probs": {ID_TO_LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
        "all_probs": {ID_TO_LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
    }

# ── Main pipeline ────────────────────────────────────────────────────────────

PIPELINE_STEPS = [
    "Loading frames",
    "Apex frame detection",
    "Onset frame detection",
    "Face detection & crop",
    "TVL1 optical flow",
    "Running inference",
]

def process_video_stream(video_path: str, model: DualStreamModel, device: torch.device,
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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        spotter = ApexFrameSpotter(t_window=min(61, len(frames)))
        apex_idx, apex_score, _ = spotter.find_apex_in_short_video(frames)
        print(f"      → apex frame: {apex_idx}  (score={apex_score:.2f})", flush=True)

        # 3. Onset detection
        yield progress(3, f"window={onset_window_size}")
        onset_idx = detect_onset_frame(frames, apex_idx, window_size=onset_window_size)
        print(f"      → onset frame: {onset_idx}", flush=True)

        onset_raw = frames[onset_idx]
        apex_raw = frames[apex_idx]

        # 4. Face crop (MTCNN-based)
        yield progress(4)
        onset_crop = crop_face_mtcnn(onset_raw, target_size=256, device=device)
        apex_crop = crop_face_mtcnn(apex_raw, target_size=256, device=device)

        # Optional: align apex to onset using landmarks
        if _MTCNN_AVAILABLE:
            apex_crop = align_image(apex_crop, onset_crop, device)

        print(f"      → crops: onset {onset_crop.shape}, apex {apex_crop.shape}", flush=True)

        # 5. TVL1 optical flow
        yield progress(5)
        flow = get_tvl1_flow(onset_crop, apex_crop, amp_factor=5.0)
        flow_vis = _flow_to_vis(flow)  # For visualization
        print(f"      → flow shape: {flow.shape}", flush=True)

        # 6. Inference
        yield progress(6)
        result = run_inference(model, flow, apex_crop, device)
        print(f"      → {result['emotion']}  conf={result['confidence']:.3f}", flush=True)

        result.update({
            "apex_frame_index": int(apex_idx),
            "onset_frame_index": int(onset_idx),
            "total_frames": len(frames),
            "fps": round(fps, 2),
            "images": {
                "onset": encode_b64(onset_raw),
                "apex": encode_b64(apex_raw),
                "onset_face": encode_b64(onset_crop),
                "apex_face": encode_b64(apex_crop),
                "optical_flow": encode_b64(flow_vis),
            },
        })

        print("── Pipeline complete ────────────────────────────\n", flush=True)
        yield {"done": True, **result}

    except Exception as exc:
        import traceback
        traceback.print_exc()
        yield {"error": str(exc)}

def _flow_to_vis(flow: np.ndarray) -> np.ndarray:
    """Convert (2, H, W) flow to (H, W, 3) uint8 RGB for visualization."""
    u, v = flow[0], flow[1]
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def process_video(video_path: str, model: DualStreamModel, device: torch.device,
                  onset_window_size: int = 15) -> dict:
    """Blocking wrapper around process_video_stream for simple request/response use."""
    for event in process_video_stream(video_path, model, device, onset_window_size=onset_window_size):
        if event.get("error"):
            raise RuntimeError(event["error"])
        if event.get("done"):
            event.pop("done")
            return event
