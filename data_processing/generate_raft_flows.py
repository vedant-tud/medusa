import os
import shutil

# Set cache directories before importing torch/torchvision
os.environ["TORCH_HOME"] = "/scratch/smiyyapuram/medusa/.cache/torch"
os.environ["HF_HOME"] = "/scratch/smiyyapuram/medusa/.cache/huggingface"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm

import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
_motion_amp_paths = [
    os.path.join(_current_dir, "motion_amp"),
    os.path.join(_project_root, "motion_amp"),
]
for _path in _motion_amp_paths:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)
from magnet import MagNet
from callbacks import gen_state_dict
from data import unit_postprocessing, unit_preprocessing

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_face_bbox(img: np.ndarray):
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        return faces[0]
    return None

def crop_face(img: np.ndarray, bbox, target_size: int = 224) -> np.ndarray:
    if bbox is not None:
        x, y, w, h = bbox
        m  = int(0.15 * min(w, h))
        x1 = max(0, x - m);   y1 = max(0, y - m)
        x2 = min(img.shape[1], x + w + m);  y2 = min(img.shape[0], y + h + m)
        crop = img[y1:y2, x1:x2]
    else:
        h, w = img.shape[:2];  s = min(h, w)
        crop = img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return cv2.resize(crop, (target_size, target_size))

def prepare_for_raft(img_np):
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_t = 2.0 * (img_t / 255.0) - 1.0
    return img_t

def process_video_clip(model, magnet_model, clip_path, out_clip_path, amp_factor=10):
    onset_path = os.path.join(clip_path, "onset.jpg")
    apex_path  = os.path.join(clip_path, "apex.jpg")

    if not (os.path.isfile(onset_path) and os.path.isfile(apex_path)):
        return False

    onset_img = cv2.imread(onset_path)
    apex_img  = cv2.imread(apex_path)
    if onset_img is None or apex_img is None:
        return False

    # Convert to RGB early for cropping consistency with our train.py pipeline
    onset_rgb = cv2.cvtColor(onset_img, cv2.COLOR_BGR2RGB)
    apex_rgb  = cv2.cvtColor(apex_img,  cv2.COLOR_BGR2RGB)

    # Face align & crop with SAME bounding box for both frames
    bbox = get_face_bbox(onset_rgb)
    onset_crop = crop_face(onset_rgb, bbox)
    apex_crop  = crop_face(apex_rgb, bbox)

    # Convert to format required for MagNet
    onset_mag_input = torch.from_numpy(onset_crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0 # (1, 3, H, W), -1 to 1
    apex_mag_input  = torch.from_numpy(apex_crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
    
    # Flipped versions matching standard horizontal augmentation probability mapping
    onset_crop_flip = cv2.flip(onset_crop, 1)
    apex_crop_flip  = cv2.flip(apex_crop,  1)

    onset_mag_flip_input = torch.from_numpy(onset_crop_flip).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0
    apex_mag_flip_input  = torch.from_numpy(apex_crop_flip).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

    with torch.no_grad():
        # APPLY MAGNET
        amp_tensor = torch.tensor([amp_factor], device=device, dtype=torch.float32)
        while len(amp_tensor.shape) < len(onset_mag_input.shape):
            amp_tensor = amp_tensor.unsqueeze(-1)
            
        # MagNet expects mode='evaluate' and returns a tuple, first is amplified frame
        amplified_apex = magnet_model(onset_mag_input, apex_mag_input, 0, 0, amp_tensor, mode='evaluate')[0]
        amplified_apex_flip = magnet_model(onset_mag_flip_input, apex_mag_flip_input, 0, 0, amp_tensor, mode='evaluate')[0]
        
        # Convert back to uint8 RGB (0-255) to be prepared for RAFT
        amplified_apex_np = torch.clamp((amplified_apex.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()
        amplified_apex_flip_np = torch.clamp((amplified_apex_flip.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()

        # NORMAL FLOW (onset -> amplified apex)
        flow_predictions = model(prepare_for_raft(onset_crop), prepare_for_raft(amplified_apex_np))
        # raft outputs a list of refined flow estimates, last is highest accuracy
        flow = flow_predictions[-1][0].cpu().numpy() # Shape (2, H, W)

        # FLIPPED FLOW
        flow_predictions_flip = model(prepare_for_raft(onset_crop_flip), prepare_for_raft(amplified_apex_flip_np))
        flow_flip = flow_predictions_flip[-1][0].cpu().numpy()

    # Create matching Grayscales
    onset_gray      = cv2.cvtColor(onset_crop,      cv2.COLOR_RGB2GRAY)
    onset_gray_flip = cv2.cvtColor(onset_crop_flip, cv2.COLOR_RGB2GRAY)

    # Save to disk
    np.save(os.path.join(out_clip_path, "flow.npy"),      flow)
    np.save(os.path.join(out_clip_path, "flow_flip.npy"), flow_flip)
    
    cv2.imwrite(os.path.join(out_clip_path, "onset_gray.jpg"),      onset_gray)
    cv2.imwrite(os.path.join(out_clip_path, "onset_gray_flip.jpg"), onset_gray_flip)
    
    # Copy info.txt for dataloader
    info_src = os.path.join(clip_path, "info.txt")
    if os.path.isfile(info_src):
        shutil.copy(info_src, os.path.join(out_clip_path, "info.txt"))
    
    return True

def main():
    print(f"Using device: {device}")
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights, progress=True).to(device)
    model.eval()
    amp_factor = 10
    # Load MagNet model
    magnet_model = MagNet().to(device)
    magnet_weights_path = os.path.join(os.path.dirname(__file__), 'motion_amp', 'magnet_epoch12_loss7.28e-02.pth')
    state_dict = gen_state_dict(magnet_weights_path)
    magnet_model.load_state_dict(state_dict)
    magnet_model.eval()
    print("Loaded MagNet weights successfully")

    data_root = "./casme_processed"
    output_root = "./casme_raft_processed"+str(amp_factor)
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        return

    clip_paths = []
    for subj in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj)
        if not os.path.isdir(subj_path): continue
        for clip in sorted(os.listdir(subj_path)):
            clip_path = os.path.join(subj_path, clip)
            if os.path.isdir(clip_path):
                out_clip_path = os.path.join(output_root, subj, clip)
                os.makedirs(out_clip_path, exist_ok=True)
                clip_paths.append((clip_path, out_clip_path))

    failed_clips = []
    success_count = 0
    for in_path, out_path in tqdm(clip_paths, desc="Generating RAFT pairs"):
        if process_video_clip(model, magnet_model, in_path, out_path, amp_factor=amp_factor):
            success_count += 1
        else:
            failed_clips.append(in_path)

    print(f"\n✅ Successfully processed {success_count}/{len(clip_paths)} clips.")
    
    if failed_clips:
        failed_log_path = "raft_failed_clips.txt"
        print(f"❌ Failed to process {len(failed_clips)} clips. Check {failed_log_path} for details.")
        with open(failed_log_path, "w") as f:
            for c in failed_clips:
                f.write(f"{c}\n")

if __name__ == "__main__":
    main()
