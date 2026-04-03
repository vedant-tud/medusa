import os
import shutil
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from tqdm import tqdm

# Ensure motion_amp is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'motion_amp'))
from magnet import MagNet
from callbacks import gen_state_dict

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

device = get_device()

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def align_and_crop_face(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        m  = int(0.15 * min(w, h))
        x1, y1 = max(0, x - m), max(0, y - m)
        x2, y2 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
        crop = img[y1:y2, x1:x2]
    else:
        h, w = img.shape[:2]
        s = min(h, w)
        crop = img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return cv2.resize(crop, (target_size, target_size))

def prepare_for_raft(img_np):
    """Numpy RGB (H, W, 3) -> Torch (1, 3, H, W) scaled to [-1, 1]."""
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return 2.0 * (img_t / 255.0) - 1.0

def rotate_image(image, angle):
    """Rotates image by given angle with reflection padding to avoid black borders."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

# ─── THE SMART LOGIC ─────────────────────────────────────────────────────────

def get_adaptive_amp_factor(raft_model, onset_crop, apex_crop, base_target=2.5):
    """Computes required boost based on raw motion magnitude."""
    with torch.no_grad():
        preds = raft_model(prepare_for_raft(onset_crop), prepare_for_raft(apex_crop))
        flow_raw = preds[-1][0].cpu().numpy()
        avg_mag = np.mean(np.sqrt(flow_raw[0]**2 + flow_raw[1]**2))
        # Target a specific movement intensity; clamp between 2x and 20x
        return np.clip(base_target / (avg_mag + 1e-6), 2.0, 20.0)

def compute_3channel_flow(raft_model, img1, img2):
    """Returns flow as (3, H, W) where channels are [U, V, Magnitude]."""
    with torch.no_grad():
        preds = raft_model(prepare_for_raft(img1), prepare_for_raft(img2))
        f = preds[-1][0].cpu().numpy() # (2, H, W)
        mag = np.sqrt(f[0]**2 + f[1]**2)[np.newaxis, ...] # (1, H, W)
        return np.concatenate([f, mag], axis=0)

# ─── MAIN PROCESSING PIPELINE ────────────────────────────────────────────────

def process_video_clip(raft_model, magnet_model, clip_path, out_clip_path, base_target=2.5):
    onset_path = os.path.join(clip_path, "onset.jpg")
    apex_path  = os.path.join(clip_path, "apex.jpg")

    if not (os.path.isfile(onset_path) and os.path.isfile(apex_path)):
        return False

    onset_img = cv2.imread(onset_path)
    apex_img  = cv2.imread(apex_path)
    if onset_img is None or apex_img is None: return False

    onset_rgb = cv2.cvtColor(onset_img, cv2.COLOR_BGR2RGB)
    apex_rgb  = cv2.cvtColor(apex_img,  cv2.COLOR_BGR2RGB)

    # 1. Align and Crop
    onset_crop = align_and_crop_face(onset_rgb)
    apex_crop  = align_and_crop_face(apex_rgb)

    # 2. Calculate Adaptive Amp Factor
    amp_factor = get_adaptive_amp_factor(raft_model, onset_crop, apex_crop, base_target)

    # 3. Setup Augmentation (Random Rotation +/- 5 degrees)
    angle = np.random.uniform(-5, 5)
    onset_crop_aug = rotate_image(onset_crop, angle)
    apex_crop_aug  = rotate_image(apex_crop,  angle)

    # 4. MagNet Amplification
    def to_mag_tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 127.5 - 1.0

    amp_tensor = torch.tensor([amp_factor], device=device, dtype=torch.float32).view(1, 1, 1, 1)
    
    with torch.no_grad():
        # Magnify Standard
        amp_apex = magnet_model(to_mag_tensor(onset_crop), to_mag_tensor(apex_crop), 0, 0, amp_tensor, mode='evaluate')[0]
        amp_apex_np = torch.clamp((amp_apex.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()

        # Magnify Augmented
        amp_apex_aug = magnet_model(to_mag_tensor(onset_crop_aug), to_mag_tensor(apex_crop_aug), 0, 0, amp_tensor, mode='evaluate')[0]
        amp_apex_aug_np = torch.clamp((amp_apex_aug.squeeze(0).permute(1, 2, 0) + 1.0) * 127.5, 0, 255).byte().cpu().numpy()

    # 5. Compute 3-Channel Flow (U, V, Mag)
    flow_3ch     = compute_3channel_flow(raft_model, onset_crop, amp_apex_np)
    flow_3ch_aug = compute_3channel_flow(raft_model, onset_crop_aug, amp_apex_aug_np)

    # 6. Save results
    np.save(os.path.join(out_clip_path, "flow.npy"),      flow_3ch)
    np.save(os.path.join(out_clip_path, "flow_aug.npy"),  flow_3ch_aug) # Replaces flow_flip.npy
    
    cv2.imwrite(os.path.join(out_clip_path, "onset_gray.jpg"), cv2.cvtColor(onset_crop, cv2.COLOR_RGB2GRAY))
    
    # Metadata update
    info_src = os.path.join(clip_path, "info.txt")
    if os.path.isfile(info_src):
        with open(info_src, 'r') as f: content = f.read()
        with open(os.path.join(out_clip_path, "info.txt"), 'w') as f:
            f.write(content + f"\nadaptive_amp: {amp_factor:.2f}\naug_rotation: {angle:.2f}")
    
    return True

# ─── MAIN EXECUTION ──────────────────────────────────────────────────────────

def main():
    # Model Loading
    raft_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).to(device).eval()
    
    magnet_model = MagNet().to(device)
    magnet_path = os.path.join(os.path.dirname(__file__), 'motion_amp', 'magnet_epoch12_loss7.28e-02.pth')
    magnet_model.load_state_dict(gen_state_dict(magnet_path))
    magnet_model.eval()
    
    data_root = "./flow_test"
    output_root = "./casme_raft_processed_v2" # New folder for improved version
    
    # Discovery
    clip_paths = []
    for subj in sorted(os.listdir(data_root)):
        subj_p = os.path.join(data_root, subj)
        if not os.path.isdir(subj_p): continue
        for clip in sorted(os.listdir(subj_p)):
            clip_p = os.path.join(subj_p, clip)
            if os.path.isdir(clip_p):
                out_p = os.path.join(output_root, subj, clip)
                os.makedirs(out_p, exist_ok=True)
                clip_paths.append((clip_p, out_p))

    # Processing
    success = 0
    for in_p, out_p in tqdm(clip_paths, desc="Processing MER Pipeline"):
        if process_video_clip(raft_model, magnet_model, in_p, out_p):
            success += 1

    print(f"\nFinished. Processed {success}/{len(clip_paths)} clips into {output_root}")

if __name__ == "__main__":
    main()