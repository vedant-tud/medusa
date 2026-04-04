import os
import cv2
import numpy as np
import torch
import shutil
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Setup MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

def get_landmarks(img_rgb):
    boxes, probs, landmarks = mtcnn.detect(img_rgb, landmarks=True)
    if landmarks is None or len(landmarks) == 0: return None
    if len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = np.argmax(areas)
    else: best_idx = 0
    pts = landmarks[best_idx]
    return np.array([pts[0], pts[1], pts[2]], dtype=np.float32)

def align_image(src_img, target_img):
    src_pts = get_landmarks(src_img)
    dst_pts = get_landmarks(target_img)
    if src_pts is None or dst_pts is None: return src_img
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None: return src_img
    h, w = target_img.shape[:2]
    return cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def get_tvl1_flow(on_img, ap_img):
    on_gray = cv2.cvtColor(on_img, cv2.COLOR_RGB2GRAY)
    ap_gray = cv2.cvtColor(ap_img, cv2.COLOR_RGB2GRAY)
    try:
        # medianFiltering=3 for slightly increased block size proxy
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(medianFiltering=3)
        flow = tvl1.calc(on_gray, ap_gray, None)
    except AttributeError:
        flow = cv2.calcOpticalFlowFarneback(on_gray, ap_gray, None, 0.5, 3, 9, 3, 5, 1.2, 0)
    return flow

def process_clip(clip_path, out_clip_path, amp_factor=5.0):
    onset_path = os.path.join(clip_path, "onset.jpg")
    apex_path = os.path.join(clip_path, "apex.jpg")
    
    if not (os.path.isfile(onset_path) and os.path.isfile(apex_path)):
        return False
        
    onset_img = cv2.imread(onset_path)
    apex_img = cv2.imread(apex_path)
    if onset_img is None or apex_img is None: return False
    
    onset_rgb = cv2.cvtColor(onset_img, cv2.COLOR_BGR2RGB)
    apex_rgb = cv2.cvtColor(apex_img, cv2.COLOR_BGR2RGB)
    
    # We crop the way the experiments do
    h, w = onset_rgb.shape[:2]
    s = min(h, w)
    cy, cx = h//2, w//2
    crop_s = int(s * 0.9)
    on_cr = onset_rgb[cy-crop_s//2:cy+crop_s//2, cx-crop_s//2:cx+crop_s//2]
    ap_cr = apex_rgb[cy-crop_s//2:cy+crop_s//2, cx-crop_s//2:cx+crop_s//2]
    
    on_cr = cv2.resize(on_cr, (224, 224))
    ap_cr = cv2.resize(ap_cr, (224, 224))
    
    # Standard Flow
    ap_al = align_image(ap_cr, on_cr)
    flow = get_tvl1_flow(on_cr, ap_al)
    
    # Flipped Flow
    on_cr_flip = cv2.flip(on_cr, 1)
    ap_al_flip = cv2.flip(ap_al, 1)
    flow_flip = get_tvl1_flow(on_cr_flip, ap_al_flip)
    
    # Apply Amp scaling to the flows linearly (as chosen by the user)
    flow *= amp_factor
    flow_flip *= amp_factor
    
    # dataset.py expects flow in shape (2, H, W)
    flow = np.transpose(flow, (2, 0, 1))
    flow_flip = np.transpose(flow_flip, (2, 0, 1))
    
    np.save(os.path.join(out_clip_path, "flow.npy"), flow)
    np.save(os.path.join(out_clip_path, "flow_flip.npy"), flow_flip)
    
    info_src = os.path.join(clip_path, "info.txt")
    if os.path.isfile(info_src):
        shutil.copy(info_src, os.path.join(out_clip_path, "info.txt"))
        
    return True

def main():
    data_root = "/scratch/smiyyapuram/medusa/casme_processed"
    output_root = "/scratch/smiyyapuram/medusa/casme_tvl1_processed_amp5"
    
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

    success = 0
    for in_path, out_path in tqdm(clip_paths, desc="Generating TVL1 Amp x5 Pairs"):
        if process_clip(in_path, out_path, amp_factor=5.0):
            success += 1
            
    print(f"✅ Successfully processed {success}/{len(clip_paths)} clips.")

if __name__ == "__main__":
    main()
