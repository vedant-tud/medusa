"""
Micro-Expression Pipeline: Riesz Pyramid EVM (Magnification Only) — Optimised
==============================================================================
Optimisations applied
---------------------
1. Vectorised Riesz + bandpass  – no Python loops over pixels/frames
2. Multiprocessing (ProcessPoolExecutor) – one worker per CPU core
3. Only magnify frames 0 → apex  (frames after apex saved as-is)
4. Downscale → EVM → upscale  (default 256 px wide)
5. Pyramid depth 3 instead of 4
6. float32 throughout (no float64 FFT)
7. α ramp capped at 15

Folder structure expected:
  <medusa_root>/
    data/
      anger/ happy/ disgust/ sad/ fear/ surprise/
    video_emotion_metadata_v2.csv

Outputs:
  output/
    magnified/   -> magnified .mp4 per video
    previews/    -> one magnified .mp4 per emotion category (for inspection)
    pipeline.log
"""

import cv2
import csv
import logging
import argparse
import warnings
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.signal import butter, sosfiltfilt   # sosfiltfilt = zero-phase (better than sosfilt)

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: Path):
    log_path = output_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

EMOTION_FOLDERS = ["anger", "happy", "disgust", "sad", "fear", "surprise"]


def load_metadata(csv_path: Path, dataset_root: Path, logger):
    records = []
    missing = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion  = row["Emotion"].strip().lower()
            filename = row["Filename"].strip()
            apex     = int(row["Apex Frame"])
            subject  = row["Subject"].strip()

            video_path = dataset_root / emotion / filename
            if not video_path.exists():
                found = False
                for folder in EMOTION_FOLDERS:
                    candidate = dataset_root / folder / filename
                    if candidate.exists():
                        video_path = candidate
                        emotion    = folder
                        found      = True
                        break
                if not found:
                    logger.warning(f"NOT FOUND – skipping: {filename}")
                    missing += 1
                    continue

            records.append({
                "subject":    subject,
                "emotion":    emotion,
                "filename":   filename,
                "apex_frame": apex,
                "video_path": video_path,
            })

    logger.info(f"Loaded {len(records)} videos  |  skipped {missing} (not found)")
    return records


def read_video_frames(video_path: Path):
    """
    Read only frames 0 → apex_frame (inclusive).
    Returns (all_frames, fps)  where all_frames is T×H×W×3 float32 [0,1].
    We read everything here; slicing to apex happens in the EVM call.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        return None, fps

    # Stack to array once – avoids repeated list appends downstream
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0   # T×H×W×3
    return arr, fps


def clamp_frame_idx(idx: int, total: int) -> int:
    return max(0, min(idx, total - 1))


# ═════════════════════════════════════════════════════════════════════════════
# 2. RIESZ PYRAMID EVM  (fully vectorised)
# ═════════════════════════════════════════════════════════════════════════════

def _resize_frames(frames: np.ndarray, target_w: int) -> tuple:
    """
    Downscale T×H×W×3 to target_w wide (preserve aspect ratio).
    Returns (resized_frames, orig_h, orig_w).
    """
    T, H, W, C = frames.shape
    if W <= target_w:
        return frames, H, W
    scale    = target_w / W
    new_w    = target_w
    new_h    = max(1, int(H * scale))
    resized  = np.stack(
        [cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
         for t in range(T)],
        axis=0
    )
    return resized, H, W


def _to_gray_batch(frames: np.ndarray) -> np.ndarray:
    """T×H×W×3 float32 → T×H×W float32 luma (BT.601 coefficients)."""
    return (0.299 * frames[..., 0] +
            0.587 * frames[..., 1] +
            0.114 * frames[..., 2]).astype(np.float32)


def _build_laplacian_pyramid_batch(gray: np.ndarray, levels: int):
    """
    Build Laplacian pyramid for a batch of frames.
    gray : T×H×W float32
    Returns list of length `levels`, each element T×h_l×w_l float32.
    """
    gauss = [gray]
    for _ in range(levels - 1):
        # pyrDown each frame – vectorised over T with list comp then stack
        T        = gauss[-1].shape[0]
        downsampled = np.stack(
            [cv2.pyrDown(gauss[-1][t]) for t in range(T)], axis=0
        )
        gauss.append(downsampled)

    lap = []
    for i in range(levels - 1):
        T    = gauss[i].shape[0]
        size = (gauss[i].shape[2], gauss[i].shape[1])   # (W, H)
        up   = np.stack(
            [cv2.pyrUp(gauss[i + 1][t], dstsize=size) for t in range(T)],
            axis=0
        )
        lap.append((gauss[i] - up).astype(np.float32))
    lap.append(gauss[-1].astype(np.float32))
    return lap


def _reconstruct_laplacian_batch(pyramid: list) -> np.ndarray:
    """Reconstruct T×H×W from batch Laplacian pyramid."""
    img = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        T    = img.shape[0]
        size = (level.shape[2], level.shape[1])
        img  = np.stack(
            [cv2.pyrUp(img[t], dstsize=size) for t in range(T)], axis=0
        ) + level
    return img


def _riesz_transform_batch(band: np.ndarray):
    """
    Vectorised Riesz transform over a batch T×H×W float32.
    Returns R1, R2 each T×H×W float32.
    All FFT ops in float32.
    """
    T, H, W = band.shape
    fy = np.fft.fftfreq(H).astype(np.float32)[:, np.newaxis]   # H×1
    fx = np.fft.fftfreq(W).astype(np.float32)[np.newaxis, :]   # 1×W
    r  = np.sqrt(fx**2 + fy**2)
    r[0, 0] = 1.0

    # Broadcast filter kernels: 1×H×W
    kx = (-1j * fx / r)[np.newaxis, :, :]   # 1×H×W
    ky = (-1j * fy / r)[np.newaxis, :, :]

    F      = np.fft.fft2(band).astype(np.complex64)   # T×H×W
    R1_F   = kx * F
    R2_F   = ky * F
    R1_F[:, 0, 0] = 0
    R2_F[:, 0, 0] = 0

    R1 = np.fft.ifft2(R1_F).real.astype(np.float32)
    R2 = np.fft.ifft2(R2_F).real.astype(np.float32)
    return R1, R2


def _bandpass_batch(phase: np.ndarray, low: float, high: float,
                    fps: float, order: int = 2) -> np.ndarray:
    """
    Vectorised temporal bandpass.
    phase : T×H×W float32
    Applies filter along axis 0 (time) to every H×W pixel simultaneously.
    Returns T×H×W float32.

    If the clip is too short for sosfiltfilt (needs > 3*order*2 = 12+ frames
    for order=2), we fall back to a single-pass sosfilt which has no minimum
    length requirement. For very short micro-expression clips this is acceptable
    since there is little temporal structure to preserve anyway.
    """
    nyq    = fps / 2.0
    low_n  = float(np.clip(low  / nyq, 1e-4, 0.99))
    high_n = float(np.clip(high / nyq, 1e-4, 0.99))
    if low_n >= high_n:
        return phase

    sos     = butter(order, [low_n, high_n], btype="band", output="sos")
    T, H, W = phase.shape
    flat    = phase.reshape(T, H * W)   # T × (H*W)

    # sosfiltfilt requires signal length > padlen (default 3 * filter_order * 2)
    min_len = 3 * (2 * order) + 1      # = 13 for order=2
    if T > min_len:
        filtered = sosfiltfilt(sos, flat, axis=0).astype(np.float32)
    else:
        # Fallback: forward-only filter — no minimum length constraint
        from scipy.signal import sosfilt
        filtered = sosfilt(sos, flat, axis=0).astype(np.float32)

    return filtered.reshape(T, H, W)


def build_alpha_ramp(n_frames: int, apex_idx: int, alpha_max: float = 15.0):
    """
    α ramps linearly:  1 → alpha_max at apex.
    Frames after apex are NOT magnified (alpha=1), since we only process to apex.
    """
    t      = np.arange(n_frames, dtype=np.float32)
    alphas = np.ones(n_frames, dtype=np.float32)
    if apex_idx > 0:
        alphas[:apex_idx + 1] = 1.0 + (alpha_max - 1.0) * (t[:apex_idx + 1] / apex_idx)
    return alphas


def riesz_evm(
    frames:     np.ndarray,   # T×H×W×3 float32 [0,1]
    apex_idx:   int,
    fps:        float,
    freq_low:   float = 0.2,
    freq_high:  float = 4.0,
    alpha_max:  float = 15.0,
    levels:     int   = 3,
    proc_width: int   = 256,
) -> np.ndarray:
    """
    Optimised Riesz Pyramid EVM.

    Returns magnified T×H×W×3 float32 [0,1].
    Only frames 0..apex_idx are magnified; the rest are returned unchanged.
    """
    T_full, H_orig, W_orig, _ = frames.shape

    # ── Slice to apex only ────────────────────────────────────────────────
    clip   = frames[:apex_idx + 1]          # T×H×W×3  (T ≤ apex+1)
    T      = clip.shape[0]
    alphas = build_alpha_ramp(T, apex_idx, alpha_max)   # T,

    # ── Downscale ─────────────────────────────────────────────────────────
    clip_small, _, _ = _resize_frames(clip, proc_width)   # T×h×w×3

    # ── Luma ──────────────────────────────────────────────────────────────
    gray  = _to_gray_batch(clip_small)           # T×h×w
    orig_gray = gray.copy()

    # ── Laplacian pyramid (batch) ─────────────────────────────────────────
    lap   = _build_laplacian_pyramid_batch(gray, levels)
    out_lap = [band.copy() for band in lap]      # will be modified in-place

    # ── Per-level Riesz + bandpass + amplify ─────────────────────────────
    for lv in range(levels - 1):               # skip coarsest DC level
        band   = lap[lv]                       # T×H_l×W_l
        R1, R2 = _riesz_transform_batch(band)

        # Local phase & orientation — fully vectorised
        phase  = np.arctan2(np.sqrt(R1**2 + R2**2), band)      # T×H_l×W_l
        orient = np.arctan2(R2, R1)

        # Temporal bandpass on phase
        ph_filt = _bandpass_batch(phase, freq_low, freq_high, fps)  # T×H_l×W_l

        # Scale filtered phase by per-frame alpha (broadcast T×1×1)
        delta   = ph_filt * alphas[:, np.newaxis, np.newaxis]

        # Phase-shifted reconstruction (vectorised over T, H, W)
        cos_ori = np.cos(orient)
        sin_ori = np.sin(orient)
        new_band = (
            band * np.cos(delta)
            - (R1 * cos_ori + R2 * sin_ori) * np.sin(delta)
        )
        out_lap[lv] = new_band.astype(np.float32)

    # ── Reconstruct luma ─────────────────────────────────────────────────
    mag_gray  = np.clip(_reconstruct_laplacian_batch(out_lap), 0.0, 1.0)  # T×h×w
    diff_small = (mag_gray - orig_gray)[:, :, :, np.newaxis]              # T×h×w×1

    # ── Upscale diff to original resolution ──────────────────────────────
    if clip_small.shape[1] != H_orig or clip_small.shape[2] != W_orig:
        diff_up = np.stack(
            [cv2.resize(diff_small[t, :, :, 0], (W_orig, H_orig),
                        interpolation=cv2.INTER_LINEAR)
             for t in range(T)],
            axis=0
        )[:, :, :, np.newaxis]
    else:
        diff_up = diff_small

    # ── Apply diff to colour clip ─────────────────────────────────────────
    mag_clip = np.clip(clip + diff_up, 0.0, 1.0)

    # ── Stitch: magnified apex segment + original remainder ───────────────
    if T_full > T:
        result = np.concatenate([mag_clip, frames[T:]], axis=0)
    else:
        result = mag_clip

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. VIDEO I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_video(frames: np.ndarray, out_path: Path, fps: float):
    """frames : T×H×W×3 float32 [0,1]"""
    if frames is None or frames.shape[0] == 0:
        return
    T, H, W, _ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    uint8  = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    for t in range(T):
        writer.write(cv2.cvtColor(uint8[t], cv2.COLOR_RGB2BGR))
    writer.release()


# ═════════════════════════════════════════════════════════════════════════════
# 4. WORKER  (runs in a subprocess)
# ═════════════════════════════════════════════════════════════════════════════

def _process_one(args: dict) -> dict:
    """
    Top-level function (must be picklable for multiprocessing).
    Returns a result dict with keys: stem, emotion, subject, apex_frame,
    status, magnified_path, error.
    """
    video_path     = Path(args["video_path"])
    emotion        = args["emotion"]
    apex_frame     = args["apex_frame"]
    subject        = args["subject"]
    filename       = args["filename"]
    magnified_dir  = Path(args["magnified_dir"])
    freq_low       = args["freq_low"]
    freq_high      = args["freq_high"]
    alpha_max      = args["alpha_max"]
    levels         = args["levels"]
    proc_width     = args["proc_width"]

    stem   = Path(filename).stem
    result = dict(stem=stem, emotion=emotion, subject=subject,
                  apex_frame=apex_frame, status="ok",
                  magnified_path="", error="")

    try:
        frames, fps = read_video_frames(video_path)
        if frames is None or frames.shape[0] == 0:
            result["status"] = "failed"
            result["error"]  = "could not read frames"
            return result

        apex_clamped = clamp_frame_idx(apex_frame, frames.shape[0])

        magnified = riesz_evm(
            frames, apex_clamped, fps,
            freq_low=freq_low, freq_high=freq_high,
            alpha_max=alpha_max, levels=levels,
            proc_width=proc_width,
        )

        out_path = magnified_dir / f"{stem}_magnified.mp4"
        save_video(magnified, out_path, fps)

        result["magnified_path"] = str(out_path)
        result["fps"]            = fps
        result["apex_clamped"]   = apex_clamped
        result["n_frames"]       = frames.shape[0]

    except Exception as e:
        result["status"] = "failed"
        result["error"]  = str(e)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    medusa_root:    Path,
    output_dir:     Path,
    alpha_max:      float = 15.0,
    freq_low:       float = 0.2,
    freq_high:      float = 4.0,
    pyramid_levels: int   = 3,
    proc_width:     int   = 256,
    num_workers:    int   = 4,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    magnified_dir = output_dir / "magnified"
    preview_dir   = output_dir / "previews"
    magnified_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("Riesz Pyramid EVM  –  Optimised Magnification Pipeline")
    logger.info("=" * 60)
    logger.info(f"Workers      : {num_workers}")
    logger.info(f"Proc width   : {proc_width}px")
    logger.info(f"Pyramid lvls : {pyramid_levels}")
    logger.info(f"Alpha max    : {alpha_max}")
    logger.info(f"Bandpass     : {freq_low}–{freq_high} Hz")

    dataset_root = medusa_root / "data"
    csv_path     = medusa_root / "video_emotion_metadata_v2.csv"

    if not csv_path.exists():
        logger.error(f"Metadata CSV not found: {csv_path}")
        return
    if not dataset_root.exists():
        logger.error(f"Dataset root not found: {dataset_root}")
        return

    records = load_metadata(csv_path, dataset_root, logger)
    if not records:
        logger.error("No valid records found. Exiting.")
        return

    # Build args list for workers
    job_args = [
        {
            "video_path":    str(r["video_path"]),
            "emotion":       r["emotion"],
            "apex_frame":    r["apex_frame"],
            "subject":       r["subject"],
            "filename":      r["filename"],
            "magnified_dir": str(magnified_dir),
            "freq_low":      freq_low,
            "freq_high":     freq_high,
            "alpha_max":     alpha_max,
            "levels":        pyramid_levels,
            "proc_width":    proc_width,
        }
        for r in records
    ]

    total      = len(job_args)
    succeeded  = 0
    failed     = 0
    previews_saved  = {}   # emotion → magnified np.ndarray for preview
    previews_fps    = {}

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_one, a): a for a in job_args}

        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            stem    = res["stem"]
            emotion = res["emotion"]

            if res["status"] == "failed":
                logger.warning(f"[{i}/{total}] FAILED  {stem}  – {res['error']}")
                failed += 1
                continue

            logger.info(
                f"[{i}/{total}]  {stem}  |  {emotion}  "
                f"apex={res['apex_clamped']}/{res['n_frames']}  "
                f"→  {res['magnified_path']}"
            )
            succeeded += 1

            # Collect one preview per emotion (read back the saved file)
            if emotion not in previews_saved:
                arr, fps = read_video_frames(Path(res["magnified_path"]))
                if arr is not None:
                    previews_saved[emotion] = arr
                    previews_fps[emotion]   = fps

    # Save previews (done in main process to avoid multiprocessing conflicts)
    for emotion, arr in previews_saved.items():
        preview_path = preview_dir / f"{emotion}_preview.mp4"
        save_video(arr, preview_path, previews_fps[emotion])
        logger.info(f"Preview saved → {preview_path}")

    elapsed = time.time() - t0
    avg     = elapsed / max(succeeded, 1)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s  |  avg {avg:.2f}s/video")
    logger.info(f"Succeeded : {succeeded}  |  Failed/Skipped : {failed}")
    logger.info(f"Magnified : {magnified_dir}")
    logger.info(f"Previews  : {preview_dir}")
    logger.info("=" * 60)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Riesz Pyramid EVM – optimised micro-expression magnification"
    )
    parser.add_argument("--medusa_root",  type=str,   default=".",
                        help="Path to medusa root (contains data/ and CSV)")
    parser.add_argument("--output_dir",   type=str,   default="output",
                        help="Where to save outputs")
    parser.add_argument("--alpha_max",    type=float, default=15.0,
                        help="Peak amplification α at apex frame (default 15)")
    parser.add_argument("--freq_low",     type=float, default=0.2,
                        help="Bandpass low frequency Hz (default 0.2)")
    parser.add_argument("--freq_high",    type=float, default=4.0,
                        help="Bandpass high frequency Hz (default 4.0)")
    parser.add_argument("--levels",       type=int,   default=3,
                        help="Riesz pyramid levels (default 3)")
    parser.add_argument("--proc_width",   type=int,   default=256,
                        help="Downscale width for EVM processing (default 256)")
    parser.add_argument("--workers",      type=int,   default=4,
                        help="Parallel worker processes (default 4)")
    args = parser.parse_args()

    run_pipeline(
        medusa_root    = Path(args.medusa_root),
        output_dir     = Path(args.output_dir),
        alpha_max      = args.alpha_max,
        freq_low       = args.freq_low,
        freq_high      = args.freq_high,
        pyramid_levels = args.levels,
        proc_width     = args.proc_width,
        num_workers    = args.workers,
    )