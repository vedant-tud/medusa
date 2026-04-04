import argparse
import importlib
import inspect
import os
import re
import sys
from typing import Dict, List, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


TARGET_CLASSES = ["Positive", "Negative", "Surprise"]
EMOTION_TO_ID = {
    "positive": 0,
    "happiness": 0,
    "negative": 1,
    "disgust": 1,
    "repression": 1,
    "anger": 1,
    "fear": 1,
    "sadness": 1,
    "surprise": 2,
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# As requested by the user agreement: do not publish/select these subjects.
DEFAULT_EXCLUDED_SUBJECTS: Set[int] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    22, 24, 27, 28, 29,
    33, 35, 37,
    40, 42, 43, 47,
    52, 53, 56,
    60, 61, 62, 63, 65, 67,
    70, 72, 74, 77, 78,
    81, 84, 87,
    90, 93, 94, 95, 97, 98, 99,
    101, 109,
    110, 111, 112, 113, 114, 116, 118,
    120, 121, 129,
    130, 131, 134, 135,
    142, 147,
    153, 155, 157,
    160, 163, 164, 165, 166, 167, 168,
    171, 172, 173, 174, 175, 177, 178, 179,
    180, 182, 183, 185, 188,
    190, 191, 196, 198, 199,
    200, 203, 204, 206,
    210, 212,
}

EXPERIMENTS = [
    {
        "name": "exp_00_dual_cnn_scl_auxflow",
        "dir": "exp_00_dual_cnn",
        "checkpoint": "save_dual_cnn/best_scl_dual_cnn_fold1_apex_rgb.pth",
        "image_size": 112,
        "loader": "medusa_dual_cnn",
    },
    {
        "name": "exp_01_face_id_baseline",
        "dir": "exp_01_face_id_baseline",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 256,
    },
    {
        "name": "exp_02_dual_swinv2_flow_resnet50_spatial",
        "dir": "exp_02_emotion_transformer",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 256,
    },
    {
        "name": "exp_03_dual_convnext_flow_vit_spatial",
        "dir": "exp_03_all_emotion",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 224,
    },
    {
        "name": "exp_04_dual_vggface2_flow_swin_spatial",
        "dir": "exp_04_vggface2_baseline",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 256,
    },
    {
        "name": "exp_05_single_swin_apex",
        "dir": "exp_05_single_swin_apex",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 256,
    },
    {
        "name": "exp_06_single_resnet_flow",
        "dir": "exp_06_single_resnet_flow",
        "checkpoint": "save_models/best_model_fold1.pth",
        "image_size": 256,
    },
]


class SpatialInputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, fixed_flow: torch.Tensor):
        super().__init__()
        self.model = model
        self.fixed_flow = fixed_flow

    def forward(self, spatial_x: torch.Tensor) -> torch.Tensor:
        return self.model(self.fixed_flow, spatial_x)


class FlowInputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, fixed_spatial: torch.Tensor):
        super().__init__()
        self.model = model
        self.fixed_spatial = fixed_spatial

    def forward(self, flow_x: torch.Tensor) -> torch.Tensor:
        return self.model(flow_x, self.fixed_spatial)


def subject_id_from_subdir_name(subdir: str) -> int:
    m = re.search(r"(\d+)", subdir)
    if m is None:
        raise ValueError(f"Could not parse subject id from '{subdir}'")
    return int(m.group(1))


def subject_id_from_clip_rel(clip_rel: str) -> int:
    subject_part = clip_rel.split("/", 1)[0]
    return subject_id_from_subdir_name(subject_part)


def parse_info_file(info_path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            data[key.strip().lower()] = val.strip().lower()
    return data


def build_single_row_df(data_root: str, clip_rel: str) -> pd.DataFrame:
    clip_folder = os.path.join(data_root, clip_rel)
    info_path = os.path.join(clip_folder, "info.txt")
    flow_path = os.path.join(clip_folder, "flow.npy")
    flow_flip_path = os.path.join(clip_folder, "flow_flip.npy")

    if not os.path.isdir(clip_folder):
        raise FileNotFoundError(f"Missing clip folder: {clip_folder}")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"Missing info.txt: {info_path}")
    if not os.path.isfile(flow_path):
        raise FileNotFoundError(f"Missing flow.npy: {flow_path}")

    info = parse_info_file(info_path)
    raw_emotion = info.get("emotion", "")
    if raw_emotion not in EMOTION_TO_ID:
        raise ValueError(f"Unsupported emotion '{raw_emotion}' in {info_path}")

    return pd.DataFrame(
        [
            {
                "subject": os.path.basename(os.path.dirname(clip_folder)),
                "clip_folder": clip_folder,
                "emotion_id": EMOTION_TO_ID[raw_emotion],
                "flow_npy": flow_path,
                "flow_flip_npy": flow_flip_path,
            }
        ]
    )


def parse_excluded_subjects(arg_value: str) -> Set[int]:
    cleaned = (arg_value or "").strip()
    if cleaned.lower() in {"", "default"}:
        return set(DEFAULT_EXCLUDED_SUBJECTS)

    out: Set[int] = set()
    for part in cleaned.split(","):
        token = part.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def list_valid_clips_for_subject(data_root: str, subject_dir: str) -> List[str]:
    subject_path = os.path.join(data_root, subject_dir)
    if not os.path.isdir(subject_path):
        return []

    clips: List[str] = []
    for clip_name in sorted(os.listdir(subject_path)):
        clip_path = os.path.join(subject_path, clip_name)
        if not os.path.isdir(clip_path):
            continue
        if (
            os.path.isfile(os.path.join(clip_path, "info.txt"))
            and os.path.isfile(os.path.join(clip_path, "flow.npy"))
        ):
            clips.append(clip_name)
    return clips


def get_clip_label_id(data_root: str, subject_dir: str, clip_name: str) -> int:
    info_path = os.path.join(data_root, subject_dir, clip_name, "info.txt")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"Missing info file: {info_path}")
    info = parse_info_file(info_path)
    raw_emotion = info.get("emotion", "")
    if raw_emotion not in EMOTION_TO_ID:
        raise ValueError(f"Unsupported emotion '{raw_emotion}' in {info_path}")
    return int(EMOTION_TO_ID[raw_emotion])


def select_random_subject_clips(
    data_root: str,
    num_subjects: int,
    seed: int,
    excluded_subjects: Set[int],
) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for subject_dir in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        try:
            subject_id = subject_id_from_subdir_name(subject_dir)
        except ValueError:
            continue
        if subject_id in excluded_subjects:
            continue

        clips = list_valid_clips_for_subject(data_root, subject_dir)
        if not clips:
            continue

        candidates.append(
            {
                "subject_id": subject_id,
                "subject_dir": subject_dir,
                "clips": clips,
            }
        )

    if len(candidates) < num_subjects:
        raise RuntimeError(
            f"Only {len(candidates)} allowed subjects available, requested {num_subjects}."
        )

    rng = np.random.default_rng(seed)
    selected_idx = rng.choice(len(candidates), size=num_subjects, replace=False)

    selected: List[Dict[str, str]] = []
    for idx in selected_idx:
        candidate = candidates[int(idx)]
        clip_options = candidate["clips"]
        clip_name = clip_options[int(rng.integers(0, len(clip_options)))]
        clip_rel = f"{candidate['subject_dir']}/{clip_name}"
        selected.append(
            {
                "subject_id": int(candidate["subject_id"]),
                "subject_dir": candidate["subject_dir"],
                "clip": clip_name,
                "clip_rel": clip_rel,
            }
        )

    selected.sort(key=lambda x: int(x["subject_id"]))
    return selected


def select_balanced_subject_clips(
    data_root: str,
    num_per_label: int,
    seed: int,
    excluded_subjects: Set[int],
) -> List[Dict[str, str]]:
    # label_id -> subject_id -> list[clip_name]
    label_subject_clips: Dict[int, Dict[int, List[str]]] = {0: {}, 1: {}, 2: {}}
    label_subject_dirs: Dict[int, Dict[int, str]] = {0: {}, 1: {}, 2: {}}

    for subject_dir in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        try:
            subject_id = subject_id_from_subdir_name(subject_dir)
        except ValueError:
            continue

        if subject_id in excluded_subjects:
            continue

        clips = list_valid_clips_for_subject(data_root, subject_dir)
        for clip_name in clips:
            try:
                label_id = get_clip_label_id(data_root, subject_dir, clip_name)
            except Exception:
                continue

            if subject_id not in label_subject_clips[label_id]:
                label_subject_clips[label_id][subject_id] = []
                label_subject_dirs[label_id][subject_id] = subject_dir
            label_subject_clips[label_id][subject_id].append(clip_name)

    rng = np.random.default_rng(seed)
    selected: List[Dict[str, str]] = []
    used_subject_ids: Set[int] = set()

    for label_id in [0, 1, 2]:
        subject_ids = sorted(label_subject_clips[label_id].keys())
        subject_ids = [sid for sid in subject_ids if sid not in used_subject_ids]

        if len(subject_ids) < num_per_label:
            raise RuntimeError(
                f"Not enough allowed unique subjects for label {TARGET_CLASSES[label_id]}: "
                f"have {len(subject_ids)}, need {num_per_label}."
            )

        chosen_ids = rng.choice(subject_ids, size=num_per_label, replace=False)
        for sid in chosen_ids:
            sid_int = int(sid)
            subject_dir = label_subject_dirs[label_id][sid_int]
            clip_options = label_subject_clips[label_id][sid_int]
            clip_name = clip_options[int(rng.integers(0, len(clip_options)))]
            clip_rel = f"{subject_dir}/{clip_name}"
            selected.append(
                {
                    "subject_id": sid_int,
                    "subject_dir": subject_dir,
                    "clip": clip_name,
                    "clip_rel": clip_rel,
                    "label_id": label_id,
                    "label_name": TARGET_CLASSES[label_id],
                }
            )
            used_subject_ids.add(sid_int)

    selected.sort(key=lambda x: (int(x["label_id"]), int(x["subject_id"])))
    return selected


def cleanup_local_modules() -> None:
    for name in ["dataset", "models", "utils", "trainKfold_scl_cnn"]:
        if name in sys.modules:
            del sys.modules[name]


def load_exp_modules(exp_dir: str):
    cleanup_local_modules()
    sys.path.insert(0, exp_dir)
    try:
        dataset_mod = importlib.import_module("dataset")
        models_mod = importlib.import_module("models")
    finally:
        # Keep imported modules in memory, but remove temporary import path.
        if sys.path and sys.path[0] == exp_dir:
            sys.path.pop(0)
    return dataset_mod, models_mod


def load_medusa_exp0_module(exp_dir: str):
    cleanup_local_modules()
    sys.path.insert(0, exp_dir)
    try:
        medusa_mod = importlib.import_module("trainKfold_scl_cnn")
    finally:
        if sys.path and sys.path[0] == exp_dir:
            sys.path.pop(0)
    return medusa_mod


def instantiate_model(models_mod, device: torch.device) -> torch.nn.Module:
    model_cls = getattr(models_mod, "DualStreamModel", None)
    if model_cls is None:
        model_cls = getattr(models_mod, "SingleStreamModel")

    sig = inspect.signature(model_cls.__init__)
    kwargs = {}
    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = 3
    if "in_channels" in sig.parameters:
        kwargs["in_channels"] = 3

    model = model_cls(**kwargs).to(device)
    return model


def reshape_transform_tokens(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        # Swin commonly emits [B, H, W, C].
        if tensor.shape[-1] > tensor.shape[1]:
            return tensor.permute(0, 3, 1, 2)
        return tensor
    if tensor.ndim == 3:
        bsz, tokens, channels = tensor.shape
        side = int(tokens**0.5)
        if side * side == tokens:
            return tensor.transpose(1, 2).reshape(bsz, channels, side, side)
        return tensor.transpose(1, 2).unsqueeze(-1)
    return tensor


def pick_target_layer(stream: torch.nn.Module):
    if hasattr(stream, "layer4") and len(stream.layer4) > 0:
        return stream.layer4[-1], None

    if hasattr(stream, "layers") and len(stream.layers) > 0:
        last_stage = stream.layers[-1]
        if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
            block = last_stage.blocks[-1]
            return getattr(block, "norm1", block), reshape_transform_tokens

    if hasattr(stream, "blocks") and len(stream.blocks) > 0:
        block = stream.blocks[-1]
        return getattr(block, "norm1", block), reshape_transform_tokens

    conv_layers = [m for m in stream.modules() if isinstance(m, torch.nn.Conv2d)]
    if conv_layers:
        return conv_layers[-1], None

    raise RuntimeError("Unable to infer CAM target layer")


def tensor_to_rgb_image(tensor_chw: torch.Tensor) -> np.ndarray:
    arr = tensor_chw.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError("Expected CHW tensor")

    if arr.shape[0] == 2:
        # Optical-flow tensors (u,v) are converted to HSV/RGB for visualization.
        u = arr[0].astype(np.float32)
        v = arr[1].astype(np.float32)
        mag, ang = cv2.cartToPolar(u, v)

        hsv = np.zeros((arr.shape[1], arr.shape[2], 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8((ang * 180 / np.pi) / 2)
        hsv[..., 1] = 255

        scale = float(np.percentile(mag, 99.0))
        if scale <= 1e-6:
            scale = 1.0
        mag_norm = np.clip(mag / scale, 0.0, 1.0)
        hsv[..., 2] = np.uint8(mag_norm * 255.0)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)

    if arr.shape[2] == 3:
        arr = arr * IMAGENET_STD + IMAGENET_MEAN

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_weights(model: torch.nn.Module, model_path: str, device: torch.device) -> str:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(checkpoint, strict=True)
        return "strict"
    except RuntimeError:
        model.load_state_dict(checkpoint, strict=False)
        return "non-strict"


def build_dataset(dataset_mod, row_df: pd.DataFrame, image_size: int):
    dataset_cls = getattr(dataset_mod, "CasmeDualSwinDataset")
    sig = inspect.signature(dataset_cls.__init__)

    kwargs = {
        "annotations": row_df,
        "augment": False,
    }
    if "image_size" in sig.parameters:
        kwargs["image_size"] = image_size
    if "frame_type" in sig.parameters:
        kwargs["frame_type"] = "apex"
    if "frame_channels" in sig.parameters:
        kwargs["frame_channels"] = "rgb"

    return dataset_cls(**kwargs)


def save_per_subject_comparison_plots(summary_df: pd.DataFrame, batch_root: str) -> None:
    exp_order = [cfg["name"] for cfg in EXPERIMENTS]
    out_dir = os.path.join(batch_root, "subject_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    index_rows: List[Dict[str, str]] = []

    group_cols = ["subject_id", "clip"]
    grouped = summary_df.groupby(group_cols, dropna=False)

    for (subject_id, clip_rel), group_df in grouped:
        cols = 4
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.6 * rows))
        axes_flat = axes.flatten()

        for i, exp_name in enumerate(exp_order):
            ax = axes_flat[i]
            row = group_df[group_df["experiment"] == exp_name]
            if row.empty:
                ax.axis("off")
                continue

            rr = row.iloc[0]
            img_path = rr["image"]
            if not os.path.exists(img_path):
                ax.axis("off")
                continue

            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_title(
                f"{exp_name}\n{rr['branch']} | T:{rr['true']} P:{rr['pred']}",
                fontsize=8,
            )
            ax.axis("off")

        for j in range(len(exp_order), len(axes_flat)):
            axes_flat[j].axis("off")

        clip_slug = str(clip_rel).replace("/", "_")
        out_png = os.path.join(out_dir, f"sub{int(subject_id)}_{clip_slug}_comparison.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=140)
        plt.close(fig)

        index_rows.append(
            {
                "subject_id": int(subject_id),
                "clip": str(clip_rel),
                "comparison_image": out_png,
            }
        )

    if index_rows:
        pd.DataFrame(index_rows).to_csv(os.path.join(out_dir, "comparison_index.csv"), index=False)


def run_one_experiment(
    workspace_root: str,
    exp_cfg: Dict[str, str],
    row_df: pd.DataFrame,
    clip_rel: str,
    out_root: str,
    device: torch.device,
) -> Tuple[str, int, int, str, str]:
    exp_name = exp_cfg["name"]
    exp_dir = os.path.join(workspace_root, exp_cfg["dir"])
    model_path = os.path.join(workspace_root, exp_cfg["dir"], exp_cfg["checkpoint"])
    exp_loader = exp_cfg.get("loader", "standard")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if exp_loader == "medusa_dual_cnn":
        medusa_mod = load_medusa_exp0_module(exp_dir)
        dataset = medusa_mod.CasmeDualCNNDataset(
            row_df,
            image_size=int(exp_cfg["image_size"]),
            augment=False,
            frame_type="apex",
            frame_channels="rgb",
        )
        (flow_x, spatial_x), label = dataset[0]

        flow_b = flow_x.unsqueeze(0).to(device)
        spatial_b = spatial_x.unsqueeze(0).to(device)

        model = medusa_mod.DualCnnMER(num_classes=3, spatial_chans=3).to(device)
    else:
        dataset_mod, models_mod = load_exp_modules(exp_dir)
        dataset = build_dataset(dataset_mod, row_df, int(exp_cfg["image_size"]))
        (flow_x, spatial_x), label = dataset[0]

        flow_b = flow_x.unsqueeze(0).to(device)
        spatial_b = spatial_x.unsqueeze(0).to(device)

        model = instantiate_model(models_mod, device)

    load_mode = load_weights(model, model_path, device)
    model.eval()

    with torch.no_grad():
        logits = model(flow_b, spatial_b)
        pred_id = int(torch.argmax(logits, dim=-1).item())

    true_id = int(label if isinstance(label, int) else label.item())
    subject_id = subject_id_from_clip_rel(clip_rel)

    dual_stream = (hasattr(model, "stream1") and hasattr(model, "stream2")) or (
        hasattr(model, "stream_motion") and hasattr(model, "stream_spatial")
    )

    clip_slug = clip_rel.replace("/", "_")
    exp_out_dir = os.path.join(out_root, clip_slug, exp_name)
    os.makedirs(exp_out_dir, exist_ok=True)

    out_png = os.path.join(exp_out_dir, "activation.png")

    if dual_stream:
        branch = "dual"

        if hasattr(model, "stream_motion") and hasattr(model, "stream_spatial"):
            flow_target_layer, flow_reshape = pick_target_layer(model.stream_motion)
            spatial_target_layer, spatial_reshape = pick_target_layer(model.stream_spatial)
        else:
            flow_target_layer, flow_reshape = pick_target_layer(model.stream1)
            spatial_target_layer, spatial_reshape = pick_target_layer(model.stream2)

        flow_wrapper = FlowInputWrapper(model, spatial_b)
        spatial_wrapper = SpatialInputWrapper(model, flow_b)

        flow_cam = GradCAM(model=flow_wrapper, target_layers=[flow_target_layer], reshape_transform=flow_reshape)
        spatial_cam = GradCAM(model=spatial_wrapper, target_layers=[spatial_target_layer], reshape_transform=spatial_reshape)

        flow_gray = flow_cam(input_tensor=flow_b, targets=None)[0, :]
        spatial_gray = spatial_cam(input_tensor=spatial_b, targets=None)[0, :]

        flow_img = tensor_to_rgb_image(flow_x)
        spatial_img = tensor_to_rgb_image(spatial_x)

        flow_overlay = show_cam_on_image(flow_img, flow_gray, use_rgb=True)
        spatial_overlay = show_cam_on_image(spatial_img, spatial_gray, use_rgb=True)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        axes[0, 0].imshow(spatial_img)
        axes[0, 0].set_title("Spatial input")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(spatial_gray, cmap="jet")
        axes[0, 1].set_title("Spatial activation")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(spatial_overlay)
        axes[0, 2].set_title("Spatial overlay")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(flow_img)
        axes[1, 0].set_title("Flow input")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(flow_gray, cmap="jet")
        axes[1, 1].set_title("Flow activation")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(flow_overlay)
        axes[1, 2].set_title("Flow overlay")
        axes[1, 2].axis("off")

        fig.suptitle(
            f"{exp_name} | sub{subject_id} | {clip_rel} | True: {TARGET_CLASSES[true_id]} | Pred: {TARGET_CLASSES[pred_id]}",
            fontsize=12,
        )
    else:
        if hasattr(model, "stream") and hasattr(model.stream, "layer4"):
            branch = "flow"
            target_layer, reshape = pick_target_layer(model.stream)
            wrapper = FlowInputWrapper(model, spatial_b)
            cam_input = flow_b
            base_img = tensor_to_rgb_image(flow_x)
        else:
            branch = "spatial"
            target_layer, reshape = pick_target_layer(model.stream)
            wrapper = SpatialInputWrapper(model, flow_b)
            cam_input = spatial_b
            base_img = tensor_to_rgb_image(spatial_x)

        cam = GradCAM(model=wrapper, target_layers=[target_layer], reshape_transform=reshape)
        grayscale_cam = cam(input_tensor=cam_input, targets=None)[0, :]
        overlay = show_cam_on_image(base_img, grayscale_cam, use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(base_img)
        axes[0].set_title(f"Input ({branch})")
        axes[0].axis("off")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Activation map")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(
            f"{exp_name}\nsub{subject_id} | {clip_rel}\nTrue: {TARGET_CLASSES[true_id]} | Pred: {TARGET_CLASSES[pred_id]}"
        )
        axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    meta_path = os.path.join(exp_out_dir, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"experiment={exp_name}\n")
        f.write(f"subject=sub{subject_id}\n")
        f.write(f"clip={clip_rel}\n")
        f.write(f"branch={branch}\n")
        f.write(f"true={TARGET_CLASSES[true_id]}\n")
        f.write(f"pred={TARGET_CLASSES[pred_id]}\n")
        f.write(f"checkpoint={model_path}\n")
        f.write(f"load_mode={load_mode}\n")

    cleanup_local_modules()
    return exp_name, true_id, pred_id, out_png, branch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-sample activations for all experiment models.")
    parser.add_argument(
        "--workspace_root",
        type=str,
        default="/scratch/smiyyapuram/medusa",
        help="Repository root path.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/smiyyapuram/medusa/casme_tvl1_processed_amp5",
        help="Input flow dataset root.",
    )
    parser.add_argument(
        "--clip_rel",
        type=str,
        default="sub1/e_352_1561",
        help="Clip path relative to data_root.",
    )
    parser.add_argument(
        "--random_subjects",
        type=int,
        default=0,
        help="If >0, randomly select this many allowed subjects and one clip per subject.",
    )
    parser.add_argument(
        "--balanced_per_label",
        type=int,
        default=0,
        help="If >0, select this many allowed subjects per label (Positive/Negative/Surprise).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject/clip sampling.",
    )
    parser.add_argument(
        "--excluded_subjects",
        type=str,
        default="default",
        help="Comma-separated subject ids to exclude, or 'default'.",
    )
    parser.add_argument(
        "--batch_name",
        type=str,
        default="",
        help="Optional subfolder name for random-subject batch output.",
    )
    parser.add_argument(
        "--selected_subjects_csv",
        type=str,
        default="",
        help="Optional CSV with a clip_rel column to reuse a fixed subject/clip list.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/scratch/smiyyapuram/medusa/activations",
        help="Output root for activation figures.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs: List[Dict[str, str]] = []

    if args.selected_subjects_csv.strip():
        selected_df = pd.read_csv(args.selected_subjects_csv)
        if "clip_rel" not in selected_df.columns:
            raise ValueError("selected_subjects_csv must contain a clip_rel column")

        batch_name = args.batch_name.strip() or "selected_subjects_batch"
        batch_root = os.path.join(args.out_root, batch_name)
        os.makedirs(batch_root, exist_ok=True)

        selection_csv = os.path.join(batch_root, "selected_subjects.csv")
        selected_df.to_csv(selection_csv, index=False)
        print(f"Saved selection: {selection_csv}")

        clip_rels = selected_df["clip_rel"].astype(str).tolist()
    elif args.balanced_per_label > 0:
        excluded = parse_excluded_subjects(args.excluded_subjects)
        selected = select_balanced_subject_clips(
            data_root=args.data_root,
            num_per_label=args.balanced_per_label,
            seed=args.seed,
            excluded_subjects=excluded,
        )

        batch_name = args.batch_name.strip() or f"balanced_{args.balanced_per_label}perlabel_seed{args.seed}"
        batch_root = os.path.join(args.out_root, batch_name)
        os.makedirs(batch_root, exist_ok=True)

        selection_csv = os.path.join(batch_root, "selected_subjects.csv")
        pd.DataFrame(selected).to_csv(selection_csv, index=False)
        print(f"Saved selection: {selection_csv}")

        clip_rels = [x["clip_rel"] for x in selected]
    elif args.random_subjects > 0:
        excluded = parse_excluded_subjects(args.excluded_subjects)
        selected = select_random_subject_clips(
            data_root=args.data_root,
            num_subjects=args.random_subjects,
            seed=args.seed,
            excluded_subjects=excluded,
        )

        batch_name = args.batch_name.strip() or f"random_subjects_{args.random_subjects}_seed{args.seed}"
        batch_root = os.path.join(args.out_root, batch_name)
        os.makedirs(batch_root, exist_ok=True)

        selection_csv = os.path.join(batch_root, "selected_subjects.csv")
        pd.DataFrame(selected).to_csv(selection_csv, index=False)
        print(f"Saved selection: {selection_csv}")

        clip_rels = [x["clip_rel"] for x in selected]
    else:
        batch_root = args.out_root
        clip_rels = [args.clip_rel]

    for clip_rel in clip_rels:
        row_df = build_single_row_df(args.data_root, clip_rel)
        subject_id = subject_id_from_clip_rel(clip_rel)

        for exp_cfg in EXPERIMENTS:
            exp_name = exp_cfg["name"]
            try:
                name, true_id, pred_id, out_png, branch = run_one_experiment(
                    workspace_root=args.workspace_root,
                    exp_cfg=exp_cfg,
                    row_df=row_df,
                    clip_rel=clip_rel,
                    out_root=batch_root,
                    device=device,
                )
                runs.append(
                    {
                        "subject_id": subject_id,
                        "clip": clip_rel,
                        "experiment": name,
                        "true": TARGET_CLASSES[true_id],
                        "pred": TARGET_CLASSES[pred_id],
                        "branch": branch,
                        "image": out_png,
                    }
                )
                print(f"[OK] {exp_name} | {clip_rel} -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] {exp_name} | {clip_rel}: {exc}")

    if runs:
        summary_csv = os.path.join(batch_root, "summary.csv")
        summary_df = pd.DataFrame(runs)
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary: {summary_csv}")

        save_per_subject_comparison_plots(summary_df, batch_root)
        print(f"Saved per-subject comparisons: {os.path.join(batch_root, 'subject_comparisons')}")


if __name__ == "__main__":
    main()
