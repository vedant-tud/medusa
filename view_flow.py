import argparse
import os
import glob

import numpy as np
import cv2


def flow_npy_to_rgb(flow_npy_path: str) -> np.ndarray:
    """
    Loads an optical flow .npy file (shape: 2, H, W) and converts
    it to an RGB image representation using the HSV color wheel.
    """
    # Load the flow tensor from the .npy file
    flow = np.load(flow_npy_path).astype(np.float32)

    # Extract the horizontal (U) and vertical (V) components
    fl_u, fl_v = flow[0], flow[1]

    # Calculate magnitude and angle
    fl_mag, fl_ang = cv2.cartToPolar(fl_u, fl_v)

    # Initialize an HSV image array
    h, w = flow.shape[1], flow.shape[2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    # Map angles to Hue (OpenCV Hue domain is 0-179)
    hsv[..., 0] = fl_ang * 180 / np.pi / 2
    # Maximize Saturation for vibrant colors
    hsv[..., 1] = 255
    # Normalize motion magnitude to Value (0-255)
    hsv[..., 2] = cv2.normalize(fl_mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the HSV representation back to standard RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb


def process_single(input_npy: str, output_image: str):
    """Convert a single flow .npy to an RGB image."""
    print(f"Loading {input_npy}...")
    rgb_img = flow_npy_to_rgb(input_npy)
    # OpenCV saves images in BGR format, so we convert back from RGB
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image, bgr_img)
    print(f"  Saved -> {output_image}")


def process_batch(data_root: str, filename: str = "flow.npy", out_name: str = "flow_vis.png"):
    """
    Walk data_root recursively, find all files matching `filename`,
    and save a visualisation named `out_name` in the same directory.
    """
    pattern = os.path.join(data_root, "**", filename)
    npy_files = sorted(glob.glob(pattern, recursive=True))

    if not npy_files:
        print(f"No '{filename}' files found under: {data_root}")
        return

    print(f"Found {len(npy_files)} flow files. Processing...")
    success, failed = 0, []

    for npy_path in npy_files:
        out_path = os.path.join(os.path.dirname(npy_path), out_name)
        try:
            process_single(npy_path, out_path)
            success += 1
        except Exception as e:
            print(f"  Failed {npy_path}: {e}")
            failed.append(npy_path)

    print(f"\nDone. {success}/{len(npy_files)} converted successfully.")
    if failed:
        print(f"{len(failed)} failed:")
        for f in failed:
            print(f"  {f}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert optical flow .npy file(s) to RGB images."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- single mode ---
    single = subparsers.add_parser("single", help="Convert one .npy file")
    single.add_argument("input_npy", type=str, help="Path to the input flow .npy file")
    single.add_argument("output_image", type=str, help="Path to save the output image (e.g. flow_out.png)")

    # --- batch mode ---
    batch = subparsers.add_parser("batch", help="Convert all flow.npy files under a directory")
    batch.add_argument("data_root", type=str, help="Root directory to search (e.g. casme_raft_processed10)")
    batch.add_argument("--filename", type=str, default="flow.npy", help="Filename to look for (default: flow.npy)")
    batch.add_argument("--out-name", type=str, default="flow_vis.png", help="Output filename saved next to each .npy (default: flow_vis.png)")

    args = parser.parse_args()

    if args.mode == "single":
        try:
            process_single(args.input_npy, args.output_image)
            print(f"Successfully saved visual flow to: {args.output_image}")
        except Exception as e:
            print(f"Failed to convert flow: {e}")

    elif args.mode == "batch":
        process_batch(args.data_root, filename=args.filename, out_name=args.out_name)


if __name__ == "__main__":
    main()