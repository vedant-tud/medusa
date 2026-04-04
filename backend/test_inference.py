"""
Test inference pipeline on a sample video.

This script:
1. Loads a test video
2. Runs the complete pipeline (frames → apex → onset → flow → inference)
3. Displays results and timing information

Usage:
    python backend/test_inference.py --video path/to/video.mp4 --model path/to/checkpoint.pth

    Optional:
    --video path/to/video.mp4        Path to test video (default: test_video.mp4)
    --model path/to/checkpoint.pth   Path to emotion model checkpoint (required)
    --device cuda/cpu                Device to use (default: auto-detect)
    --output path/to/output.json     Save results to JSON (optional)
"""

import argparse
import sys
import time
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Test Medusa inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--video", type=str, default="test_video.mp4",
                        help="Path to test video")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to emotion model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cuda", "cpu", "mps", "auto"],
                        help="Device to use (auto = CUDA if available else CPU)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Import here to check dependencies
    try:
        import torch
        import cv2
        import numpy as np
        from backend.pipeline import load_model, process_video
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Run: python backend/test_imports.py")
        return 1

    # Check files exist
    if not Path(args.video).exists():
        print(f"Error: Video not found - {args.video}")
        return 1

    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found - {args.model}")
        return 1

    print("\n=== Medusa Inference Test ===\n")

    # Detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Video: {args.video}")
    print(f"Model: {args.model}\n")

    try:
        # Load model
        print("Loading model...")
        start = time.time()
        model, loaded_device = load_model(args.model, device=device)
        load_time = time.time() - start
        print(f"✓ Model loaded ({load_time:.2f}s)\n")

        # Process video
        print("Processing video...\n")
        start = time.time()
        result = process_video(args.video, model, loaded_device)
        process_time = time.time() - start

        # Print results
        print(f"\n✓ Inference complete ({process_time:.2f}s)\n")
        print("=== Results ===")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['group_probs']}")
        print(f"Apex Frame: {result['apex_frame_index']} / {result['total_frames']}")
        print(f"Onset Frame: {result['onset_frame_index']}")
        print(f"Video FPS: {result['fps']}")

        results = {
            "status": "success",
            "emotion": result['emotion'],
            "confidence": result['confidence'],
            "probabilities": result['group_probs'],
            "apex_frame": result['apex_frame_index'],
            "onset_frame": result['onset_frame_index'],
            "total_frames": result['total_frames'],
            "fps": result['fps'],
            "load_time_seconds": load_time,
            "process_time_seconds": process_time,
            "total_time_seconds": load_time + process_time,
            "device": str(device),
        }

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results = {
            "status": "error",
            "error": str(e),
            "device": str(device),
        }
        return 1
    finally:
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
