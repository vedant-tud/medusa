"""
Test script to verify all required dependencies are installed.
Run this before attempting inference.

Usage:
    python backend/test_imports.py
"""

import sys
import importlib


def test_import(package_name, import_name=None):
    """Test if a package can be imported."""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20} (v{version})")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20} - {e}")
        return False


def main():
    print("\n=== Medusa Dependency Check ===\n")

    required = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("NumPy", "numpy"),
        ("OpenCV", "cv2"),
        ("scikit-image", "skimage"),
        ("timm (Vision Models)", "timm"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
    ]

    optional = [
        ("facenet_pytorch (MTCNN)", "facenet_pytorch"),
        ("Jupyter", "jupyter"),
        ("Matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
    ]

    print("Required Dependencies:")
    required_ok = sum(test_import(name, pkg) for name, pkg in required)

    print(f"\nOptional Dependencies:")
    optional_ok = sum(test_import(name, pkg) for name, pkg in optional)

    print(f"\n=== Summary ===")
    print(f"Required: {required_ok}/{len(required)} ✓")
    print(f"Optional: {optional_ok}/{len(optional)} ✓")

    if required_ok == len(required):
        print("\n✓ All required dependencies installed!")
        return 0
    else:
        print(f"\n✗ Missing {len(required) - required_ok} required dependency/dependencies")
        print("\nInstall missing packages with:")
        print("  pip install torch torchvision opencv-python timm fastapi uvicorn")
        return 1


if __name__ == "__main__":
    sys.exit(main())
