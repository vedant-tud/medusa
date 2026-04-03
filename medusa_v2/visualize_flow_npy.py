import argparse
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

def main():
    parser = argparse.ArgumentParser(description="Convert an optical flow .npy file to an RGB image block.")
    parser.add_argument("input_npy", type=str, help="Path to the input flow .npy file")
    parser.add_argument("output_image", type=str, help="Path to save the output RGB image (e.g., flow_out.png)")
    
    args = parser.parse_args()
    
    try:
        print(f"Loading {args.input_npy}...")
        rgb_img = flow_npy_to_rgb(args.input_npy)
        
        # OpenCV saves images in BGR format, so we convert back from RGB
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output_image, bgr_img)
        print(f"✅ Successfully saved visual flow to: {args.output_image}")
        
    except Exception as e:
        print(f"❌ Failed to convert flow: {str(e)}")

if __name__ == "__main__":
    main()
