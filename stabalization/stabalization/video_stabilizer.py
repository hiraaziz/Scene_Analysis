import cv2
import numpy as np
import os
from typing import List


def stabilize_video(input_path: str, output_path: str, smoothing_radius: int = 15):
    """
    Video stabilization using per-pixel dense optical flow warping (non-rigid, handles parallax).
    
    Args:
        input_path (str): Path to the input unstable video
        output_path (str): Path to save the stabilized video
        smoothing_radius (int): Radius for flow field smoothing
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    print(f"Processing video: {input_path}")
    print(f"Output: {output_path}")
    print(f"Smoothing radius: {smoothing_radius}")
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'.")
        return False
    
    # Get video properties
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {width}x{height}, {fps} fps, {n_frames} frames")
    
    # Define output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read all frames into memory (for flow accumulation)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    if n_frames < 2:
        print("Not enough frames for stabilization.")
        return False

    # Compute dense optical flow between consecutive frames
    print("Phase 1: Computing dense optical flow fields...")
    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, n_frames):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
        prev_gray = curr_gray
        if (i+1) % 20 == 0 or i+1 == n_frames:
            print(f"Computed flow for {i+1}/{n_frames} frames")

    # Accumulate flow fields to get per-frame warp to the first frame
    print("Phase 2: Accumulating flow fields...")
    h, w = frames[0].shape[:2]
    accumulated_flows = [np.zeros((h, w, 2), dtype=np.float32)]
    for i in range(1, n_frames):
        prev_acc = accumulated_flows[-1]
        flow = flows[i-1]
        # Warp previous accumulated flow by current flow
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow[..., 0]).astype(np.float32)
        map_y = (map_y + flow[..., 1]).astype(np.float32)
        warped_prev_acc = cv2.remap(prev_acc, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        accumulated_flows.append(warped_prev_acc + flow)
        if (i+1) % 20 == 0 or i+1 == n_frames:
            print(f"Accumulated flow for {i+1}/{n_frames} frames")

    # Smooth the accumulated flows over time (temporal smoothing)
    print("Phase 3: Smoothing flow fields...")
    smoothed_flows = smooth_flow_fields(accumulated_flows, smoothing_radius)

    # Compute stabilization flow: difference between smoothed and original accumulated flow
    print("Phase 4: Computing stabilization warps and writing output...")
    for i in range(n_frames):
        flow_correction = smoothed_flows[i] - accumulated_flows[i]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + flow_correction[..., 0]).astype(np.float32)
        map_y = (map_y + flow_correction[..., 1]).astype(np.float32)
        stabilized = cv2.remap(frames[i], map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # --- Wobble reduction: blend stabilized with original using a spatial mask ---
        # Create a mask that is 1 in the center, 0 at the borders (cosine window)
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xv, yv = np.meshgrid(x, y)
        # Correct mask shape: (h, w, 1)
        mask_1d_x = 0.5 * (1 + np.cos(np.pi * xv[0]))  # shape (w,)
        mask_1d_y = 0.5 * (1 + np.cos(np.pi * yv[:,0]))  # shape (h,)
        mask_2d = np.outer(mask_1d_y, mask_1d_x)  # shape (h, w)
        mask = mask_2d[..., None].astype(np.float32)  # shape (h, w, 1)
        # Blend: more stabilized in center, more original at borders
        blended = stabilized.astype(np.float32) * mask + frames[i].astype(np.float32) * (1 - mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        # Optional: apply a small Gaussian blur to further reduce high-frequency artifacts
        blended = cv2.GaussianBlur(blended, (3, 3), 0.5)
        out.write(blended)
        if (i+1) % 20 == 0 or i+1 == n_frames:
            print(f"Stabilized {i+1}/{n_frames} frames")
    
    # Release resources
    out.release()
    print(f"Video stabilization complete. Output saved to: {output_path}")
    return True


def smooth_flow_fields(flows: List[np.ndarray], radius: int) -> List[np.ndarray]:
    """
    Smooth a list of flow fields temporally using a moving average window.
    
    Args:
        flows: List of (H, W, 2) flow fields
        radius: Smoothing radius (window size = 2*radius+1)
    
    Returns:
        List of smoothed flow fields
    """
    n = len(flows)
    h, w, _ = flows[0].shape
    smoothed = []
    for i in range(n):
        # Collect window of flows
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        window = flows[start:end]
        # Pad if at the edges
        if len(window) < 2*radius+1:
            if i < radius:
                pad = [flows[0]] * (radius - i)
                window = pad + window
            if i + radius + 1 > n:
                pad = [flows[-1]] * (i + radius + 1 - n)
                window = window + pad
        # Average the flows in the window
        avg_flow = np.mean(window, axis=0)
        smoothed.append(avg_flow)
    return smoothed


if __name__ == "__main__":
    import argparse
    import os
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Video stabilization using per-pixel dense optical flow warping")
    parser.add_argument("--input", type=str, default=os.path.join(script_dir, "test10.mp4"),
                        help="Input video file path")
    parser.add_argument("--output", type=str, default=os.path.join(script_dir, "stabilized_video.mp4"),
                        help="Output video file path")
    parser.add_argument("--smooth", type=int, default=10,
                        help="Smoothing radius (default: 15)")
    
    args = parser.parse_args()
    
    # Run video stabilization
    stabilize_video(args.input, args.output, args.smooth)
