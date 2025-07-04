import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

def meshflow_stabilize(input_path, output_path='meshflow_stabilized.mp4', mesh_size=10):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("❌ Could not open video.")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Optimization: Use lower resolution for optical flow computation
    scale_factor = 0.5
    w_small, h_small = int(w * scale_factor), int(h * scale_factor)

    ret, prev = cap.read()
    if not ret:
        raise IOError("❌ Could not read first frame.")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray_small = cv2.resize(prev_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    out.write(prev)

    # Show resized frame (step 1)
    for i in range(2):
        cv2.imshow('Resized Frame', prev_gray_small)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Mesh parameters on small image
    # mesh_y_small, mesh_x_small: mesh grid coordinates (row, col) for mesh cells
    mesh_y_small, mesh_x_small = np.mgrid[0:h_small:mesh_size, 0:w_small:mesh_size]
    mesh_h_small, mesh_w_small = mesh_y_small.shape
    # mesh_trajectories: stores the motion vectors for each mesh cell for each frame
    mesh_trajectories = np.zeros((n_frames - 1, mesh_h_small, mesh_w_small, 2), dtype=np.float32)

    for i in range(n_frames - 1):
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_gray_small = cv2.resize(curr_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        
        # Show resized current frame (step 2)
        if i < 2:
            cv2.imshow('Resized Current Frame', curr_gray_small)
            cv2.waitKey(0)
        if i == 1:
            cv2.destroyAllWindows()
        
        # Compute optical flow on small image
        flow = cv2.calcOpticalFlowFarneback(prev_gray_small, curr_gray_small, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        # Show optical flow magnitude (step 3)
        if i < 2:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imshow('Optical Flow Magnitude', flow_vis)
            cv2.waitKey(0)
        if i == 1:
            cv2.destroyAllWindows()
        
        # mesh_y_small, mesh_x_small: mesh grid coordinates (row, col) for mesh cells
        # flow_x, flow_y: motion vectors for each mesh cell
        flow_x = flow[mesh_y_small, mesh_x_small, 0]
        flow_y = flow[mesh_y_small, mesh_x_small, 1]
        # mesh_trajectories[i]: stores the motion vector for each mesh cell at frame i
        mesh_trajectories[i] = np.stack([flow_x, flow_y], axis=-1)
        
        # Visualize mesh grid and motion (step 4)
        if i < 2:
            vis = cv2.cvtColor(curr_gray_small, cv2.COLOR_GRAY2BGR)
            for y in range(mesh_h_small):
                for x in range(mesh_w_small):
                    pt1 = (int(mesh_x_small[y, x]), int(mesh_y_small[y, x]))
                    dx, dy = flow_x[y, x], flow_y[y, x]
                    pt2 = (int(pt1[0] + dx * 5), int(pt1[1] + dy * 5))
                    cv2.arrowedLine(vis, pt1, pt2, (0, 0, 255), 1, tipLength=0.3)
            cv2.imshow('Mesh Grid Motion', vis)
            cv2.waitKey(0)
        if i == 1:
            cv2.destroyAllWindows()
        prev_gray_small = curr_gray_small

    # Vectorized smoothing operation
    # mesh_cumsum: cumulative sum of mesh trajectories (motion vectors)
    mesh_cumsum = np.cumsum(mesh_trajectories, axis=0)
    window = 5  # window size for smoothing
    kernel = np.ones(window) / window  # smoothing kernel
    
    # mesh_cumsum_smooth: smoothed mesh motion using convolution
    mesh_cumsum_smooth = np.zeros_like(mesh_cumsum)
    for d in range(2):
        # Vectorized convolution across all mesh points
        for i in range(mesh_h_small):
            for j in range(mesh_w_small):
                mesh_cumsum_smooth[:, i, j, d] = np.convolve(mesh_cumsum[:, i, j, d], kernel, mode='same')
    # Show smoothed mesh motion for first frame (step 5)
    vis_smooth = np.zeros((h_small, w_small, 3), dtype=np.uint8)
    for y in range(mesh_h_small):
        for x in range(mesh_w_small):
            pt1 = (int(mesh_x_small[y, x]), int(mesh_y_small[y, x]))
            dx, dy = mesh_cumsum_smooth[0, y, x, 0], mesh_cumsum_smooth[0, y, x, 1]
            pt2 = (int(pt1[0] + dx * 5), int(pt1[1] + dy * 5))
            cv2.arrowedLine(vis_smooth, pt1, pt2, (255, 0, 0), 1, tipLength=0.3)
    cv2.imshow('Smoothed Mesh Motion (first frame)', vis_smooth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # correction: difference between smoothed and original mesh motion
    correction = mesh_cumsum_smooth - mesh_cumsum
    # Show correction vectors for first frame (step 6)
    vis_corr = np.zeros((h_small, w_small, 3), dtype=np.uint8)
    for y in range(mesh_h_small):
        for x in range(mesh_w_small):
            pt1 = (int(mesh_x_small[y, x]), int(mesh_y_small[y, x]))
            dx, dy = correction[0, y, x, 0], correction[0, y, x, 1]
            pt2 = (int(pt1[0] + dx * 5), int(pt1[1] + dy * 5))
            cv2.arrowedLine(vis_corr, pt1, pt2, (0, 255, 255), 1, tipLength=0.3)
    cv2.imshow('Correction Vectors (first frame)', vis_corr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Pre-compute coordinate maps for efficiency
    map_x_base = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y_base = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    out.write(frame)
    
    for idx in range(len(correction)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Scale correction back to full resolution and apply
        corr_x = cv2.resize(correction[idx, ..., 0], (w, h), interpolation=cv2.INTER_LINEAR) / scale_factor
        corr_y = cv2.resize(correction[idx, ..., 1], (w, h), interpolation=cv2.INTER_LINEAR) / scale_factor
        
        # Apply correction to coordinate maps
        map_x = map_x_base - corr_x
        map_y = map_y_base - corr_y
        
        # Apply the correction to the frame
        stabilized = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out.write(stabilized)
        
        # Show stabilized frame for first 2 frames (step 7)
        if idx < 2:
            cv2.imshow('Stabilized Frame', stabilized)
            cv2.waitKey(0)
        if idx == 1:
            cv2.destroyAllWindows()
    cap.release()
    out.release()
    print(f"✅ Meshflow stabilization completed and saved as {output_path}")

# ▶ Example usage:
# stabilize_with_optical_flow("your_input.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Video Stabilization Toolbox - Supports MP4, AVI, MOV, MKV, M4V, WMV, FLV, WEBM formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test video compatibility
  python stabalization_v1.py --test --input video.m4v
  
  # Meshflow stabilization (AVI support)
  python stabalization_v1.py --meshflow --input recording.avi --output mesh_stable.mp4
  
  # Compare M4V original with MP4 stabilized (with X/Y analysis)
  python stabalization_v1.py --compare --original source.m4v --output result.mp4
  
  # Compare AVI files with side-by-side video creation
  python stabalization_v1.py --compare --create-video --original shaky.avi --output smooth.avi
  
  # Create only comparison video (no plots)
  python stabalization_v1.py --create-video --original source.m4v --output stabilized.mp4
        """
    )
    parser.add_argument('--meshflow', action='store_true', help='Run meshflow stabilization')
    parser.add_argument('--input', type=str, default='improved_enhanced_video.mp4', 
                       help='Input video (supports: mp4, avi, mov, mkv, m4v, wmv, flv, webm)')
    parser.add_argument('--output', type=str, default='stabilized.mp4', help='Output video')
    args = parser.parse_args()

    if args.meshflow:
        print(f"🔄 Starting meshflow stabilization...")
        start_time = time.time()
        meshflow_stabilize(args.input, args.output)
        elapsed = time.time() - start_time
        print(f"\nEnhancement completed in {elapsed:.2f} seconds.")

