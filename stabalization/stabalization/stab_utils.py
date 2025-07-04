import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor
import sys

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
@jit(nopython=True)
def moving_average_fast(curve, radius=75):
    """Optimized 1D moving average filter using numba."""
    n = len(curve)
    result = np.zeros(n)
    window_size = 2 * radius + 1
    
    # Handle edges by extending
    extended = np.zeros(n + 2 * radius)
    extended[:radius] = curve[0]
    extended[radius:radius+n] = curve
    extended[radius+n:] = curve[-1]
    
    # Compute moving average
    for i in range(n):
        result[i] = np.mean(extended[i:i+window_size])
    
    return result

def moving_average(curve, radius=75):
    """Apply 1D moving average filter."""
    return moving_average_fast(curve, radius)

def fix_border(frame, scale=1.1):
    """Zoom into the image to remove border artifacts."""
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    return cv2.warpAffine(frame, T, (w, h))

def get_optical_flow_jitter_optimized(video_path, scale_factor=0.3, frame_skip=1):
    """Optimized jitter computation with reduced resolution and optional frame skipping."""
    displacement_data = get_optical_flow_displacements(video_path, scale_factor, frame_skip)
    return displacement_data['jitters']

def get_optical_flow_jitter(video_path):
    """Legacy function for backward compatibility."""
    return get_optical_flow_jitter_optimized(video_path)

def process_video_jitter(args):
    """Helper function for parallel processing."""
    video_path, label = args
    print(f"🔄 Processing {label} video jitter...")
    jitter = get_optical_flow_jitter_optimized(video_path, scale_factor=0.3, frame_skip=2)
    print(f"✅ Completed {label} video analysis")
    return jitter

def process_video_displacements(args):
    """Helper function for parallel processing with detailed displacement data."""
    video_path, label = args
    print(f"🔄 Processing {label} video displacement analysis...")
    displacement_data = get_optical_flow_displacements(video_path, scale_factor=0.3, frame_skip=2)
    print(f"✅ Completed {label} video analysis")
    return displacement_data

def get_optical_flow_displacements(video_path, scale_factor=0.3, frame_skip=1):
    """Get detailed X and Y displacement data from optical flow analysis."""
    # Validate video format first
    file_ext = validate_video_format(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # Try alternative approaches for problematic formats
        print(f"⚠️  Initial open failed, trying alternative methods for {file_ext} file...")
        
        # Method 1: Try with different backend
        cap = cv2.VideoCapture()
        success = cap.open(video_path)
        
        # Method 2: Format-specific handling
        if not success and file_ext in ['.m4v', '.avi']:
            print(f"🔄 Applying {file_ext.upper()} specific codec handling...")
            cap.release()
            
            # For M4V files, try with CAP_FFMPEG backend
            if file_ext == '.m4v':
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            # For AVI files, try different approaches
            elif file_ext == '.avi':
                # Try with different backends for AVI
                backends = [cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_MSMF]
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(video_path, backend)
                        if cap.isOpened():
                            print(f"✅ Successfully opened AVI with backend: {backend}")
                            break
                    except:
                        continue
        
        if not cap.isOpened():
            raise Exception(f"❌ Could not open video: {video_path}. Format {file_ext} may not be supported by your OpenCV installation. Try converting to MP4 format.")
    
    # Get video properties with validation
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Validate video properties
    if n_frames <= 0 or w <= 0 or h <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"⚠️  Video properties: frames={n_frames}, width={w}, height={h}, fps={fps}")
        if n_frames <= 0:
            print("⚠️  Frame count detection failed, will process until end of stream")
            n_frames = 10000  # Set a large number as fallback
    
    # Use smaller resolution for faster processing
    w_small, h_small = int(w * scale_factor), int(h * scale_factor)
    
    # Pre-allocate arrays for detailed analysis
    expected_frames = (n_frames - 1) // frame_skip
    jitters = np.zeros(expected_frames, dtype=np.float32)
    x_displacements = np.zeros(expected_frames, dtype=np.float32)
    y_displacements = np.zeros(expected_frames, dtype=np.float32)
    x_std = np.zeros(expected_frames, dtype=np.float32)
    y_std = np.zeros(expected_frames, dtype=np.float32)
    
    ret, prev = cap.read()
    if not ret:
        raise Exception("Could not read first frame.")
    
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray_small = cv2.resize(prev_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    
    # Optimized optical flow parameters for speed
    flow_params = dict(
        pyr_scale=0.5,
        levels=2,  # Reduced levels for speed
        winsize=10,  # Smaller window for speed
        iterations=2,  # Fewer iterations
        poly_n=3,  # Smaller polynomial
        poly_sigma=1.1,
        flags=0
    )
    
    jitter_idx = 0
    frame_count = 0
    
    while jitter_idx < expected_frames:
        # Skip frames if frame_skip > 1
        for _ in range(frame_skip):
            ret, curr = cap.read()
            frame_count += 1
            if not ret:
                break
        
        if not ret:
            break
            
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_gray_small = cv2.resize(curr_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        
        # Compute optical flow on small image
        flow = cv2.calcOpticalFlowFarneback(prev_gray_small, curr_gray_small, None, **flow_params)
        
        # Extract detailed displacement data
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Compute statistics
        mean_jitter, std_jitter = compute_jitter_metric(flow)
        mean_x = np.mean(flow_x)
        mean_y = np.mean(flow_y)
        std_x = np.std(flow_x)
        std_y = np.std(flow_y)
        
        # Store data
        jitters[jitter_idx] = std_jitter
        x_displacements[jitter_idx] = mean_x
        y_displacements[jitter_idx] = mean_y
        x_std[jitter_idx] = std_x
        y_std[jitter_idx] = std_y
        
        prev_gray_small = curr_gray_small
        jitter_idx += 1
    
    cap.release()
    return {
        'jitters': jitters[:jitter_idx],
        'x_displacement': x_displacements[:jitter_idx],
        'y_displacement': y_displacements[:jitter_idx],
        'x_std': x_std[:jitter_idx],
        'y_std': y_std[:jitter_idx]
    }

def compute_jitter_metric(flow):
    """Compute jitter metric from optical flow."""
    if NUMBA_AVAILABLE:
        return compute_jitter_metric_fast(flow[..., 0], flow[..., 1])
    else:
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude), np.std(magnitude)

@jit(nopython=True)
def compute_jitter_metric_fast(flow_x, flow_y):
    """Optimized jitter computation using numba."""
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    return np.mean(magnitude), np.std(magnitude)

def validate_video_format(video_path):
    """Validate and check video format compatibility with enhanced M4V and AVI support."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check file extension with priority formats
    supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv', '.webm']
    priority_formats = ['.mp4', '.avi', '.m4v']  # Formats with enhanced support
    
    file_ext = os.path.splitext(video_path.lower())[1]
    
    if file_ext not in supported_formats:
        print(f"⚠️  Warning: {file_ext} format may not be fully supported. Supported formats: {', '.join(supported_formats)}")
    elif file_ext in priority_formats:
        print(f"✅ Detected optimized format: {file_ext} (enhanced support)")
    else:
        print(f"✅ Detected supported video format: {file_ext}")
    
    # Format-specific recommendations
    if file_ext == '.m4v':
        print("📱 M4V format detected - optimized for Apple/mobile content")
    elif file_ext == '.avi':
        print("🎬 AVI format detected - using enhanced codec compatibility")
    
    return file_ext

def create_comparison_video(original_path, stabilized_path, output_path='comparison_video.mp4'):
    """Create side-by-side comparison video."""
    print("🎬 Creating comparison video...")
    
    # Open both videos
    cap_orig = cv2.VideoCapture(original_path)
    cap_stab = cv2.VideoCapture(stabilized_path)
    
    if not cap_orig.isOpened() or not cap_stab.isOpened():
        raise Exception("❌ Could not open one or both videos for comparison")
    
    # Get properties from original video
    w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    n_frames = min(int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)), 
                   int(cap_stab.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Create output video writer (double width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (2 * w, h))
    
    frame_count = 0
    
    while frame_count < n_frames:
        ret_orig, frame_orig = cap_orig.read()
        ret_stab, frame_stab = cap_stab.read()
        
        if not ret_orig or not ret_stab:
            break
        
        # Resize frames to ensure they match
        frame_orig = cv2.resize(frame_orig, (w, h))
        frame_stab = cv2.resize(frame_stab, (w, h))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame_orig, 'Original', (10, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(frame_stab, 'Stabilized', (10, 30), font, 1, (0, 255, 0), 2)
        
        # Combine frames side by side
        combined = np.hstack((frame_orig, frame_stab))
        out.write(combined)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"🎬 Processing frame {frame_count}/{n_frames}")
    
    # Release everything
    cap_orig.release()
    cap_stab.release()
    out.release()
    
    print(f"✅ Comparison video saved as {output_path}")
    return output_path

def compare_stability_plot(original_path, stabilized_path, save_plot=True, show_plot=True, create_video=False, video_output='comparison_video.mp4'):
    """
    Optimized stability comparison with parallel processing and X/Y displacement analysis.
    
    Supports multiple video formats including: MP4, AVI, MOV, MKV, M4V, WMV, FLV, WEBM
    
    Args:
        original_path (str): Path to original video file
        stabilized_path (str): Path to stabilized video file  
        save_plot (bool): Save plot as PNG file
        show_plot (bool): Display plot window
        create_video (bool): Create side-by-side comparison video
        video_output (str): Output path for comparison video
    
    Returns:
        tuple: (improvement_percentage, original_data_dict, stabilized_data_dict)
    """
    print("🚀 Starting enhanced stability analysis with X/Y displacement tracking...")
    print(f"📁 Original video: {original_path}")
    print(f"📁 Stabilized video: {stabilized_path}")
    
    # Validate both video files before processing
    try:
        validate_video_format(original_path)
        validate_video_format(stabilized_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None, None
    except Exception as e:
        print(f"⚠️  Format validation warning: {e}")
    
    # Process both videos in parallel with detailed displacement analysis
    with ThreadPoolExecutor(max_workers=2) as executor:
        video_args = [(original_path, "original"), (stabilized_path, "stabilized")]
        try:
            results = list(executor.map(process_video_displacements, video_args))
        except Exception as e:
            print(f"❌ Error during parallel processing: {e}")
            print("🔄 Falling back to sequential processing...")
            results = []
            for args in video_args:
                try:
                    result = process_video_displacements(args)
                    results.append(result)
                except Exception as seq_e:
                    print(f"❌ Failed to process {args[1]} video: {seq_e}")
                    return None, None, None
    
    orig_data, stab_data = results
    
    # Extract jitter data for compatibility
    orig = orig_data['jitters']
    stab = stab_data['jitters']
    
    # Ensure all arrays have the same length for comparison
    min_len = min(len(orig), len(stab), 
                  len(orig_data['x_displacement']), len(stab_data['x_displacement']),
                  len(orig_data['y_displacement']), len(stab_data['y_displacement']))
    
    # Trim all arrays to the same length
    orig = orig[:min_len]
    stab = stab[:min_len]
    
    # Trim displacement data arrays
    for key in ['x_displacement', 'y_displacement', 'x_std', 'y_std']:
        orig_data[key] = orig_data[key][:min_len]
        stab_data[key] = stab_data[key][:min_len]
    
    # Vectorized computations
    mean_orig = np.mean(orig)
    mean_stab = np.mean(stab)
    improvement = ((mean_orig - mean_stab) / mean_orig) * 100 if mean_orig > 0 else 0
    
    # Calculate X/Y displacement improvements
    x_orig_mean = np.mean(np.abs(orig_data['x_displacement']))
    x_stab_mean = np.mean(np.abs(stab_data['x_displacement']))
    x_improvement = ((x_orig_mean - x_stab_mean) / x_orig_mean) * 100 if x_orig_mean > 0 else 0
    
    y_orig_mean = np.mean(np.abs(orig_data['y_displacement']))
    y_stab_mean = np.mean(np.abs(stab_data['y_displacement']))
    y_improvement = ((y_orig_mean - y_stab_mean) / y_orig_mean) * 100 if y_orig_mean > 0 else 0
    
    # Create comprehensive plot with X/Y displacement analysis
    plt.figure(figsize=(16, 12))
    frame_indices = np.arange(len(orig))
    
    # Main jitter comparison
    plt.subplot(3, 2, 1)
    plt.plot(frame_indices, orig, label="Original", color='red', alpha=0.7, linewidth=1)
    plt.plot(frame_indices, stab, label="Stabilized", color='green', alpha=0.8, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("Overall Jitter (std dev)")
    plt.title(f"Overall Stability Comparison\nImprovement: {improvement:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # X displacement comparison
    plt.subplot(3, 2, 2)
    plt.plot(frame_indices, orig_data['x_displacement'], label="Original X", color='red', alpha=0.7, linewidth=1)
    plt.plot(frame_indices, stab_data['x_displacement'], label="Stabilized X", color='green', alpha=0.8, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("X Displacement (pixels)")
    plt.title(f"X-Axis Displacement\nImprovement: {x_improvement:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y displacement comparison
    plt.subplot(3, 2, 3)
    plt.plot(frame_indices, orig_data['y_displacement'], label="Original Y", color='red', alpha=0.7, linewidth=1)
    plt.plot(frame_indices, stab_data['y_displacement'], label="Stabilized Y", color='green', alpha=0.8, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("Y Displacement (pixels)")
    plt.title(f"Y-Axis Displacement\nImprovement: {y_improvement:.2f}%")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # X Standard deviation comparison
    plt.subplot(3, 2, 4)
    plt.plot(frame_indices, orig_data['x_std'], label="Original X Std", color='red', alpha=0.7, linewidth=1)
    plt.plot(frame_indices, stab_data['x_std'], label="Stabilized X Std", color='green', alpha=0.8, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("X Displacement Std Dev")
    plt.title("X-Axis Motion Consistency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y Standard deviation comparison
    plt.subplot(3, 2, 5)
    plt.plot(frame_indices, orig_data['y_std'], label="Original Y Std", color='red', alpha=0.7, linewidth=1)
    plt.plot(frame_indices, stab_data['y_std'], label="Stabilized Y Std", color='green', alpha=0.8, linewidth=1)
    plt.xlabel("Frame Index")
    plt.ylabel("Y Displacement Std Dev")
    plt.title("Y-Axis Motion Consistency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined displacement magnitude histogram
    plt.subplot(3, 2, 6)
    orig_magnitude = np.sqrt(orig_data['x_displacement']**2 + orig_data['y_displacement']**2)
    stab_magnitude = np.sqrt(stab_data['x_displacement']**2 + stab_data['y_displacement']**2)
    plt.hist(orig_magnitude, bins=30, alpha=0.6, color='red', label=f'Original (μ={np.mean(orig_magnitude):.3f})', density=True)
    plt.hist(stab_magnitude, bins=30, alpha=0.6, color='green', label=f'Stabilized (μ={np.mean(stab_magnitude):.3f})', density=True)
    plt.xlabel("Displacement Magnitude")
    plt.ylabel("Density")
    plt.title("Overall Displacement Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig("displacement_analysis.png", dpi=150, bbox_inches='tight')
        print("📊 Detailed displacement analysis saved as displacement_analysis.png")
    
    if show_plot:
        plt.show()
    
    # Create comparison video if requested
    if create_video:
        try:
            create_comparison_video(original_path, stabilized_path, video_output)
        except Exception as e:
            print(f"❌ Failed to create comparison video: {e}")
    
    # Print detailed statistics
    print(f"📈 Comprehensive Analysis Results:")
    print(f"   Overall Jitter:")
    print(f"     Original: {mean_orig:.4f} ± {np.std(orig):.4f}")
    print(f"     Stabilized: {mean_stab:.4f} ± {np.std(stab):.4f}")
    print(f"     ✅ Improvement: {improvement:.2f}%")
    print(f"   X-Axis Displacement:")
    print(f"     Original: {x_orig_mean:.4f} ± {np.std(orig_data['x_displacement']):.4f}")
    print(f"     Stabilized: {x_stab_mean:.4f} ± {np.std(stab_data['x_displacement']):.4f}")
    print(f"     ✅ Improvement: {x_improvement:.2f}%")
    print(f"   Y-Axis Displacement:")
    print(f"     Original: {y_orig_mean:.4f} ± {np.std(orig_data['y_displacement']):.4f}")
    print(f"     Stabilized: {y_stab_mean:.4f} ± {np.std(stab_data['y_displacement']):.4f}")
    print(f"     ✅ Improvement: {y_improvement:.2f}%")
    print(f"   📊 Processed {len(orig)} frame pairs for analysis")
    
    return improvement, orig_data, stab_data

def test_video_compatibility(video_path):
    """Test if a video file can be opened and processed for comparison."""
    try:
        print(f"🧪 Testing video compatibility: {os.path.basename(video_path)}")
        file_ext = validate_video_format(video_path)
        
        # Try to open and read first frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Apply enhanced opening logic for M4V and AVI
            if file_ext == '.m4v':
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            elif file_ext == '.avi':
                backends = [cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_MSMF]
                for backend in backends:
                    try:
                        cap = cv2.VideoCapture(video_path, backend)
                        if cap.isOpened():
                            break
                    except:
                        continue
        
        if not cap.isOpened():
            print(f"❌ Cannot open {file_ext} file")
            return False
            
        # Test basic properties
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Cannot read frames from {file_ext} file")
            cap.release()
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Video properties - Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {frame_count}")
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Video Stability Analysis and Comparison Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test video compatibility
  python stab_utils.py --test --input video.m4v

  # Compare stability with X/Y displacement analysis
  python stab_utils.py --compare --original source.m4v --output result.mp4

  # Create only comparison video (no plots)
  python stab_utils.py --create-video --original source.m4v --output stabilized.mp4
        """
    )
    parser.add_argument('--test', action='store_true', help='Test video format compatibility')
    parser.add_argument('--compare', action='store_true', help='Compare stability with X/Y displacement analysis')
    parser.add_argument('--create-video', action='store_true', help='Create side-by-side comparison video')
    parser.add_argument('--input', type=str, default='improved_enhanced_video.mp4', help='Input video (supports: mp4, avi, mov, mkv, m4v, wmv, flv, webm)')
    parser.add_argument('--output', type=str, default='stabilized.mp4', help='Output video')
    parser.add_argument('--original', type=str, help='Original video for comparison (supports all formats including M4V and AVI)')
    parser.add_argument('--video-output', type=str, default='comparison_video.mp4', help='Output path for comparison video')
    args = parser.parse_args()

    if args.test:
        print("🧪 Testing video format compatibility...")
        test_video_compatibility(args.input)
        sys.exit(0)
    if args.compare:
        if args.original is None:
            print('❌ Please provide --original for comparison.')
            print('📱 M4V example: python stab_utils.py --compare --original video.m4v --output stabilized.mp4')
            print('🎬 AVI example: python stab_utils.py --compare --original shaky.avi --output smooth.avi')
            print('🎬 With video: python stab_utils.py --compare --create-video --original video.m4v --output stabilized.mp4')
            sys.exit(1)
        print("🔄 Starting enhanced comparison with X/Y displacement analysis and M4V/AVI support...")
        compare_stability_plot(args.original, args.output, create_video=args.create_video, video_output=args.video_output)
        sys.exit(0)
    if args.create_video:
        if args.original is None:
            print('❌ Please provide --original for video comparison.')
            print('🎬 Example: python stab_utils.py --create-video --original video.m4v --output stabilized.mp4')
            sys.exit(1)
        print("🎬 Creating standalone comparison video...")
        create_comparison_video(args.original, args.output, args.video_output)
        sys.exit(0)

if __name__ == "__main__":
    main()
