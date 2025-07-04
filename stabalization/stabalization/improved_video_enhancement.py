import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def analyze_single_frame(frame):
    """
    Analyze a single frame for brightness, contrast, and sharpness.
    
    Args:
        frame: Input frame (BGR format)
    
    Returns:
        tuple: (brightness)
    """
    # Convert to LAB for brightness and contrast analysis
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    
    brightness = np.mean(l)
    
    return brightness

def determine_enhancement_strategy(brightness):
    """
    Determine enhancement strategy for CLAHE and sharpness based on brightness only.
    If frame is bright, CLAHE=2.0, else CLAHE=1.0. Always apply sharpness.
    """
    BRIGHT = 110
    clahe_strength = 1.0
    if brightness > 110 and brightness <= 170:
        clahe_strength = 1.5
    elif brightness > 170:
        clahe_strength = 2.0
    strategy = {
        'apply_clahe': True,
        'clahe_strength': clahe_strength,
        'apply_unsharp': True,
        'unsharp_strength': 'medium',
        'reason': f"{'Bright' if brightness > BRIGHT else 'Not bright'} frame (brightness: {brightness:.1f})"
    }
    return strategy

def apply_dynamic_clahe(frame, strength=1.0, tile_grid_size=(3, 3)):
    """
    Apply CLAHE with dynamic strength (clipLimit) based on frame analysis.
    """
    clahe = cv2.createCLAHE(clipLimit=float(strength), tileGridSize=tile_grid_size)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def apply_unsharp_mask(frame, strength='medium'):
    """
    Apply unsharp mask with dynamic strength based on frame analysis.
    Args:
        frame: Input frame
        strength: 'low', 'medium', or 'high'
    Returns:
        Sharpened frame
    """
    params = {
        'low': {'kernel_size': (3, 3), 'sigma': 1.0, 'amount': 1.2},
        'medium': {'kernel_size': (5, 5), 'sigma': 1.0, 'amount': 1.5},
        'high': {'kernel_size': (7, 7), 'sigma': 1.2, 'amount': 2.0}
    }
    param = params.get(strength, params['medium'])
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(frame, param['kernel_size'], param['sigma'])
    # Create the unsharp mask
    sharpened = cv2.addWeighted(frame, param['amount'], blurred, -(param['amount']-1), 0)
    # Ensure values are in valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def video_enhancement(input_video, output_video):
    """
    Improved video quality enhancement using CLAHE and sharpness only.
    Now uses dynamic frame analysis every 5 frames to adapt enhancement strategies.
    """
    tile_grid_size=(3, 3)
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {width}x{height} at {fps} fps, {total_frames} frames")
    print("Using dynamic frame analysis every 5 frames for adaptive enhancement")

    # Create video writer for enhanced output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Dynamic enhancement variables
    frame_count = 0
    analysis_interval = 5  # Analyze every 5 frames
    current_strategy = None
    frames_since_analysis = 0
    strategy_changes = 0
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Ensure at start
    
    print("Enhancing video frames with dynamic analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply bilateral filter first for noise reduction
        denoised_frame = cv2.bilateralFilter(frame, 5, 50, 50)
        enhanced_frame = denoised_frame.copy()
        
        # Analyze frame every 5 frames or on first frame
        if frame_count % analysis_interval == 0:
            brightness = analyze_single_frame(frame)
            new_strategy = determine_enhancement_strategy(brightness)
            
            # Check if strategy changed
            if current_strategy is None or new_strategy != current_strategy:
                strategy_changes += 1
                current_strategy = new_strategy
                
                # Print strategy change
                print(f"[Frame {frame_count:4d}] Strategy: CLAHE={current_strategy['clahe_strength']}, Unsharp=always - {current_strategy['reason']}")
            
            frames_since_analysis = 0
        
        # Apply current enhancement strategy
        if current_strategy['apply_clahe']:
            enhanced_frame = apply_dynamic_clahe(enhanced_frame, 
                                               strength=current_strategy['clahe_strength'],
                                               tile_grid_size=tile_grid_size)
        if current_strategy['apply_unsharp']:
            enhanced_frame = apply_unsharp_mask(enhanced_frame, strength=current_strategy['unsharp_strength'])
        
        # Write the enhanced frame
        out.write(enhanced_frame)
        
        frame_count += 1
        frames_since_analysis += 1
        
        # Progress reporting
        if frame_count % 100 == 0:
            print(f"Enhanced {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) - "
                  f"Strategy changes: {strategy_changes}")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Dynamic video enhancement completed. Output saved to {output_video}")
    print(f"Total strategy changes: {strategy_changes} (analyzed every {analysis_interval} frames)")
    return output_video

if __name__ == "__main__":
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Improved video enhancement with unsharp masking and noise reduction")
    parser.add_argument("--input", type=str, default="test11.mp4", help="Input video file path")
    parser.add_argument("--output", type=str, default="improved_enhanced_video.mp4", help="Output video file path")
    
    args = parser.parse_args()
    
    start_time = time.time()
    enhanced_video = video_enhancement(args.input, args.output)
    elapsed = time.time() - start_time
    print(f"\nEnhancement completed in {elapsed:.2f} seconds.")
    