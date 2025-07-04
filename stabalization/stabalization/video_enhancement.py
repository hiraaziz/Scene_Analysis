import cv2
import os
import numpy as np

def video_enhancement(input_video, output_video, clip_limit=3.0, tile_grid_size=(3, 3)):
    """
    Enhances video quality using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        input_video (str): Path to the input video file
        output_video (str): Path to save the enhanced video
        clip_limit (float): Threshold for contrast limiting (default: 3.0)
        tile_grid_size (tuple): Size of grid for histogram equalization (default: (3, 3))
    
    Returns:
        str: Path to the enhanced video file
    """
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
    
    # Create video writer for enhanced output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Process all frames
    frame_count = 0
    
    print("Enhancing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert RGB image to Lab format as specified in the algorithm
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split the Lab image into L, a, and b channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel (luminance)
        # This enhances the contrast while preserving the color information
        l_enhanced = clahe.apply(l)
        
        # Merge the enhanced L channel with the original a and b channels
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        
        # Convert back to RGB
        enhanced_frame = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Write the enhanced frame
        out.write(enhanced_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Enhanced {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Video enhancement completed. Output saved to {output_video}")
    return output_video

if __name__ == "__main__":
    # Example usage
    video_enhancement("input/test11.mp4", "output/enhanced_scene_1.mp4") 