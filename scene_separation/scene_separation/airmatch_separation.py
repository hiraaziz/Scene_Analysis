import cv2
import os
import numpy as np

def scene_separation(input_video, output_folder, threshold=0.4, min_scene_length=0):
    """Improved scene separation with more robust thresholds and minimum scene length."""
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    scene_count = 1
    writer = cv2.VideoWriter(os.path.join(output_folder, f"scene_{scene_count}.mp4"), fourcc, fps, (width, height))
    
    prev_hist = None
    prev_edge_energy = None
    low_correlation_count = 0
    consecutive_frames_to_confirm = 15
    frame_count = 0
    scene_start_frame = 0
    edge_threshold = 0.15
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        edge_energy = np.mean(edges**2)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            edge_change = abs(edge_energy - prev_edge_energy) if prev_edge_energy is not None else 0
            
            # Scene cut logic: require EITHER low correlation OR significant edge change, and min scene length
            if ((correlation < threshold or edge_change > edge_threshold) and (frame_count - scene_start_frame) >= min_scene_length):
                low_correlation_count += 1
                if low_correlation_count >= consecutive_frames_to_confirm:
                    print(f"Scene cut at frame {frame_count} (Corr: {correlation:.2f}, Edge Δ: {edge_change:.2f})")
                    writer.release()
                    scene_count += 1
                    writer = cv2.VideoWriter(os.path.join(output_folder, f"scene_{scene_count}.mp4"), fourcc, fps, (width, height))
                    low_correlation_count = 0
                    scene_start_frame = frame_count
            else:
                low_correlation_count = 0
        
        writer.write(frame)
        prev_hist = hist
        prev_edge_energy = edge_energy
        frame_count += 1
    
    cap.release()
    writer.release()
    print(f"Separated into {scene_count} scenes.")

if __name__ == "__main__":
    # Example usage
    scene_separation("test11.mp4", "separated_scenes") 