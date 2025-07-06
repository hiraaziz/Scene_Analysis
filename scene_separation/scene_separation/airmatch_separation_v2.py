import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

def scene_separation(input_video, output_folder, hist_threshold=0.7, edge_threshold=0.10, 
                    ssim_threshold=0.8, min_scene_length=30, weights=None, show_display=True):
    """
    Advanced scene separation using multiple techniques with weighted scoring.
    
    Compares consecutive frames using three techniques for robust scene cut detection:
    1. Histogram correlation - detects color/lighting changes
    2. Edge energy analysis - detects structural changes  
    3. SSIM (Structural Similarity) - detects overall visual similarity
    
    Args:
        input_video (str): Path to input video file
        output_folder (str): Directory to save separated scene videos
        hist_threshold (float): Histogram correlation threshold (lower = more sensitive)
        edge_threshold (float): Edge energy change threshold (higher = more sensitive)
        ssim_threshold (float): SSIM similarity threshold (lower = more sensitive)
        min_scene_length (int): Minimum frames per scene to avoid micro-cuts
        weights (dict): Weights for combining techniques {'hist': 0.4, 'edge': 0.3, 'ssim': 0.3}
        show_display (bool): Whether to display intermediate results and frames
    """
    # Default weights if not provided
    if weights is None:
        weights = {
            'hist': 0.4,    # Histogram correlation weight
            'edge': 0.3,    # Edge energy change weight  
            'ssim': 0.3     # SSIM similarity weight
        }
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize video capture and get properties
    cap = cv2.VideoCapture(input_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup video writer for first scene
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    scene_count = 1
    writer = cv2.VideoWriter(os.path.join(output_folder, f"scene_{scene_count}.mp4"), fourcc, fps, (width, height))
    
    # Initialize tracking variables
    frame_count = 0
    scene_start_frame = 0
    
    # Variables to store previous frame data for comparison
    prev_hist = None
    prev_edge_energy = None
    prev_gray_resized = None
    
    print(f"🎬 Starting scene separation with weighted approach:")
    print(f"⚖️ Weights - Histogram: {weights['hist']}, Edge: {weights['edge']}, SSIM: {weights['ssim']}")
    print(f"🎯 Thresholds - Hist: {hist_threshold}, Edge: {edge_threshold}, SSIM: {ssim_threshold}")
    print(f"👁️ Display mode: {'ON' if show_display else 'OFF'}")
    print("=" * 80)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\n📷 FRAME {frame_count} ANALYSIS:")
        print("-" * 50)
        
        # === FEATURE EXTRACTION FOR CURRENT FRAME ===
        
        # 1. CONVERT TO GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f"✅ Grayscale conversion: {gray.shape} -> Mean intensity: {np.mean(gray):.2f}")
        
        # 2. EDGE ENERGY CALCULATION
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 1)
        edge_energy = np.mean(edges**2)
        print(f"⚡ Edge energy calculation: {edge_energy:.4f}")
        
        # 3. HISTOGRAM CALCULATION  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [256, 256], [0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        print(f"📊 Histogram calculation: Shape {hist.shape}, Sum: {np.sum(hist):.2f}")
        
        # 4. PREPARE GRAYSCALE FOR SSIM
        gray_resized = cv2.resize(gray, (width//4, height//4))
        print(f"🔍 SSIM grayscale prepared: {gray_resized.shape}")
        
        # === DISPLAY INTERMEDIATE RESULTS ===
        if show_display:
            # Create display images
            display_frame = frame.copy()
            
            # Add text overlay with frame info
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Scene: {scene_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Edge Energy: {edge_energy:.2f}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Convert edges to displayable format
            edges_display = cv2.convertScaleAbs(edges)
            edges_colored = cv2.applyColorMap(edges_display, cv2.COLORMAP_JET)
            
            # Create histogram visualization
            hist_2d = hist.reshape(256, 256)
            hist_normalized = (hist_2d * 255).astype(np.uint8)
            hist_colored = cv2.applyColorMap(hist_normalized, cv2.COLORMAP_HOT)
            
            # Resize images for display
            display_frame_small = cv2.resize(display_frame, (400, 300))
            gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            gray_small = cv2.resize(gray_colored, (400, 300))
            edges_small = cv2.resize(edges_colored, (400, 300))
            hist_small = cv2.resize(hist_colored, (400, 300))
            
            # Create combined display
            top_row = np.hstack([display_frame_small, gray_small])
            bottom_row = np.hstack([edges_small, hist_small])
            combined_display = np.vstack([top_row, bottom_row])
            
            # Add labels
            cv2.putText(combined_display, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_display, "Grayscale", (410, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_display, "Edges", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(combined_display, "Histogram", (410, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow('Scene Separation Analysis', combined_display)
        
        # === SCENE CUT DETECTION ===
        if prev_hist is not None and (frame_count - scene_start_frame) >= min_scene_length:
            
            print(f"🔍 SCENE CUT ANALYSIS:")
            
            # 1. HISTOGRAM CORRELATION ANALYSIS
            hist_correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            print(f"   📊 Histogram correlation: {hist_correlation:.4f}")
            
            # 2. EDGE ENERGY CHANGE ANALYSIS
            edge_change = abs(edge_energy - prev_edge_energy)
            print(f"   ⚡ Edge energy change: {edge_change:.4f}")
            
            # 3. SSIM SIMILARITY ANALYSIS
            ssim_score = ssim(prev_gray_resized, gray_resized)
            print(f"   🔍 SSIM similarity: {ssim_score:.4f}")
            
            # === WEIGHTED SCORING SYSTEM ===
            print(f"📈 SCORING CALCULATION:")
            
            # Histogram score: lower correlation = higher cut probability
            hist_cut_score = max(0, (hist_threshold - hist_correlation) / hist_threshold)
            print(f"   📊 Histogram cut score: {hist_cut_score:.4f} (weight: {weights['hist']})")
            
            # Edge score: higher change = higher cut probability  
            edge_cut_score = min(1, edge_change / edge_threshold)
            print(f"   ⚡ Edge cut score: {edge_cut_score:.4f} (weight: {weights['edge']})")
            
            # SSIM score: lower similarity = higher cut probability
            ssim_cut_score = max(0, (ssim_threshold - ssim_score) / ssim_threshold)
            print(f"   🔍 SSIM cut score: {ssim_cut_score:.4f} (weight: {weights['ssim']})")
            
            # Calculate weighted combined score
            combined_score = (weights['hist'] * hist_cut_score + 
                            weights['edge'] * edge_cut_score + 
                            weights['ssim'] * ssim_cut_score)
            
            print(f"🎯 COMBINED SCORE: {combined_score:.4f}")
            
            # === SCENE CUT DECISION ===
            cut_threshold = 0.5
            print(f"🎬 CUT DECISION: {combined_score:.4f} {'>' if combined_score > cut_threshold else '<='} {cut_threshold}")
            
            if combined_score > cut_threshold:
                print(f"🎬 *** SCENE CUT DETECTED at frame {frame_count} ***")
                print(f"   Combined Score: {combined_score:.4f} > {cut_threshold}")
                print(f"   Details: Hist={hist_correlation:.3f}, Edge={edge_change:.3f}, SSIM={ssim_score:.3f}")
                
                # === FINALIZE CURRENT SCENE ===
                writer.release()
                
                # === START NEW SCENE ===
                scene_count += 1
                writer = cv2.VideoWriter(os.path.join(output_folder, f"scene_{scene_count}.mp4"), 
                                       fourcc, fps, (width, height))
                
                # Reset scene tracking
                scene_start_frame = frame_count
                
                # Add scene cut indicator to display
                if show_display:
                    cut_indicator = np.zeros((100, 800, 3), dtype=np.uint8)
                    cut_indicator[:] = (0, 0, 255)  # Red background
                    cv2.putText(cut_indicator, f"SCENE CUT DETECTED! New Scene: {scene_count}", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    cv2.imshow('Scene Cut Alert', cut_indicator)
            else:
                print(f"✅ No scene cut - continuing current scene {scene_count}")
        else:
            if prev_hist is None:
                print(f"🔄 First frame - initializing data")
            else:
                print(f"⏳ Scene too short ({frame_count - scene_start_frame} < {min_scene_length}) - skipping analysis")
        
        # === WRITE CURRENT FRAME ===
        writer.write(frame)
        
        # === UPDATE PREVIOUS FRAME DATA ===
        prev_hist = hist.copy()
        prev_edge_energy = edge_energy
        prev_gray_resized = gray_resized.copy()
        
        frame_count += 1
        
    # === CLEANUP ===
    cap.release()
    writer.release()
    if show_display:
        cv2.destroyAllWindows()
    
    print(f"\n" + "=" * 80)
    print(f"✅ Scene separation complete!")
    print(f"📹 Total scenes created: {scene_count}")
    print(f"🎞️ Frames processed: {frame_count}")
    print(f"📁 Output folder: {output_folder}")

if __name__ == "__main__":
    # Example usage with custom parameters
    scene_separation(
        input_video="enhanced_13.mp4", 
        output_folder="separated_scenes",
        hist_threshold=0.7,    # Lower = more sensitive to color changes
        edge_threshold=0.10,   # Higher = more sensitive to structural changes  
        ssim_threshold=0.8,    # Lower = more sensitive to overall visual changes
        min_scene_length=30,   # Minimum frames per scene
        weights={'hist': 0.5, 'edge': 0.3, 'ssim': 0.2},  # Adjust technique importance
        show_display=True      # Enable visual display
    ) 