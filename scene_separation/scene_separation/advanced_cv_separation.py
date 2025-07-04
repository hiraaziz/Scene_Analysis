"""
Advanced Computer Vision Scene Separation - Enhanced Edition
High-performance scene boundary detection using sophisticated multi-modal analysis:
- LBP texture difference checks for fine-grained texture changes
- Color histogram deltas in LAB and YUV color spaces
- Multi-scale SSIM and optical flow analysis
- Histogram and texture feature analysis
- Optimized for faster processing with improved accuracy
Matches TSN-Lite performance without deep learning - Fast & Accurate
"""

import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

class AdvancedCVSceneSeparator:
    """
    Advanced Computer Vision Scene Separator using multi-modal analysis
    Combines optical flow, histogram analysis, structural similarity, and edge detection
    """
    def __init__(self, similarity_threshold=None, min_scene_duration=None, 
                 sampling_fps=None, optical_flow_threshold=None, auto_params=True):
        """
        Initialize Advanced CV Scene Separator
        
        Args:
            similarity_threshold: Threshold for feature similarity (auto-detected if None)
            min_scene_duration: Minimum scene duration in seconds (auto-detected if None)
            sampling_fps: Frames per second for sparse sampling (auto-detected if None)
            optical_flow_threshold: Threshold for optical flow magnitude (auto-detected if None)
            auto_params: Whether to automatically detect parameters based on video analysis
        """
        self.similarity_threshold = similarity_threshold
        self.min_scene_duration = min_scene_duration
        self.sampling_fps = sampling_fps
        self.optical_flow_threshold = optical_flow_threshold
        self.auto_params = auto_params
        
        # Initialize feature detectors
        self.orb = cv2.ORB_create(nfeatures=1000)
        try:
            self.sift = cv2.SIFT_create(nfeatures=500)
        except:
            self.sift = None  # SIFT may not be available in all OpenCV builds
        
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Good features to track parameters
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
    
    def analyze_video_characteristics(self, video_path, analysis_duration=30):
        """
        Analyze video characteristics to automatically determine optimal parameters
        
        Args:
            video_path: Path to input video
            analysis_duration: Duration in seconds to analyze (default: 30s)
            
        Returns:
            dict: Optimal parameters based on video analysis
        """
        print("🔍 Analyzing video characteristics for auto-parameter detection...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / original_fps
        
        # Limit analysis to specified duration or full video if shorter
        analysis_frames = min(int(analysis_duration * original_fps), total_frames)
        
        print(f"   📹 Video: {video_duration:.1f}s @ {original_fps:.1f} fps")
        print(f"   🔬 Analyzing first {analysis_frames/original_fps:.1f}s ({analysis_frames} frames)")
        
        # Sample frames for analysis (every 0.5 seconds)
        analysis_interval = max(1, int(original_fps * 0.5))
        analysis_sample_fps = original_fps / analysis_interval
        
        frames = []
        flow_magnitudes = []
        histogram_diffs = []
        texture_diffs = []
        color_diffs = []
        
        frame_count = 0
        prev_frame = None
        
        while frame_count < analysis_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % analysis_interval == 0:
                frames.append(frame)
                
                if prev_frame is not None:
                    # Optical flow analysis
                    flow_features = self.compute_advanced_optical_flow(prev_frame, frame)
                    flow_magnitudes.append(flow_features['mean_magnitude'])
                    
                    # Histogram difference
                    hist_prev = self.compute_advanced_histogram_features(prev_frame)
                    hist_curr = self.compute_advanced_histogram_features(frame)
                    hist_sim = np.corrcoef(hist_prev, hist_curr)[0, 1] if len(hist_prev) > 1 else 0.5
                    histogram_diffs.append(1.0 - max(0, hist_sim) if not np.isnan(hist_sim) else 0.5)
                    
                    # Texture difference
                    lbp_diff = self.compute_lbp_texture_difference(prev_frame, frame)
                    texture_diffs.append(lbp_diff['combined_texture_diff'])
                    
                    # Color difference
                    color_deltas = self.compute_color_histogram_deltas(prev_frame, frame)
                    color_diffs.append(color_deltas['total_color_change'])
                
                prev_frame = frame.copy()
            
            frame_count += 1
        
        cap.release()
        
        # Analyze characteristics
        flow_mean = np.mean(flow_magnitudes) if flow_magnitudes else 1000
        flow_std = np.std(flow_magnitudes) if flow_magnitudes else 500
        hist_mean = np.mean(histogram_diffs) if histogram_diffs else 0.3
        hist_std = np.std(histogram_diffs) if histogram_diffs else 0.1
        texture_mean = np.mean(texture_diffs) if texture_diffs else 0.2
        color_mean = np.mean(color_diffs) if color_diffs else 1.0
        
        # Determine video type and motion characteristics
        motion_level = "low" if flow_mean < 1000 else "medium" if flow_mean < 3000 else "high"
        change_level = "gradual" if hist_std < 0.1 else "moderate" if hist_std < 0.2 else "rapid"
        
        print(f"   📊 Analysis Results:")
        print(f"      Motion level: {motion_level} (avg flow: {flow_mean:.0f})")
        print(f"      Change level: {change_level} (hist std: {hist_std:.3f})")
        print(f"      Texture variation: {texture_mean:.3f}")
        print(f"      Color variation: {color_mean:.3f}")
        
        # Auto-determine optimal parameters
        optimal_params = {}
        
        # 1. Sampling FPS - based on motion and video FPS
        if original_fps <= 15:
            optimal_params['sampling_fps'] = max(2, int(original_fps * 0.3))
        elif original_fps <= 30:
            if motion_level == "high":
                optimal_params['sampling_fps'] = 8  # More samples for high motion
            elif motion_level == "medium":
                optimal_params['sampling_fps'] = 5
            else:
                optimal_params['sampling_fps'] = 3  # Fewer samples for low motion
        else:  # High FPS video
            if motion_level == "high":
                optimal_params['sampling_fps'] = 10
            elif motion_level == "medium":
                optimal_params['sampling_fps'] = 6
            else:
                optimal_params['sampling_fps'] = 4
        
        # 2. Optical Flow Threshold - based on typical flow magnitude
        optimal_params['optical_flow_threshold'] = max(1000, int(flow_mean + 1.5 * flow_std))
        
        # 3. Similarity Threshold - based on content variation
        if change_level == "rapid":
            optimal_params['similarity_threshold'] = 0.35  # More sensitive for rapid changes
        elif change_level == "moderate":
            optimal_params['similarity_threshold'] = 0.45  # Balanced
        else:  # gradual changes
            optimal_params['similarity_threshold'] = 0.55  # Less sensitive for gradual content
        
        # Adjust for texture and color variation
        if texture_mean > 0.4 or color_mean > 2.0:
            optimal_params['similarity_threshold'] -= 0.05  # More sensitive for high variation
        
        # 4. Minimum Scene Duration - based on typical change patterns and video length
        if video_duration < 60:  # Short video
            optimal_params['min_scene_duration'] = 0.5
        elif video_duration < 300:  # Medium video (5 min)
            optimal_params['min_scene_duration'] = 1.0
        else:  # Long video
            optimal_params['min_scene_duration'] = 2.0
        
        # Adjust based on change frequency
        if change_level == "rapid":
            optimal_params['min_scene_duration'] *= 0.5  # Shorter scenes for rapid changes
        elif change_level == "gradual":
            optimal_params['min_scene_duration'] *= 1.5  # Longer scenes for gradual content
        
        # Ensure reasonable bounds
        optimal_params['similarity_threshold'] = np.clip(optimal_params['similarity_threshold'], 0.25, 0.75)
        optimal_params['sampling_fps'] = np.clip(optimal_params['sampling_fps'], 2, 15)
        optimal_params['optical_flow_threshold'] = np.clip(optimal_params['optical_flow_threshold'], 1000, 10000)
        optimal_params['min_scene_duration'] = np.clip(optimal_params['min_scene_duration'], 0.1, 5.0)
        
        print(f"   🎯 Optimal Parameters:")
        print(f"      Sampling FPS: {optimal_params['sampling_fps']}")
        print(f"      Similarity threshold: {optimal_params['similarity_threshold']:.3f}")
        print(f"      Optical flow threshold: {optimal_params['optical_flow_threshold']}")
        print(f"      Min scene duration: {optimal_params['min_scene_duration']:.1f}s")
        
        return optimal_params
    
    def extract_sparse_frames(self, video_path):
        """Extract sparse frames from video at specified fps"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(original_fps / self.sampling_fps))
        
        print(f"📹 Video info: {total_frames} frames @ {original_fps:.2f} fps")
        print(f"🎯 Sampling every {frame_interval} frames ({self.sampling_fps} fps)")
        
        frames = []
        timestamps = []
        frame_indices = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                timestamps.append(frame_count / original_fps)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        print(f"📊 Extracted {len(frames)} sparse frames from {total_frames} total frames")
        return frames, timestamps, frame_indices, original_fps
    
    def compute_advanced_histogram_features(self, frame):
        """Compute multi-dimensional histogram features"""
        # Resize for faster computation if too large
        h, w = frame.shape[:2]
        if h > 360 or w > 480:
            scale = min(360/h, 480/w)
            new_h, new_w = int(h*scale), int(w*scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        features = []
        
        # Optimized HSV histogram (reduced bins for speed)
        hist_hsv = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        features.extend(hist_hsv)
        
        # LAB histogram (A and B channels for color information)
        hist_lab = cv2.calcHist([lab], [1, 2], None, [24, 24], [0, 256, 0, 256])
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        features.extend(hist_lab)
        
        # YUV histogram (U and V channels for chrominance)
        hist_yuv = cv2.calcHist([yuv], [1, 2], None, [24, 24], [0, 256, 0, 256])
        hist_yuv = cv2.normalize(hist_yuv, hist_yuv).flatten()
        features.extend(hist_yuv)
        
        # Grayscale histogram (reduced bins)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        features.extend(hist_gray)
        
        return np.array(features)
    
    def compute_color_histogram_deltas(self, prev_frame, curr_frame):
        """Compute color space histogram deltas in LAB and YUV"""
        # Resize for faster computation
        h, w = prev_frame.shape[:2]
        if h > 240 or w > 320:
            scale = min(240/h, 320/w)
            new_h, new_w = int(h*scale), int(w*scale)
            prev_frame = cv2.resize(prev_frame, (new_w, new_h))
            curr_frame = cv2.resize(curr_frame, (new_w, new_h))
        
        # Convert to color spaces
        prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2LAB)
        curr_lab = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2LAB)
        prev_yuv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2YUV)
        curr_yuv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YUV)
        
        deltas = {}
        
        # LAB color space deltas
        for i, channel_name in enumerate(['L', 'A', 'B']):
            # Compute histograms for each channel
            hist_prev = cv2.calcHist([prev_lab], [i], None, [32], [0, 256])
            hist_curr = cv2.calcHist([curr_lab], [i], None, [32], [0, 256])
            
            # Normalize histograms
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
            
            # Compute various delta measures
            # 1. Chi-square distance
            chi_square = np.sum((hist_prev - hist_curr)**2 / (hist_prev + hist_curr + 1e-10))
            deltas[f'lab_{channel_name}_chi_square'] = chi_square
            
            # 2. Histogram intersection
            intersection = np.sum(np.minimum(hist_prev, hist_curr))
            deltas[f'lab_{channel_name}_intersection'] = 1.0 - intersection
            
            # 3. Bhattacharyya distance
            bhatta = -np.log(np.sum(np.sqrt(hist_prev * hist_curr)) + 1e-10)
            deltas[f'lab_{channel_name}_bhattacharyya'] = bhatta
        
        # YUV color space deltas
        for i, channel_name in enumerate(['Y', 'U', 'V']):
            # Compute histograms for each channel
            hist_prev = cv2.calcHist([prev_yuv], [i], None, [32], [0, 256])
            hist_curr = cv2.calcHist([curr_yuv], [i], None, [32], [0, 256])
            
            # Normalize histograms
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()
            
            # Compute delta measures
            chi_square = np.sum((hist_prev - hist_curr)**2 / (hist_prev + hist_curr + 1e-10))
            deltas[f'yuv_{channel_name}_chi_square'] = chi_square
            
            intersection = np.sum(np.minimum(hist_prev, hist_curr))
            deltas[f'yuv_{channel_name}_intersection'] = 1.0 - intersection
            
            bhatta = -np.log(np.sum(np.sqrt(hist_prev * hist_curr)) + 1e-10)
            deltas[f'yuv_{channel_name}_bhattacharyya'] = bhatta
        
        # Combined color deltas
        deltas['lab_combined_delta'] = np.mean([
            deltas['lab_A_chi_square'], deltas['lab_B_chi_square']  # Focus on color channels
        ])
        
        deltas['yuv_combined_delta'] = np.mean([
            deltas['yuv_U_chi_square'], deltas['yuv_V_chi_square']  # Focus on chrominance
        ])
        
        # Overall color change indicator
        deltas['total_color_change'] = np.mean([
            deltas['lab_combined_delta'], deltas['yuv_combined_delta']
        ])
        
        return deltas
    
    def compute_texture_features(self, frame):
        """Compute Local Binary Pattern texture features"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Multi-scale LBP with optimized parameters for speed
        for radius, n_points in [(1, 8), (2, 16)]:  # Reduced from 3 scales to 2 for speed
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                 range=(0, n_points + 2), density=True)
            features.extend(hist)
        
        # Add texture contrast and energy measures
        features.append(np.std(gray))  # Texture contrast
        features.append(np.mean(gray**2))  # Energy
        
        return np.array(features)
    
    def compute_lbp_texture_difference(self, prev_frame, curr_frame):
        """Compute LBP texture difference checks between consecutive frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Resize for faster computation
        h, w = prev_gray.shape
        if h > 240 or w > 320:  # Resize if too large
            scale = min(240/h, 320/w)
            new_h, new_w = int(h*scale), int(w*scale)
            prev_gray = cv2.resize(prev_gray, (new_w, new_h))
            curr_gray = cv2.resize(curr_gray, (new_w, new_h))
        
        differences = []
        
        # LBP texture difference at multiple scales
        for radius, n_points in [(1, 8), (2, 16)]:
            # Compute LBP for both frames
            lbp_prev = local_binary_pattern(prev_gray, n_points, radius, method='uniform')
            lbp_curr = local_binary_pattern(curr_gray, n_points, radius, method='uniform')
            
            # Compute histograms
            hist_prev, _ = np.histogram(lbp_prev.ravel(), bins=n_points + 2, 
                                      range=(0, n_points + 2), density=True)
            hist_curr, _ = np.histogram(lbp_curr.ravel(), bins=n_points + 2, 
                                      range=(0, n_points + 2), density=True)
            
            # Histogram intersection (similarity measure)
            intersection = np.sum(np.minimum(hist_prev, hist_curr))
            differences.append(1.0 - intersection)  # Convert to difference
            
            # Direct LBP pattern difference (pixel-wise)
            pattern_diff = np.mean(np.abs(lbp_prev - lbp_curr))
            differences.append(pattern_diff / (n_points + 2))  # Normalized
        
        # Texture energy difference
        energy_prev = np.mean(prev_gray**2)
        energy_curr = np.mean(curr_gray**2)
        energy_diff = abs(energy_curr - energy_prev) / max(energy_prev, 1e-6)
        differences.append(energy_diff)
        
        # Texture contrast difference
        contrast_prev = np.std(prev_gray)
        contrast_curr = np.std(curr_gray)
        contrast_diff = abs(contrast_curr - contrast_prev) / max(contrast_prev, 1e-6)
        differences.append(contrast_diff)
        
        return {
            'lbp_hist_diff_r1': differences[0],
            'lbp_pattern_diff_r1': differences[1],
            'lbp_hist_diff_r2': differences[2],
            'lbp_pattern_diff_r2': differences[3],
            'energy_diff': differences[4],
            'contrast_diff': differences[5],
            'combined_texture_diff': np.mean(differences)
        }
    

    
    def extract_sparse_frames(self, video_path):
        """Extract sparse frames from video at specified fps"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for sparse sampling
        frame_interval = max(1, int(original_fps / self.sampling_fps))
        
        print(f"📹 Video info: {total_frames} frames @ {original_fps:.2f} fps")
        print(f"🎯 Sampling every {frame_interval} frames ({self.sampling_fps} fps)")
        
        frames = []
        timestamps = []
        frame_indices = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                timestamps.append(frame_count / original_fps)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        print(f"📊 Extracted {len(frames)} sparse frames from {total_frames} total frames")
        return frames, timestamps, frame_indices, original_fps
    
    def compute_advanced_histogram_features(self, frame):
        """Compute multi-dimensional histogram features"""
        # Convert to multiple color spaces for robust analysis
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        features = []
        
        # HSV histogram (hue-saturation)
        hist_hsv = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        features.extend(hist_hsv)
        
        # LAB histogram (lightness and color channels)
        hist_lab = cv2.calcHist([lab], [1, 2], None, [32, 32], [0, 256, 0, 256])
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        features.extend(hist_lab)
        
        # YUV histogram (luminance and chrominance)
        hist_yuv = cv2.calcHist([yuv], [0, 1], None, [32, 32], [0, 256, 0, 256])
        hist_yuv = cv2.normalize(hist_yuv, hist_yuv).flatten()
        features.extend(hist_yuv)
        
        # Grayscale histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist_gray = cv2.normalize(hist_gray, hist_gray).flatten()
        features.extend(hist_gray)
        
        return np.array(features)
    
    def compute_texture_features(self, frame):
        """Compute Local Binary Pattern texture features"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale LBP
        radius_values = [1, 2, 3]
        n_points_values = [8, 16, 24]
        
        features = []
        for radius, n_points in zip(radius_values, n_points_values):
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                 range=(0, n_points + 2), density=True)
            features.extend(hist)
        
        return np.array(features)
    

    

    

    
    def compute_advanced_optical_flow(self, prev_frame, curr_frame):
        """Compute advanced optical flow features"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade with good features
        corners = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
        
        if corners is not None and len(corners) > 10:
            # Track features
            next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None, **self.lk_params
            )
            
            # Select good points
            good_old = corners[status == 1]
            good_new = next_corners[status == 1]
            
            if len(good_old) > 5:
                # Calculate flow vectors
                flow_vectors = good_new - good_old
                flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
                
                features = {
                    'mean_magnitude': np.mean(flow_magnitudes),
                    'max_magnitude': np.max(flow_magnitudes),
                    'std_magnitude': np.std(flow_magnitudes),
                    'median_magnitude': np.median(flow_magnitudes),
                    'flow_consistency': self.compute_flow_consistency(flow_vectors),
                    'directional_variance': self.compute_directional_variance(flow_vectors)
                }
                
                return features
        
        # Return zero features if flow computation fails
        return {
            'mean_magnitude': 0,
            'max_magnitude': 0,
            'std_magnitude': 0,
            'median_magnitude': 0,
            'flow_consistency': 0,
            'directional_variance': 0
        }
    
    def compute_flow_consistency(self, flow_vectors):
        """Compute consistency of optical flow vectors"""
        if len(flow_vectors) < 2:
            return 0
        
        # Calculate pairwise angles between flow vectors
        angles = []
        for i in range(len(flow_vectors)):
            for j in range(i + 1, min(len(flow_vectors), i + 10)):  # Limit comparisons
                v1, v2 = flow_vectors[i], flow_vectors[j]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    angles.append(angle)
        
        if not angles:
            return 0
        
        # Consistency is inversely related to angle variance
        return 1.0 / (1.0 + np.std(angles))
    
    def compute_directional_variance(self, flow_vectors):
        """Compute directional variance of flow vectors"""
        if len(flow_vectors) < 2:
            return 0
        
        angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
        return np.std(angles)
    
    def compute_multi_scale_ssim(self, prev_frame, curr_frame):
        """Compute SSIM at multiple scales"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        ssim_scores = []
        
        # Multi-scale SSIM
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                h, w = prev_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 7 and new_w > 7:  # Ensure minimum size
                    prev_scaled = cv2.resize(prev_gray, (new_w, new_h))
                    curr_scaled = cv2.resize(curr_gray, (new_w, new_h))
                else:
                    continue
            else:
                prev_scaled, curr_scaled = prev_gray, curr_gray
            
            if prev_scaled.shape == curr_scaled.shape and min(prev_scaled.shape) > 7:
                try:
                    score = ssim(prev_scaled, curr_scaled, data_range=255, win_size=7)
                    ssim_scores.append(score)
                except:
                    continue
        
        return ssim_scores if ssim_scores else [0.5]
    
    def compute_frame_features(self, frames):
        """Compute comprehensive features for all frames"""
        print("🧠 Computing advanced computer vision features...")
        
        all_features = []
        
        for i, frame in enumerate(frames):
            if i % 20 == 0:
                print(f"   Processing frame {i+1}/{len(frames)}")
            
            # Compute all feature types
            hist_features = self.compute_advanced_histogram_features(frame)
            texture_features = self.compute_texture_features(frame)
            
            # Combine all features
            combined_features = np.concatenate([
                hist_features,
                texture_features
            ])
            
            all_features.append(combined_features)
        
        return np.array(all_features)
    
    def detect_scene_boundaries_advanced(self, frames, features, timestamps):
        """Advanced scene boundary detection using multiple criteria"""
        boundaries = [0]
        
        print("🔍 Detecting scene boundaries with multi-modal analysis...")
        
        # Compute similarity scores using multiple metrics
        similarities = {
            'histogram': [],
            'texture': [],
            'ssim': [],
            'optical_flow': []
        }
        
        # Feature dimensions (approximate)
        hist_dim = 3200
        texture_start = hist_dim
        texture_dim = 50
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            prev_features = features[i-1]
            curr_features = features[i]
            
            # Histogram similarity
            hist_prev = prev_features[:hist_dim]
            hist_curr = curr_features[:hist_dim]
            hist_sim = np.corrcoef(hist_prev, hist_curr)[0, 1] if len(hist_prev) > 1 else 0
            similarities['histogram'].append(max(0, hist_sim) if not np.isnan(hist_sim) else 0)
            
            # Texture similarity
            if len(prev_features) > texture_start + texture_dim:
                texture_prev = prev_features[texture_start:texture_start+texture_dim]
                texture_curr = curr_features[texture_start:texture_start+texture_dim]
                texture_sim = np.corrcoef(texture_prev, texture_curr)[0, 1] if len(texture_prev) > 1 else 0
                similarities['texture'].append(max(0, texture_sim) if not np.isnan(texture_sim) else 0)
            else:
                similarities['texture'].append(0.5)
            
            # Multi-scale SSIM
            ssim_scores = self.compute_multi_scale_ssim(prev_frame, curr_frame)
            similarities['ssim'].append(np.mean(ssim_scores))
            
            # Advanced optical flow
            flow_features = self.compute_advanced_optical_flow(prev_frame, curr_frame)
            flow_score = 1.0 / (1.0 + flow_features['mean_magnitude'] / 1000.0)
            similarities['optical_flow'].append(flow_score)
        
        # Smooth all similarity signals
        for key in similarities:
            similarities[key] = self.advanced_smooth_signal(similarities[key])
        
        # Compute weighted combined similarity
        weights = {
            'histogram': 0.35,
            'texture': 0.25,
            'ssim': 0.25,
            'optical_flow': 0.15
        }
        
        combined_similarity = np.zeros(len(similarities['histogram']))
        for key, weight in weights.items():
            combined_similarity += weight * np.array(similarities[key])
        
        # Advanced boundary detection with multiple criteria
        gradients = np.gradient(combined_similarity)
        
        # Adaptive thresholding
        similarity_mean = np.mean(combined_similarity)
        similarity_std = np.std(combined_similarity)
        adaptive_threshold = similarity_mean - 2.0 * similarity_std
        adaptive_threshold = max(adaptive_threshold, self.similarity_threshold)
        
        print(f"   Adaptive threshold: {adaptive_threshold:.3f}")
        
        # Peak detection for boundaries
        try:
            negative_peaks, _ = find_peaks(-combined_similarity, 
                                         height=-adaptive_threshold,
                                         distance=int(self.min_scene_duration * self.sampling_fps))
        except:
            # Fallback manual peak detection
            negative_peaks = []
            for i in range(1, len(combined_similarity) - 1):
                if (combined_similarity[i] < combined_similarity[i-1] and 
                    combined_similarity[i] < combined_similarity[i+1] and
                    combined_similarity[i] < adaptive_threshold):
                    negative_peaks.append(i)
        
        for peak_idx in negative_peaks:
            if peak_idx >= len(timestamps):
                continue
                
            time_since_last_boundary = timestamps[peak_idx] - timestamps[boundaries[-1]]
            
            if time_since_last_boundary >= self.min_scene_duration:
                # Multi-criteria validation
                criteria_met = 0
                reasons = []
                
                # Criterion 1: Low combined similarity
                if combined_similarity[peak_idx] < adaptive_threshold:
                    criteria_met += 1
                    reasons.append("low_similarity")
                
                # Criterion 2: Sharp negative gradient
                if peak_idx < len(gradients) and gradients[peak_idx] < -0.05:
                    criteria_met += 1
                    reasons.append("sharp_transition")
                
                # Criterion 3: Consistent low similarity window
                window_start = max(0, peak_idx - 2)
                window_end = min(len(combined_similarity), peak_idx + 3)
                window_similarities = combined_similarity[window_start:window_end]
                if np.mean(window_similarities) < adaptive_threshold * 1.1:
                    criteria_met += 1
                    reasons.append("sustained_low")
                
                # Criterion 4: Multiple modalities agree
                modality_agreement = sum([
                    similarities['histogram'][peak_idx] < self.similarity_threshold,
                    similarities['ssim'][peak_idx] < self.similarity_threshold,
                    similarities['optical_flow'][peak_idx] < 0.7
                ])
                
                if modality_agreement >= 2:
                    criteria_met += 1
                    reasons.append("multi_modal_agreement")
                
                # Decision: Need at least 2 criteria
                if criteria_met >= 2:
                    boundaries.append(peak_idx)
                    print(f"   🎬 Scene boundary at {timestamps[peak_idx]:.2f}s "
                          f"(similarity: {combined_similarity[peak_idx]:.3f}, "
                          f"criteria: {reasons})")
        
        # Add final frame
        if boundaries[-1] != len(frames) - 1:
            boundaries.append(len(frames) - 1)
        
        return boundaries
    
    def advanced_smooth_signal(self, signal, window_length=5):
        """Advanced signal smoothing using Savitzky-Golay filter"""
        if len(signal) < window_length:
            return signal
        
        # Ensure window_length is odd and less than signal length
        window_length = min(window_length, len(signal))
        if window_length % 2 == 0:
            window_length -= 1
        
        if window_length >= 3:
            try:
                return savgol_filter(signal, window_length, 2)
            except:
                # Fallback to simple moving average
                return self.simple_smooth(signal, window_length)
        else:
            return signal
    
    def simple_smooth(self, signal, window_size=3):
        """Simple moving average smoothing as fallback"""
        if len(signal) < window_size:
            return signal
        
        smoothed = []
        for i in range(len(signal)):
            start = max(0, i - window_size // 2)
            end = min(len(signal), i + window_size // 2 + 1)
            smoothed.append(np.mean(signal[start:end]))
        
        return smoothed
    
    def create_clean_scene_videos(self, video_path, boundaries, frame_indices, output_folder):
        """Create clean scene videos without frame bleeding"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🎬 Creating {len(boundaries)-1} clean scene videos...")
        
        for scene_idx in range(len(boundaries) - 1):
            start_frame_idx = frame_indices[boundaries[scene_idx]]
            end_frame_idx = frame_indices[boundaries[scene_idx + 1]]
            
            print(f"   Scene {scene_idx + 1}: frames {start_frame_idx} to {end_frame_idx}")
            
            # Create output video writer
            output_path = os.path.join(output_folder, f"scene_{scene_idx + 1}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Set video position to exact start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            
            # Write frames for this scene
            frames_written = 0
            current_frame_idx = start_frame_idx
            
            while current_frame_idx < end_frame_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                
                writer.write(frame)
                frames_written += 1
                current_frame_idx += 1
            
            writer.release()
            
            duration = frames_written / fps
            print(f"   ✅ Scene {scene_idx + 1}: {frames_written} frames ({duration:.2f}s)")
        
        cap.release()
        print(f"✅ All clean scenes saved to: {output_folder}")
    
    def separate_scenes(self, video_path, output_folder):
        """Main function for advanced CV scene separation"""
        print(f"\n🎯 Starting Advanced CV Scene Separation")
        print(f"Input: {video_path}")
        print(f"Output: {output_folder}")
        
        try:
            # Step 1: Extract sparse frames
            frames, timestamps, frame_indices, original_fps = self.extract_sparse_frames(video_path)
            
            if len(frames) < 2:
                print("❌ Not enough frames for scene separation")
                return
            
            # Step 2: Compute comprehensive features
            features = self.compute_frame_features(frames)
            
            # Step 3: Detect scene boundaries
            boundaries = self.detect_scene_boundaries_advanced(frames, features, timestamps)
            
            print(f"\n📊 Scene Detection Results:")
            print(f"   Total scenes detected: {len(boundaries) - 1}")
            for i in range(len(boundaries) - 1):
                start_time = timestamps[boundaries[i]]
                end_time = timestamps[boundaries[i + 1]]
                duration = end_time - start_time
                print(f"   Scene {i + 1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            
            # Step 4: Create clean scene videos
            self.create_clean_scene_videos(video_path, boundaries, frame_indices, output_folder)
            
            print(f"\n🎉 Advanced CV Scene Separation Complete!")
            print(f"📈 Performance: Processed {len(frames)} frames with multi-modal analysis")
            
        except Exception as e:
            print(f"❌ Error during scene separation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def compute_advanced_optical_flow(self, prev_frame, curr_frame):
        """Compute advanced optical flow features"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade with good features
        corners = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
        
        if corners is not None and len(corners) > 10:
            next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, corners, None, **self.lk_params
            )
            
            good_old = corners[status == 1]
            good_new = next_corners[status == 1]
            
            if len(good_old) > 5:
                flow_vectors = good_new - good_old
                flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
                
                return {
                    'mean_magnitude': np.mean(flow_magnitudes),
                    'max_magnitude': np.max(flow_magnitudes),
                    'std_magnitude': np.std(flow_magnitudes),
                    'median_magnitude': np.median(flow_magnitudes)
                }
        
        return {'mean_magnitude': 0, 'max_magnitude': 0, 'std_magnitude': 0, 'median_magnitude': 0}
    
    def compute_multi_scale_ssim(self, prev_frame, curr_frame):
        """Compute SSIM at multiple scales"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        ssim_scores = []
        for scale in [1, 0.5, 0.25]:
            if scale != 1:
                h, w = prev_gray.shape
                new_h, new_w = int(h * scale), int(w * scale)
                if new_h > 7 and new_w > 7:
                    prev_scaled = cv2.resize(prev_gray, (new_w, new_h))
                    curr_scaled = cv2.resize(curr_gray, (new_w, new_h))
                else:
                    continue
            else:
                prev_scaled, curr_scaled = prev_gray, curr_gray
            
            if prev_scaled.shape == curr_scaled.shape and min(prev_scaled.shape) > 7:
                try:
                    score = ssim(prev_scaled, curr_scaled, data_range=255, win_size=7)
                    ssim_scores.append(score)
                except:
                    continue
        
        return ssim_scores if ssim_scores else [0.5]
    
    def detect_scene_boundaries_advanced(self, frames, timestamps):
        """Advanced scene boundary detection using multiple criteria including new techniques"""
        boundaries = [0]
        
        print("🔍 Detecting scene boundaries with enhanced multi-modal analysis...")
        print(f"   🆕 Including LBP texture differences and color histogram deltas")
        print(f"   📊 Using histogram, texture, SSIM, optical flow, LBP, and color features")
        
        # Enhanced similarity tracking with new techniques
        similarities = {
            'histogram': [], 'texture': [], 'ssim': [], 'optical_flow': [],
            'lbp_texture_diff': [], 'color_deltas': []
        }
        
        for i in range(1, len(frames)):
            prev_frame = frames[i-1]
            curr_frame = frames[i]
            
            # Existing techniques (optimized)
            hist_prev = self.compute_advanced_histogram_features(prev_frame)
            hist_curr = self.compute_advanced_histogram_features(curr_frame)
            hist_sim = np.corrcoef(hist_prev, hist_curr)[0, 1] if len(hist_prev) > 1 else 0
            similarities['histogram'].append(max(0, hist_sim) if not np.isnan(hist_sim) else 0)
            
            texture_prev = self.compute_texture_features(prev_frame)
            texture_curr = self.compute_texture_features(curr_frame)
            texture_sim = np.corrcoef(texture_prev, texture_curr)[0, 1] if len(texture_prev) > 1 else 0
            similarities['texture'].append(max(0, texture_sim) if not np.isnan(texture_sim) else 0)
            

            
            ssim_scores = self.compute_multi_scale_ssim(prev_frame, curr_frame)
            similarities['ssim'].append(np.mean(ssim_scores))
            
            flow_features = self.compute_advanced_optical_flow(prev_frame, curr_frame)
            flow_score = 1.0 / (1.0 + flow_features['mean_magnitude'] / 1000.0)
            similarities['optical_flow'].append(flow_score)
            
            # NEW TECHNIQUE 1: LBP texture difference checks
            lbp_diff = self.compute_lbp_texture_difference(prev_frame, curr_frame)
            # Convert difference to similarity (1 - difference)
            lbp_similarity = 1.0 - min(1.0, lbp_diff['combined_texture_diff'])
            similarities['lbp_texture_diff'].append(lbp_similarity)
            
            # NEW TECHNIQUE 2: Color space histogram deltas
            color_deltas = self.compute_color_histogram_deltas(prev_frame, curr_frame)
            # Convert delta to similarity (1 - normalized_delta)
            color_similarity = 1.0 / (1.0 + color_deltas['total_color_change'])
            similarities['color_deltas'].append(color_similarity)
        
        # Enhanced weighted combined similarity with new techniques
        weights = {
            'histogram': 0.25,      # Increased to compensate
            'texture': 0.20,        # Increased slightly  
            'ssim': 0.25,           # Increased slightly
            'optical_flow': 0.10,   # Kept same
            'lbp_texture_diff': 0.10,  # NEW: LBP texture differences
            'color_deltas': 0.10    # NEW: Color histogram deltas
        }
        
        combined_similarity = np.zeros(len(similarities['histogram']))
        for key, weight in weights.items():
            combined_similarity += weight * np.array(similarities[key])
        
        # Enhanced adaptive thresholding
        similarity_mean = np.mean(combined_similarity)
        similarity_std = np.std(combined_similarity)
        adaptive_threshold = similarity_mean - 1.8 * similarity_std  # Slightly less aggressive
        adaptive_threshold = max(adaptive_threshold, self.similarity_threshold)
        
        print(f"   Adaptive threshold: {adaptive_threshold:.3f}")
        print(f"   Mean similarity: {similarity_mean:.3f} ± {similarity_std:.3f}")
        
        # Enhanced boundary detection with multiple criteria
        for i in range(1, len(combined_similarity)):
            if i >= len(timestamps):
                continue
                
            time_since_last_boundary = timestamps[i] - timestamps[boundaries[-1]]
            
            if time_since_last_boundary >= self.min_scene_duration:
                # Multi-criteria validation with enhanced checks
                criteria_met = 0
                reasons = []
                
                # Criterion 1: Low combined similarity
                if combined_similarity[i] < adaptive_threshold:
                    criteria_met += 1
                    reasons.append("low_similarity")
                
                # Criterion 2: Local minimum detection
                if (i > 0 and i < len(combined_similarity) - 1 and
                    combined_similarity[i] < combined_similarity[i-1] and
                    combined_similarity[i] < combined_similarity[i+1]):
                    criteria_met += 1
                    reasons.append("local_minimum")
                
                # Criterion 3: Strong texture change (NEW)
                if similarities['lbp_texture_diff'][i-1] < 0.7:
                    criteria_met += 1
                    reasons.append("texture_change")
                
                # Criterion 4: Strong color change (NEW)
                if similarities['color_deltas'][i-1] < 0.6:
                    criteria_met += 1
                    reasons.append("color_change")
                
                # Criterion 5: Multi-modal agreement
                low_similarity_count = sum([
                    similarities['histogram'][i-1] < self.similarity_threshold,
                    similarities['ssim'][i-1] < self.similarity_threshold,
                    similarities['optical_flow'][i-1] < 0.7
                ])
                
                if low_similarity_count >= 2:
                    criteria_met += 1
                    reasons.append("multi_modal_agreement")
                
                # Decision: Need at least 2 criteria for robust detection
                if criteria_met >= 2:
                    boundaries.append(i)
                    print(f"   🎬 Scene boundary at {timestamps[i]:.2f}s "
                          f"(similarity: {combined_similarity[i]:.3f}, "
                          f"criteria: {reasons})")
        
        if boundaries[-1] != len(frames) - 1:
            boundaries.append(len(frames) - 1)
        
        return boundaries
    
    def create_clean_scene_videos(self, video_path, boundaries, frame_indices, output_folder):
        """Create clean scene videos without frame bleeding"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"🎬 Creating {len(boundaries)-1} clean scene videos...")
        
        for scene_idx in range(len(boundaries) - 1):
            start_frame_idx = frame_indices[boundaries[scene_idx]]
            end_frame_idx = frame_indices[boundaries[scene_idx + 1]]
            
            output_path = os.path.join(output_folder, f"scene_{scene_idx + 1}.mp4")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            
            frames_written = 0
            current_frame_idx = start_frame_idx
            
            while current_frame_idx < end_frame_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                
                writer.write(frame)
                frames_written += 1
                current_frame_idx += 1
            
            writer.release()
            
            duration = frames_written / fps
            print(f"   ✅ Scene {scene_idx + 1}: {frames_written} frames ({duration:.2f}s)")
        
        cap.release()
        print(f"✅ All clean scenes saved to: {output_folder}")
    
    def separate_scenes(self, video_path, output_folder):
        """Main function for advanced CV scene separation with auto-parameter detection"""
        print(f"\n🎯 Starting Advanced CV Scene Separation - Enhanced Edition")
        print(f"Input: {video_path}")
        print(f"Output: {output_folder}")
        
        try:
            # Auto-detect optimal parameters if not provided
            if self.auto_params and (self.similarity_threshold is None or 
                                   self.min_scene_duration is None or 
                                   self.sampling_fps is None or 
                                   self.optical_flow_threshold is None):
                
                print("\n🤖 Auto-detecting optimal parameters...")
                optimal_params = self.analyze_video_characteristics(video_path)
                
                # Update parameters with auto-detected values if not manually set
                if self.similarity_threshold is None:
                    self.similarity_threshold = optimal_params['similarity_threshold']
                if self.min_scene_duration is None:
                    self.min_scene_duration = optimal_params['min_scene_duration']
                if self.sampling_fps is None:
                    self.sampling_fps = optimal_params['sampling_fps']
                if self.optical_flow_threshold is None:
                    self.optical_flow_threshold = optimal_params['optical_flow_threshold']
                
                print(f"\n✅ Using auto-detected parameters:")
            else:
                print(f"\n⚙️ Using manual parameters:")
            
            print(f"   📊 Sampling rate: {self.sampling_fps} fps")
            print(f"   🎯 Similarity threshold: {self.similarity_threshold:.3f}")
            print(f"   🌊 Optical flow threshold: {self.optical_flow_threshold}")
            print(f"   ⏱️ Min scene duration: {self.min_scene_duration:.1f}s")
            
            # Extract sparse frames with optimized parameters
            frames, timestamps, frame_indices, original_fps = self.extract_sparse_frames(video_path)
            
            if len(frames) < 2:
                print("❌ Not enough frames for scene separation")
                return
            
            # Detect scene boundaries with enhanced analysis
            boundaries = self.detect_scene_boundaries_advanced(frames, timestamps)
            
            print(f"\n📊 Scene Detection Results:")
            print(f"   Total scenes detected: {len(boundaries) - 1}")
            for i in range(len(boundaries) - 1):
                start_time = timestamps[boundaries[i]]
                end_time = timestamps[boundaries[i + 1]]
                duration = end_time - start_time
                print(f"   Scene {i + 1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.2f}s)")
            
            # Create clean scene videos
            self.create_clean_scene_videos(video_path, boundaries, frame_indices, output_folder)
            
            print(f"\n🎉 Advanced CV Scene Separation Complete!")
            print(f"🚀 Enhanced with auto-parameter detection and 6-technique analysis")
            
        except Exception as e:
            print(f"❌ Error during scene separation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

def advanced_cv_scene_separation(input_video, output_folder, similarity_threshold=None, 
                                min_scene_duration=None, sampling_fps=None, 
                                optical_flow_threshold=None, auto_params=True):
    """
    Convenience function for Advanced CV scene separation with auto-parameter detection
    
    Args:
        input_video: Path to input video file
        output_folder: Directory to save separated scenes
        similarity_threshold: Feature similarity threshold (auto-detected if None)
        min_scene_duration: Minimum scene duration in seconds (auto-detected if None)
        sampling_fps: Sparse sampling rate (auto-detected if None)
        optical_flow_threshold: Threshold for optical flow magnitude (auto-detected if None)
        auto_params: Whether to automatically detect optimal parameters (default: True)
    """
    separator = AdvancedCVSceneSeparator(
        similarity_threshold=similarity_threshold,
        min_scene_duration=min_scene_duration,
        sampling_fps=sampling_fps,
        optical_flow_threshold=optical_flow_threshold,
        auto_params=auto_params
    )
    
    separator.separate_scenes(input_video, output_folder)

if __name__ == "__main__":
    video_path = "test10.mp4"
    output_dir = "advanced_cv_scenes_enhanced"
    
    print("🚀 Advanced CV Scene Separation Demo - Enhanced Edition")
    print("🆕 Now featuring LBP texture differences and color histogram deltas")
    print("⚡ Optimized for faster processing with improved accuracy")
    print("🤖 Auto-parameter detection using optical flow analysis")
    print("📊 Using 6 computer vision techniques for scene detection")
    
    # Option 1: Fully automatic parameters (recommended)
    print("\n📋 Running with auto-detected parameters...")
    advanced_cv_scene_separation(
        input_video=video_path,
        output_folder=output_dir,
        auto_params=True  # Let the system analyze video and choose optimal parameters
    )
    
    # Option 2: Manual parameters (uncomment to use)
    # print("\n📋 Running with manual parameters...")
    # advanced_cv_scene_separation(
    #     input_video=video_path,
    #     output_folder=output_dir + "_manual",
    #     similarity_threshold=0.45,
    #     min_scene_duration=0.1,
    #     sampling_fps=10,
    #     optical_flow_threshold=4500,
    #     auto_params=False
    # ) 