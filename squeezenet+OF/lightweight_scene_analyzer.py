"""
Lightweight Scene Analyzer - Advanced version with 6 machine learning models
This is the most comprehensive script that trains multiple AI models to classify video scenes
It uses advanced feature extraction and compares different machine learning approaches
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

class LightweightFeatureExtractor:
    """
    Advanced feature extractor that gets detailed information from videos
    Like a super-smart detective that analyzes videos in many different ways
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Set up the advanced feature extractor
        
        What it does: Prepares all the tools needed for extracting features
        Why important: Uses the best device available for fast processing
        Simple explanation: Like getting all your detective tools ready before investigating
        """
        self.device = device
        self.setup_squeezenet()
        self.setup_transforms()
        
    def setup_squeezenet(self):
        """
        Load and prepare the SqueezeNet neural network for feature extraction
        
        What it does: Sets up a lightweight but powerful AI model for image analysis
        Why important: SqueezeNet is fast and accurate for mobile/lightweight applications
        Simple explanation: Like hiring a smart but efficient expert to analyze images
        """
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        # Remove the classifier to get features
        self.squeezenet.features = nn.Sequential(*list(self.squeezenet.features.children()))
        self.squeezenet.eval()
        self.squeezenet.to(self.device)
        
        # Freeze parameters for faster inference
        for param in self.squeezenet.parameters():
            param.requires_grad = False
            
        print(f"✅ SqueezeNet loaded on {self.device}")
    
    def setup_transforms(self):
        """
        Prepare image preprocessing steps for the neural network
        
        What it does: Creates a standard way to prepare images for SqueezeNet
        Why important: Neural networks need images in a specific format to work properly
        Simple explanation: Like having a standard recipe for preparing ingredients
        """
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_advanced_optical_flow(self, frames, max_frames=50):
        """
        Extract detailed movement patterns from video frames
        
        What it does: Analyzes how objects move between frames using advanced techniques
        Why important: Motion patterns are key to distinguishing different scene types
        Simple explanation: Like studying how things move to understand what's happening
        """
        if len(frames) < 2:
            return np.zeros(80)  # Return empty features if not enough frames
        
        # Smart frame sampling - keep important frames
        if len(frames) > max_frames:
            # Keep first, last, and uniformly sampled frames
            indices = [0] + list(np.linspace(1, len(frames)-2, max_frames-2, dtype=int)) + [len(frames)-1]
            frames = [frames[i] for i in indices]
        
        # Initialize feature collectors
        flow_features = []
        motion_patterns = []
        direction_vectors = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Use corner detection for Lucas-Kanade optical flow
            # Find good points to track (corners and edges)
            corners = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=10)
            
            if corners is not None and len(corners) > 10:
                # Lucas-Kanade optical flow - track specific points
                next_pts, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
                # Filter good points that were tracked successfully
                good_new = next_pts[status == 1]
                good_old = corners[status == 1]
                flow_vectors = good_new - good_old.reshape(-1, 2)
            else:
                # Fallback to dense optical flow if no corners found
                dense_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Sample points from dense flow
                h, w = dense_flow.shape[:2]
                y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
                flow_vectors = dense_flow[y, x]
                step = 10
            
            if len(flow_vectors) > 0:
                # Calculate magnitudes and angles of movement
                magnitude = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                angle = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
                
                # Enhanced statistical features for magnitude (speed of movement)
                mag_stats = [
                    np.mean(magnitude), np.std(magnitude), np.median(magnitude),
                    np.percentile(magnitude, 10), np.percentile(magnitude, 25), 
                    np.percentile(magnitude, 75), np.percentile(magnitude, 90),
                    np.max(magnitude), np.min(magnitude),
                    np.var(magnitude)  # Variance in speed
                ]
                
                # Enhanced statistical features for angles (direction of movement)
                angle_stats = [
                    np.mean(angle), np.std(angle), np.median(angle),
                    np.percentile(angle, 25), np.percentile(angle, 75),
                    np.var(angle)  # Variance in direction
                ]
                
                # Motion consistency and patterns
                direction_consistency = 1.0 - (np.std(angle) / np.pi)  # How consistent is movement direction
                motion_intensity = np.mean(magnitude)  # Overall speed of movement
                motion_concentration = np.sum(magnitude > np.mean(magnitude)) / len(magnitude)  # How much fast movement
                
                # Dominant direction analysis
                dominant_angle = np.median(angle)  # Main direction of movement
                angle_spread = np.percentile(angle, 75) - np.percentile(angle, 25)  # How spread out directions are
                
                frame_features = (mag_stats + angle_stats + 
                                [direction_consistency, motion_intensity, motion_concentration,
                                 dominant_angle, angle_spread])
                
                flow_features.extend(frame_features)
                motion_patterns.append(motion_intensity)
                direction_vectors.append(dominant_angle)
        
        if not flow_features:
            return np.zeros(80)
        
        # Reshape and aggregate features
        flow_features = np.array(flow_features).reshape(-1, 21)  # 21 features per frame pair
        
        # Temporal motion analysis - how motion changes over time
        motion_patterns = np.array(motion_patterns)
        direction_vectors = np.array(direction_vectors)
        
        temporal_features = [
            # Motion intensity patterns
            np.mean(motion_patterns), np.std(motion_patterns),
            np.max(motion_patterns), np.min(motion_patterns),
            np.median(motion_patterns),
            
            # Direction stability
            np.mean(direction_vectors), np.std(direction_vectors),
            np.median(direction_vectors),
            
            # Motion dynamics
            len(motion_patterns),  # Number of valid flow computations
            np.sum(motion_patterns > np.mean(motion_patterns)) / len(motion_patterns),  # High motion ratio
            
            # Motion trends - does movement increase or decrease over time?
            np.corrcoef(range(len(motion_patterns)), motion_patterns)[0, 1] if len(motion_patterns) > 1 else 0,
        ]
        
        # Statistical aggregation across all frame pairs
        aggregated_features = [
            np.mean(flow_features, axis=0),    # Average of all features
            np.std(flow_features, axis=0),     # Standard deviation of all features  
            np.max(flow_features, axis=0),     # Maximum values
            np.min(flow_features, axis=0),     # Minimum values
        ]
        
        # Combine all features into one big feature vector
        combined = np.concatenate([
            np.concatenate(aggregated_features),  # 84 features
            temporal_features                     # 11 features
        ])
        
        return combined[:80]  # Return fixed size feature vector (80 features)
    
    def extract_squeezenet_features(self, frames, max_frames=15):
        """
        Extract deep learning features using SqueezeNet
        
        What it does: Uses a pre-trained neural network to get high-level image features
        Why important: Neural networks can see patterns that traditional methods miss
        Simple explanation: Like having an expert artist describe what makes each scene unique
        """
        if not frames:
            return np.zeros(256)  # Return empty features if no frames
        
        # Intelligent frame sampling for key moments
        if len(frames) > max_frames:
            # Sample key frames: beginning, middle, end + some random ones
            key_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
            if max_frames > 5:
                random_indices = np.random.choice(
                    range(1, len(frames)-1), 
                    min(max_frames-5, len(frames)-3), 
                    replace=False
                ).tolist()
                indices = sorted(set(key_indices + random_indices))
            else:
                indices = key_indices[:max_frames]
            frames = [frames[i] for i in indices]
        
        features = []
        
        with torch.no_grad():  # Don't calculate gradients (faster inference)
            for frame in frames:
                # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms to prepare for neural network
                input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                
                # Extract features with mixed precision for speed
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():  # Use mixed precision on GPU
                        feature_map = self.squeezenet.features(input_tensor)
                else:
                    feature_map = self.squeezenet.features(input_tensor)
                
                # Global average pooling to reduce dimensions
                pooled_features = torch.mean(feature_map, dim=(2, 3))
                features.append(pooled_features.cpu().numpy().flatten())
        
        if not features:
            return np.zeros(256)
        
        # Aggregate features across frames
        features = np.array(features)
        
        # Use both mean and std for temporal aggregation
        aggregated = np.concatenate([
            np.mean(features, axis=0),  # Average features across frames
            np.std(features, axis=0)    # Standard deviation across frames
        ])
        
        # Ensure fixed size and reduce to lightweight representation
        return aggregated[:256]  # Return 256 features
    
    def extract_image_processing_features(self, frames):
        """
        Extract traditional computer vision features from frames
        
        What it does: Uses classical image processing to find edges, colors, textures
        Why important: Traditional features complement deep learning features
        Simple explanation: Like using basic photography principles alongside expert analysis
        """
        if not frames:
            return np.zeros(60)
        
        # Sample a few key frames for analysis
        key_frames = []
        if len(frames) >= 5:
            indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1]
            key_frames = [frames[i] for i in indices]
        else:
            key_frames = frames
        
        all_features = []
        
        for frame in key_frames:
            frame_features = []
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Edge features - find boundaries and sharp changes
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size  # How much of the image has edges
            edge_mean = np.mean(edges)
            edge_std = np.std(edges)
            
            # Texture features using Laplacian (measures roughness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)  # How varied the texture is
            texture_mean = np.mean(np.abs(laplacian))
            
            # Color histogram features - what colors are present
            hist_b = cv2.calcHist([frame], [0], None, [16], [0, 256])  # Blue channel
            hist_g = cv2.calcHist([frame], [1], None, [16], [0, 256])  # Green channel
            hist_r = cv2.calcHist([frame], [2], None, [16], [0, 256])  # Red channel
            
            # Normalize histograms
            hist_b = hist_b.flatten() / np.sum(hist_b)
            hist_g = hist_g.flatten() / np.sum(hist_g)
            hist_r = hist_r.flatten() / np.sum(hist_r)
            
            # Color statistics - average colors in the image
            color_mean = [np.mean(frame[:, :, i]) for i in range(3)]
            color_std = [np.std(frame[:, :, i]) for i in range(3)]
            
            # HSV statistics - Hue, Saturation, Value analysis
            hsv_mean = [np.mean(hsv[:, :, i]) for i in range(3)]
            hsv_std = [np.std(hsv[:, :, i]) for i in range(3)]
            
            # Brightness and contrast measures
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Combine all basic features for this frame
            frame_features = [
                edge_density, edge_mean, edge_std,
                texture_variance, texture_mean,
                brightness, contrast
            ] + color_mean + color_std + hsv_mean + hsv_std
            
            # Add top histogram bins (most common colors)
            frame_features.extend(hist_b[:3].tolist())  # Top 3 blue bins
            frame_features.extend(hist_g[:3].tolist())  # Top 3 green bins
            frame_features.extend(hist_r[:3].tolist())  # Top 3 red bins
            
            all_features.append(frame_features)
        
        if not all_features:
            return np.zeros(60)
        
        # Aggregate across frames
        all_features = np.array(all_features)
        aggregated = np.concatenate([
            np.mean(all_features, axis=0),  # Average across frames
            np.std(all_features, axis=0)    # Standard deviation across frames
        ])
        
        return aggregated[:60]  # Return 60 traditional features

class OptimizedVideoProcessor:
    """
    Advanced video processor that handles videos and extracts features efficiently
    Like a smart video processing factory that turns raw videos into useful data
    """
    def __init__(self, dataset_path, feature_extractor):
        """
        Set up the video processor with dataset location and feature extractor
        
        What it does: Prepares to process videos from the dataset using our feature extractor
        Why important: Organizes all the tools and paths needed for video processing
        Simple explanation: Like setting up a workshop with all tools and materials ready
        """
        self.dataset_path = dataset_path
        self.feature_extractor = feature_extractor
        self.categories = ['beach', 'chaparral', 'forest', 'intersection', 'mountain', 'port']
        
    def extract_frames_by_fps(self, video_path, target_fps=2):
        """
        Extract frames from video at a specific frame rate
        
        What it does: Takes a video and pulls out frames at regular time intervals
        Why important: We don't need every frame - sampling saves time and memory
        Simple explanation: Like taking a photo every few seconds instead of constantly
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return []
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(original_fps / target_fps))
        
        frames = []
        frame_indices = []
        frame_count = 0
        
        print(f"  📽️  Processing: {video_path}")
        print(f"    Original FPS: {original_fps:.2f}, Target FPS: {target_fps}")
        print(f"    Frame interval: {frame_interval}, Duration: {duration:.2f}s")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sample frames according to target FPS
            if frame_count % frame_interval == 0:
                # Resize frame for consistent processing
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        print(f"    Extracted {len(frames)} frames from {total_frames} total frames")
        return frames, frame_indices
    
    def extract_features_from_frames(self, frames, video_category):
        """
        Extract features from each individual frame in a video
        
        What it does: Analyzes each frame separately to get detailed features
        Why important: Frame-level analysis gives us more training data and detail
        Simple explanation: Like examining each photo in an album individually
        """
        frame_features = []
        
        for i, frame in enumerate(frames):
            # Extract features for each frame individually
            
            # 1. Basic image features (simplified for single frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_mean = np.mean(edges)
            
            # Texture features
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_variance = np.var(laplacian)
            
            # Color features
            color_mean = [np.mean(frame[:, :, j]) for j in range(3)]
            color_std = [np.std(frame[:, :, j]) for j in range(3)]
            
            # HSV features
            hsv_mean = [np.mean(hsv[:, :, j]) for j in range(3)]
            
            # Combine basic features (15 features)
            basic_features = [edge_density, edge_mean, texture_variance] + color_mean + color_std + hsv_mean
            
            # 2. SqueezeNet features for single frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.feature_extractor.transform(frame_rgb).unsqueeze(0).to(self.feature_extractor.device)
            
            with torch.no_grad():
                if self.feature_extractor.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        feature_map = self.feature_extractor.squeezenet.features(input_tensor)
                else:
                    feature_map = self.feature_extractor.squeezenet.features(input_tensor)
                
                # Global average pooling
                squeezenet_features = torch.mean(feature_map, dim=(2, 3)).cpu().numpy().flatten()
            
            # Combine all features for this frame
            combined_features = np.concatenate([
                basic_features,  # 15 features
                squeezenet_features[:128]  # 128 SqueezeNet features (reduced)
            ])  # Total: 143 features per frame
            
            frame_features.append({
                'features': combined_features,
                'label': video_category,
                'frame_index': i
            })
        
        return frame_features
    
    def process_video_frame_level(self, video_info, target_fps=2):
        """
        Process a complete video and extract features from all frames
        
        What it does: Takes a video, extracts frames, and gets features from each frame
        Why important: This gives us many training examples from each video
        Simple explanation: Like analyzing every page of a book instead of just the cover
        """
        video_path, category = video_info
        
        # Extract frames according to FPS
        frames, frame_indices = self.extract_frames_by_fps(video_path, target_fps)
        
        if not frames:
            return []
        
        # Extract features from each frame
        frame_features = self.extract_features_from_frames(frames, category)
        
        # Add video path to each frame feature
        for frame_feature in frame_features:
            frame_feature['video_path'] = video_path
        
        return frame_features
    
    def process_single_video_lightweight(self, video_info):
        """
        Process one video using our lightweight feature extraction method
        
        What it does: Takes one video and extracts all types of features efficiently
        Why important: Combines multiple feature types for comprehensive analysis
        Simple explanation: Like examining a photo using multiple different techniques
        """
        video_path, category = video_info
        
        try:
            frames = self.load_video_frames_optimized(video_path)
            
            if len(frames) < 2:
                print(f"Warning: Insufficient frames in {video_path}")
                return None
            
            # Extract different types of features
            print(f"Processing {os.path.basename(video_path)}...", end=" ")
            
            optical_flow_features = self.feature_extractor.extract_advanced_optical_flow(frames)
            squeezenet_features = self.feature_extractor.extract_squeezenet_features(frames)
            image_features = self.feature_extractor.extract_image_processing_features(frames)
            
            # Combine all features
            combined_features = np.concatenate([
                optical_flow_features,    # 80 features
                squeezenet_features,      # 256 features
                image_features           # 60 features
            ])  # Total: 396 features
            
            print("✅")
            
            return {
                'features': combined_features,
                'label': category,
                'video_path': video_path
            }
            
        except Exception as e:
            print(f"❌ Error processing {video_path}: {str(e)}")
            return None
    
    def extract_features_parallel(self, n_workers=4):
        """
        Process multiple videos at the same time for speed
        
        What it does: Uses multiple CPU cores to process several videos simultaneously
        Why important: Much faster than processing videos one by one
        Simple explanation: Like having multiple workers on an assembly line
        """
        video_list = []
        
        for category in self.categories:
            category_path = os.path.join(self.dataset_path, category)
            for i in range(1, 16):  # Now processing all 15 videos (01-15)
                video_name = f"{category}{i:02d}.mp4"
                video_path = os.path.join(category_path, video_name)
                if os.path.exists(video_path):
                    video_list.append((video_path, category))
        
        print(f"🎬 Found {len(video_list)} videos to process")
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.process_single_video_lightweight, video_info) 
                      for video_info in video_list]
            
            for future in tqdm(futures, desc="Extracting features"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def extract_features_from_video_list(self, video_list, n_workers=4):
        """
        Extract features from a specific list of videos
        
        What it does: Takes a custom list of videos and processes them in parallel
        Why important: Allows us to process only selected videos (like train/test splits)
        Simple explanation: Like choosing specific books to analyze from a library
        """
        print(f"🎬 Extracting features from {len(video_list)} selected videos...")
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.process_single_video_lightweight, video_info) 
                      for video_info in video_list]
            
            for future in tqdm(futures, desc="Extracting features"):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results

    def extract_features_from_video_list_frame_level(self, video_list, target_fps=2, n_workers=4):
        """
        Extract features from video list analyzing each frame individually
        
        What it does: Processes videos to get features from every individual frame
        Why important: Gives us much more training data - every frame becomes a sample
        Simple explanation: Like studying every single page instead of just chapter summaries
        """
        print(f"🎬 Extracting frame-level features from {len(video_list)} videos...")
        print(f"   Target FPS for sampling: {target_fps}")
        
        all_frame_features = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.process_video_frame_level, video_info, target_fps) 
                      for video_info in video_list]
            
            for future in tqdm(futures, desc="Processing videos for frames"):
                frame_features = future.result()
                if frame_features:
                    all_frame_features.extend(frame_features)
        
        print(f"✅ Extracted features from {len(all_frame_features)} frames total")
        return all_frame_features

class ComprehensiveModelTrainer:
    """
    Advanced trainer that tests 6 different machine learning models
    Like a teacher who uses multiple teaching methods to find what works best
    """
    def __init__(self):
        """
        Set up all the tools needed for comprehensive model training
        
        What it does: Prepares data preprocessing tools and storage for multiple models
        Why important: We need to standardize data and compare different approaches
        Simple explanation: Like preparing different teaching materials and grade books
        """
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.models = {}
        self.results = {}
        
    def split_videos_stratified(self, dataset_path, train_per_category=12, test_per_category=3, random_state=42):
        """
        Carefully divide videos into training and testing groups
        
        What it does: Splits videos ensuring equal representation from each scene category
        Why important: Prevents bias and ensures fair testing on unseen videos
        Simple explanation: Like dividing a deck of cards evenly for a fair game
        """
        categories = ['beach', 'chaparral', 'forest', 'intersection', 'mountain', 'port']
        
        print(f"🔄 Step 1: Splitting videos into train/test sets (80/20 split)")
        print(f"  Train per category: {train_per_category} videos")
        print(f"  Test per category: {test_per_category} videos")
        
        # Collect all videos by category
        category_videos = {}
        for category in categories:
            category_path = os.path.join(dataset_path, category)
            videos = []
            for i in range(1, 16):  # Videos 01-15
                video_name = f"{category}{i:02d}.mp4"
                video_path = os.path.join(category_path, video_name)
                if os.path.exists(video_path):
                    videos.append((video_path, category))
            category_videos[category] = videos
            print(f"  📁 {category}: {len(videos)} videos found")
        
        # Split each category
        np.random.seed(random_state)
        train_videos = []
        test_videos = []
        
        for category, videos in category_videos.items():
            # Shuffle videos for random selection
            shuffled_videos = videos.copy()
            np.random.shuffle(shuffled_videos)
            
            # Split based on available videos
            available_videos = len(shuffled_videos)
            min_videos = train_per_category + test_per_category
            
            if available_videos >= min_videos:
                # Normal case: enough videos
                train_vids = shuffled_videos[:train_per_category]
                test_vids = shuffled_videos[train_per_category:train_per_category + test_per_category]
            else:
                # Not enough videos: use all but test_per_category for training
                train_count = max(1, available_videos - test_per_category)
                train_vids = shuffled_videos[:train_count]
                test_vids = shuffled_videos[train_count:]
            
            train_videos.extend(train_vids)
            test_videos.extend(test_vids)
            
            print(f"  {category}: {len(train_vids)} train, {len(test_vids)} test")
        
        print(f"\n📊 Final video split:")
        print(f"  Training videos: {len(train_videos)}")
        print(f"  Test videos: {len(test_videos)}")
        
        return train_videos, test_videos
    
    def prepare_data_from_separate_sets(self, train_features, test_features):
        """
        Prepare training and testing data from separate feature sets
        
        What it does: Takes pre-split features and prepares them for machine learning
        Why important: Converts raw features into the format needed by ML algorithms
        Simple explanation: Like organizing study materials and test papers separately
        """
        print(f"\n🔧 Step 3: Preparing data from extracted features...")
        
        # Extract features and labels from training set
        X_train = np.array([item['features'] for item in train_features])
        y_train = np.array([item['label'] for item in train_features])
        
        # Extract features and labels from test set
        X_test = np.array([item['features'] for item in test_features])
        y_test = np.array([item['label'] for item in test_features])
        
        print(f"📊 Dataset summary:")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature dimension: {X_train.shape[1]}")
        
        # Count samples per category
        train_categories = {}
        test_categories = {}
        
        for label in y_train:
            train_categories[label] = train_categories.get(label, 0) + 1
        for label in y_test:
            test_categories[label] = test_categories.get(label, 0) + 1
            
        print(f"\n📊 Samples per category:")
        print(f"  Training: {train_categories}")
        print(f"  Testing: {test_categories}")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Handle any problematic values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train_encoded)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        selected_features = self.feature_selector.get_support().sum()
        print(f"📊 Selected features: {selected_features}")
        
        return X_train_selected, X_test_selected, y_train_encoded, y_test_encoded
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train 6 different machine learning models and compare their performance
        
        What it does: Trains XGBoost, Gradient Boosting, Random Forest, SVM, Logistic Regression, and Ensemble
        Why important: Different models work better for different problems - we test them all
        Simple explanation: Like trying different study methods to see which one works best
        """
        print("\n🤖 Training Multiple Models...")
        
        # 1. XGBoost
        print("  📈 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(self.label_encoder.classes_),
            max_depth=8,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        xgb_model.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_model
        
        # 2. Gradient Boosting
        print("  📈 Training Gradient Boosting...")
        # Check if dataset is large enough for validation-based early stopping
        min_validation_samples = len(self.label_encoder.classes_) * 2  # At least 2 samples per class
        required_train_samples = min_validation_samples / 0.2  # For 20% validation
        
        if X_train.shape[0] >= required_train_samples:
            # Large enough dataset - use validation-based early stopping
            gb_model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.85,
                random_state=42,
                validation_fraction=0.2,
                n_iter_no_change=15
            )
        else:
            # Small dataset - disable early stopping
            gb_model = GradientBoostingClassifier(
                n_estimators=150,  # Reduced to prevent overfitting
                learning_rate=0.1,
                max_depth=6,
                subsample=0.85,
                random_state=42
                # No validation_fraction or n_iter_no_change
            )
        gb_model.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb_model
        
        # 3. Random Forest
        print("  🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['Random Forest'] = rf_model
        
        # 4. Support Vector Machine
        print("  🎯 Training SVM...")
        svm_model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train, y_train)
        self.models['SVM'] = svm_model
        
        # 5. Logistic Regression
        print("  📊 Training Logistic Regression...")
        lr_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            multi_class='ovr'
        )
        lr_model.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr_model
        
        # 6. Ensemble Model
        print("  🎭 Creating Ensemble...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('gb', gb_model),
                ('rf', rf_model)
            ],
            voting='soft'
        )
        ensemble_model.fit(X_train, y_train)
        self.models['Ensemble'] = ensemble_model
        
        # Evaluate all models
        print("\n📊 Evaluating all models...")
        for name, model in self.models.items():
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            print(f"  {name:18}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    def cross_validate_models(self, X_train, y_train):
        """
        Test how well models perform using cross-validation
        
        What it does: Tests each model multiple times on different data splits
        Why important: Gives us a more reliable estimate of how good each model really is
        Simple explanation: Like taking multiple practice tests to see your real skill level
        """
        print("\n🔄 Cross-validation results:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_results = {}
        for name, model in self.models.items():
            if name != 'Ensemble':  # Skip ensemble to avoid nested CV
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    cv_results[name] = {
                        'mean': scores.mean(),
                        'std': scores.std()
                    }
                    print(f"  {name:18}: {scores.mean():.4f} ± {scores.std():.4f}")
                except:
                    print(f"  {name:18}: CV failed")
        
        return cv_results
    
    def generate_detailed_report(self, y_test):
        """
        Create detailed performance reports for all models
        
        What it does: Shows exactly how well each model performs on each scene type
        Why important: Helps us understand which models work best for which scenes
        Simple explanation: Like getting a detailed report card showing grades in each subject
        """
        print("\n📋 Detailed Classification Reports:")
        
        class_names = self.label_encoder.classes_
        
        for model_name, result in self.results.items():
            print(f"\n{'='*20} {model_name} {'='*20}")
            
            predictions = result['predictions']
            accuracy = result['accuracy']
            
            print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Per-class metrics
            report = classification_report(y_test, predictions, 
                                         target_names=class_names, 
                                         output_dict=True,
                                         zero_division=0)
            
            print("\nPer-class Performance:")
            for class_name in class_names:
                if class_name in report:
                    metrics = report[class_name]
                    print(f"  {class_name:12}: Precision={metrics['precision']:.3f}, "
                          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    def plot_comprehensive_results(self, y_test):
        """
        Create visual charts showing how well each model performed
        
        What it does: Makes confusion matrices and accuracy charts for all models
        Why important: Pictures make it easy to see which models work best
        Simple explanation: Like making charts and graphs to show test results visually
        """
        n_models = len(self.models)
        
        # Create subplots for confusion matrices
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        class_names = self.label_encoder.classes_
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
                
            predictions = result['predictions']
            accuracy = result['accuracy']
            
            cm = confusion_matrix(y_test, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {accuracy:.3f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create accuracy comparison bar plot
        plt.figure(figsize=(12, 6))
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        bars = plt.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_all_models(self, save_dir='models'):
        """
        Save all trained models and preprocessing tools to files
        
        What it does: Stores all models and data preparation tools on disk
        Why important: We can use these models later without retraining everything
        Simple explanation: Like saving your completed homework so you don't have to do it again
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f'{name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(model, os.path.join(save_dir, filename))
        
        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
        joblib.dump(self.feature_selector, os.path.join(save_dir, 'feature_selector.pkl'))
        
        print(f"💾 All models saved to '{save_dir}' directory")

    def evaluate_frame_level_results(self, y_test, test_features):
        """
        Analyze results at the individual frame level for detailed insights
        
        What it does: Shows how well models perform on individual frames and videos
        Why important: Gives detailed breakdown of where models succeed or fail
        Simple explanation: Like examining each answer on a test instead of just the total score
        """
        print("\n📊 FRAME-LEVEL EVALUATION RESULTS")
        print("=" * 50)
        
        # Group results by video and category
        video_results = {}
        category_stats = {}
        
        for model_name, result in self.results.items():
            predictions = result['predictions']
            
            print(f"\n🔍 {model_name} Frame Analysis:")
            
            # Initialize stats
            total_frames = len(predictions)
            correct_frames = np.sum(predictions == y_test)
            incorrect_frames = total_frames - correct_frames
            
            print(f"  📊 Overall Frame Statistics:")
            print(f"    Total frames: {total_frames}")
            print(f"    Correctly classified: {correct_frames} ({correct_frames/total_frames*100:.2f}%)")
            print(f"    Incorrectly classified: {incorrect_frames} ({incorrect_frames/total_frames*100:.2f}%)")
            
            # Per-category analysis
            print(f"  📋 Per-Category Frame Analysis:")
            categories = self.label_encoder.classes_
            
            for i, category in enumerate(categories):
                category_mask = y_test == i
                if np.any(category_mask):
                    category_predictions = predictions[category_mask]
                    category_true = y_test[category_mask]
                    
                    correct_in_category = np.sum(category_predictions == category_true)
                    total_in_category = len(category_true)
                    
                    print(f"    {category:12}: {correct_in_category}/{total_in_category} frames correct ({correct_in_category/total_in_category*100:.2f}%)")
            
            # Per-video analysis
            print(f"  🎥 Per-Video Frame Analysis:")
            video_frame_stats = {}
            
            # Group frames by video
            current_video = None
            video_frames = []
            video_preds = []
            video_true = []
            
            for idx, frame_data in enumerate(test_features):
                video_path = frame_data['video_path']
                if current_video != video_path:
                    # Process previous video if exists
                    if current_video is not None and len(video_frames) > 0:
                        video_correct = np.sum(np.array(video_preds) == np.array(video_true))
                        video_total = len(video_frames)
                        video_name = os.path.basename(current_video)
                        print(f"    {video_name:15}: {video_correct}/{video_total} frames correct ({video_correct/video_total*100:.2f}%)")
                        
                        video_frame_stats[current_video] = {
                            'correct': video_correct,
                            'total': video_total,
                            'accuracy': video_correct/video_total
                        }
                    
                    # Start new video
                    current_video = video_path
                    video_frames = []
                    video_preds = []
                    video_true = []
                
                video_frames.append(idx)
                video_preds.append(predictions[idx])
                video_true.append(y_test[idx])
            
            # Process last video
            if current_video is not None and len(video_frames) > 0:
                video_correct = np.sum(np.array(video_preds) == np.array(video_true))
                video_total = len(video_frames)
                video_name = os.path.basename(current_video)
                print(f"    {video_name:15}: {video_correct}/{video_total} frames correct ({video_correct/video_total*100:.2f}%)")
                
                video_frame_stats[current_video] = {
                    'correct': video_correct,
                    'total': video_total,
                    'accuracy': video_correct/video_total
                }
            
            # Store results for this model
            if model_name not in video_results:
                video_results[model_name] = {}
            video_results[model_name] = {
                'total_frames': total_frames,
                'correct_frames': correct_frames,
                'frame_accuracy': correct_frames/total_frames,
                'video_stats': video_frame_stats
            }
        
        return video_results

def main():
    """
    Main function that runs the complete advanced scene analysis training
    
    What it does: Coordinates the entire process from video splitting to model evaluation
    Why important: This is the master coordinator that manages all the complex steps
    Simple explanation: Like a conductor leading an orchestra through a complete symphony
    """
    print("🚀 Lightweight Scene Analysis - Split First, Then Extract Features")
    print("=" * 70)
    
    # Configuration
    dataset_path = "dataset"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    print(f"🔧 PyTorch version: {torch.__version__}")
    
    # Initialize components
    print("\n📊 Initializing lightweight components...")
    feature_extractor = LightweightFeatureExtractor(device=device)
    video_processor = OptimizedVideoProcessor(dataset_path, feature_extractor)
    trainer = ComprehensiveModelTrainer()
    
    # Step 1: Split videos into train/test sets (80/20)
    train_videos, test_videos = trainer.split_videos_stratified(dataset_path, train_per_category=12, test_per_category=3)
    
    # Step 2: Extract frame-level features from training videos
    print(f"\n🎬 Step 2: Extracting frame-level features from training videos...")
    train_features = video_processor.extract_features_from_video_list_frame_level(train_videos, target_fps=2, n_workers=4)
    
    if not train_features:
        print("❌ Error: No training features extracted!")
        return
    
    print(f"✅ Extracted features from {len(train_features)} training frames")
    
    # Step 3: Extract frame-level features from test videos
    print(f"\n🎬 Step 3: Extracting frame-level features from test videos...")
    test_features = video_processor.extract_features_from_video_list_frame_level(test_videos, target_fps=2, n_workers=4)
    
    if not test_features:
        print("❌ Error: No test features extracted!")
        return
    
    print(f"✅ Extracted features from {len(test_features)} test frames")
    
    # Step 4: Prepare data from extracted features
    X_train, X_test, y_train, y_test = trainer.prepare_data_from_separate_sets(train_features, test_features)
    
    # Step 5: Train all models
    print("\n🤖 Step 4: Training all models...")
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 6: Cross-validation
    print("\n📈 Step 5: Cross-validation...")
    cv_results = trainer.cross_validate_models(X_train, y_train)
    
    # Step 7: Detailed evaluation
    print("\n📊 Step 6: Detailed evaluation...")
    trainer.generate_detailed_report(y_test)
    
    # Step 7.5: Frame-level evaluation
    print("\n🎯 Step 6.5: Frame-level evaluation...")
    frame_results = trainer.evaluate_frame_level_results(y_test, test_features)
    
    # Step 8: Visualizations
    print("\n📈 Step 7: Creating visualizations...")
    trainer.plot_comprehensive_results(y_test)
    
    # Step 9: Save models
    print("\n💾 Step 8: Saving all models...")
    trainer.save_all_models()
    
    # Final summary
    best_model = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])
    best_name, best_result = best_model
    
    print("\n" + "=" * 70)
    print("🎯 COMPREHENSIVE TRAINING COMPLETED!")
    print("=" * 70)
    print(f"🏆 Best Model: {best_name}")
    print(f"🎯 Best Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    
    print(f"\n📊 All Model Results:")
    for name, result in sorted(trainer.results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"  {name:18}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    
    target_accuracy = 0.93
    if best_result['accuracy'] >= target_accuracy:
        print(f"\n🎉 SUCCESS! Target accuracy of {target_accuracy*100:.1f}% achieved!")
        print(f"🚀 Best model exceeded target by {(best_result['accuracy']-target_accuracy)*100:.2f} percentage points!")
    else:
        print(f"\n⚠️  Target accuracy of {target_accuracy*100:.1f}% not reached.")
        gap = (target_accuracy - best_result['accuracy']) * 100
        print(f"📊 Gap to target: {gap:.2f} percentage points")
    
    print(f"\n🔧 Workflow Summary:")
    print(f"  1. Split 90 videos → {len(train_videos)} train + {len(test_videos)} test")
    print(f"  2. Extract frames at 2 FPS from each video")
    print(f"  3. Extract features from {len(train_features)} training frames")
    print(f"  4. Extract features from {len(test_features)} test frames")
    print(f"  5. Train and evaluate 6 different models on frame-level")
    
    # Frame-level summary
    if frame_results:
        best_model_name = max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]
        if best_model_name in frame_results:
            best_frame_stats = frame_results[best_model_name]
            print(f"\n📊 Frame-Level Results (Best Model: {best_model_name}):")
            print(f"  • Total test frames: {best_frame_stats['total_frames']}")
            print(f"  • Correctly classified frames: {best_frame_stats['correct_frames']}")
            print(f"  • Incorrectly classified frames: {best_frame_stats['total_frames'] - best_frame_stats['correct_frames']}")
            print(f"  • Frame-level accuracy: {best_frame_stats['frame_accuracy']*100:.2f}%")
    
    print(f"\n🔧 Feature Breakdown (Per Frame):")
    print(f"  • Basic Image Features: 15 features (edges, colors, textures)")
    print(f"  • SqueezeNet CNN: 128 features (deep visual features)")
    print(f"  • Total: 143 features per frame")

if __name__ == "__main__":
    main() 