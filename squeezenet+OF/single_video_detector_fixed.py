"""
Single Video Scene Detector (Fixed Version)
This script detects the scene type of a single video using pre-trained models
It uses the same FRAME-LEVEL feature extraction methods as the training script for consistency
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import warnings
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

class SingleVideoFeatureExtractor:
    """
    Feature extractor for single videos using FRAME-LEVEL analysis
    Uses the same methods as training for consistency (143 features per frame)
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the feature extractor with the same setup as training
        
        Args:
            device: Computing device (cuda or cpu)
        """
        self.device = device
        self.setup_squeezenet()
        self.setup_transforms()
    
    def setup_squeezenet(self):
        """Load and prepare SqueezeNet for feature extraction (same as training)"""
        print(f"🧠 Loading SqueezeNet model on {self.device}...")
        
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        # Remove the classifier to get features only
        self.squeezenet.features = nn.Sequential(*list(self.squeezenet.features.children()))
        self.squeezenet.eval()
        self.squeezenet.to(self.device)
        
        # Freeze parameters for faster inference
        for param in self.squeezenet.parameters():
            param.requires_grad = False
        
        print("✅ SqueezeNet loaded successfully")
    
    def setup_transforms(self):
        """Setup image preprocessing for SqueezeNet (same as training)"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_frames_by_fps(self, video_path: str, target_fps: float = 2.0) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS (same method as training)
        
        Args:
            video_path: Path to the video file
            target_fps: Target frames per second to extract
            
        Returns:
            List of extracted frames
        """
        print(f"📹 Loading video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        print(f"  📊 Video info: {original_fps:.1f} FPS, {total_frames} frames, {duration:.1f}s")
        
        # Calculate frame sampling interval
        frame_interval = max(1, int(original_fps / target_fps))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames according to target FPS
            if frame_count % frame_interval == 0:
                # Resize frame for consistent processing (same as training)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        print(f"  ✅ Extracted {len(frames)} frames (sampling every {frame_interval} frames)")
        
        return frames
    
    def extract_features_from_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract features from each individual frame (EXACTLY SAME AS TRAINING)
        This is the key method that must match the training exactly
        
        Args:
            frames: List of video frames
            
        Returns:
            List of feature vectors (143 features per frame)
        """
        if not frames:
            return []
        
        print(f"🔍 Extracting frame-level features from {len(frames)} frames...")
        
        frame_features = []
        
        for i, frame in enumerate(frames):
            # Extract features for each frame individually (EXACTLY SAME AS TRAINING)
            
            # 1. Basic image features (simplified for single frame) - 15 features
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
            
            # 2. SqueezeNet features for single frame - 128 features
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        feature_map = self.squeezenet.features(input_tensor)
                else:
                    feature_map = self.squeezenet.features(input_tensor)
                
                # Global average pooling
                squeezenet_features = torch.mean(feature_map, dim=(2, 3)).cpu().numpy().flatten()
            
            # Combine all features for this frame (EXACTLY SAME AS TRAINING)
            combined_features = np.concatenate([
                basic_features,  # 15 features
                squeezenet_features[:128]  # 128 SqueezeNet features (reduced)
            ])  # Total: 143 features per frame
            
            frame_features.append(combined_features)
        
        print(f"✅ Extracted {len(frame_features)} frame feature vectors ({len(frame_features[0])} features each)")
        return frame_features


class SceneDetector:
    """
    Main scene detector class that loads trained models and makes predictions
    Uses frame-level analysis to match the training methodology
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize the scene detector with trained models
        
        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.preprocessors = {}
        self.scene_categories = ['beach', 'chaparral', 'forest', 'intersection', 'mountain', 'port']
        
        self.load_models_and_preprocessors()
        
        # Initialize feature extractor
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Using device: {device}")
        self.feature_extractor = SingleVideoFeatureExtractor(device=device)
    
    def load_models_and_preprocessors(self):
        """Load all trained models and preprocessing objects"""
        print("📦 Loading trained models and preprocessors...")
        
        # Define model filenames (same as saved in training)
        model_files = {
            'XGBoost': 'xgboost_model.pkl',
            'Gradient Boosting': 'gradient_boosting_model.pkl',
            'Random Forest': 'random_forest_model.pkl',
            'SVM': 'svm_model.pkl',
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Ensemble': 'ensemble_model.pkl'
        }
        
        # Load models
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"  ✅ {model_name} loaded")
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
            else:
                print(f"  ⚠️  {model_name} not found at {model_path}")
        
        # Load preprocessors
        preprocessor_files = {
            'scaler': 'scaler.pkl',
            'label_encoder': 'label_encoder.pkl',
            'feature_selector': 'feature_selector.pkl'
        }
        
        for prep_name, filename in preprocessor_files.items():
            prep_path = self.models_dir / filename
            if prep_path.exists():
                try:
                    self.preprocessors[prep_name] = joblib.load(prep_path)
                    print(f"  ✅ {prep_name} loaded")
                except Exception as e:
                    print(f"  ❌ Failed to load {prep_name}: {e}")
            else:
                print(f"  ⚠️  {prep_name} not found at {prep_path}")
        
        # Check if we have minimum required components
        if not self.models:
            raise ValueError("No models were loaded successfully!")
        
        required_preprocessors = ['scaler', 'label_encoder', 'feature_selector']
        missing = [p for p in required_preprocessors if p not in self.preprocessors]
        if missing:
            raise ValueError(f"Missing required preprocessors: {missing}")
        
        print(f"🎯 Successfully loaded {len(self.models)} models and {len(self.preprocessors)} preprocessors")
    
    def analyze_video(self, video_path: str, show_details: bool = True) -> Dict:
        """
        Complete video analysis with detailed results using frame-level analysis
        
        Args:
            video_path: Path to the video file
            show_details: Whether to show detailed analysis
            
        Returns:
            Complete analysis results
        """
        print(f"\n🎬 Analyzing video: {os.path.basename(video_path)}")
        print("=" * 50)
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Step 1: Extract frames
        print("🔍 Step 1: Extracting frames...")
        frames = self.feature_extractor.extract_frames_by_fps(video_path, target_fps=2.0)
        
        if len(frames) < 1:
            raise ValueError(f"No frames extracted from video: {video_path}")
        
        # Step 2: Extract features from frames
        print("🔍 Step 2: Extracting features from frames...")
        frame_features = self.feature_extractor.extract_features_from_frames(frames)
        
        if not frame_features:
            raise ValueError(f"No features extracted from video: {video_path}")
        
        # Convert to numpy array
        features_matrix = np.array(frame_features)  # Shape: (N, 143)
        print(f"  📊 Feature matrix shape: {features_matrix.shape}")
        
        # Step 3: Preprocess features
        print("🔧 Step 3: Preprocessing features...")
        
        # Handle problematic values (same as training)
        features_matrix = np.nan_to_num(features_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Apply scaling (same as training)
        features_scaled = self.preprocessors['scaler'].transform(features_matrix)
        
        # Apply feature selection (same as training)
        processed_features = self.preprocessors['feature_selector'].transform(features_scaled)
        print(f"  📊 Processed feature matrix shape: {processed_features.shape}")
        
        # Step 4: Make predictions with all models
        print("🤖 Step 4: Making predictions...")
        
        frame_predictions = {}
        for model_name, model in self.models.items():
            try:
                # Get predictions for all frames
                if hasattr(model, 'predict_proba'):
                    frame_probabilities = model.predict_proba(processed_features)  # Shape: (n_frames, n_classes)
                    frame_predicted_classes = np.argmax(frame_probabilities, axis=1)
                    frame_confidences = np.max(frame_probabilities, axis=1)
                else:
                    # For models without predict_proba
                    frame_predicted_classes = model.predict(processed_features)
                    frame_probabilities = None
                    frame_confidences = None
                
                # Convert class indices to scene names
                frame_scene_names = self.preprocessors['label_encoder'].inverse_transform(frame_predicted_classes)
                
                # Aggregate predictions across frames (majority vote)
                unique_scenes, scene_counts = np.unique(frame_scene_names, return_counts=True)
                most_common_idx = np.argmax(scene_counts)
                consensus_scene = unique_scenes[most_common_idx]
                consensus_count = scene_counts[most_common_idx]
                
                # Calculate average confidence for the consensus scene
                if frame_confidences is not None:
                    consensus_mask = frame_scene_names == consensus_scene
                    avg_confidence = np.mean(frame_confidences[consensus_mask])
                else:
                    avg_confidence = consensus_count / len(frame_scene_names)
                
                frame_predictions[model_name] = {
                    'scene': consensus_scene,
                    'confidence': avg_confidence,
                    'frame_votes': consensus_count,
                    'total_frames': len(frame_scene_names),
                    'vote_percentage': consensus_count / len(frame_scene_names)
                }
                
                print(f"  {model_name:18}: {consensus_scene} ({consensus_count}/{len(frame_scene_names)} frames, {avg_confidence:.3f} conf)")
                
            except Exception as e:
                print(f"  ❌ {model_name} prediction failed: {e}")
                frame_predictions[model_name] = None
        
        # Get overall consensus
        scene_votes = {}
        for model_name, pred in frame_predictions.items():
            if pred is not None:
                scene = pred['scene']
                confidence = pred['confidence']
                
                if scene not in scene_votes:
                    scene_votes[scene] = {'count': 0, 'total_confidence': 0}
                
                scene_votes[scene]['count'] += 1
                scene_votes[scene]['total_confidence'] += confidence
        
        if scene_votes:
            consensus_scene = max(scene_votes.items(), key=lambda x: x[1]['count'])[0]
            consensus_confidence = scene_votes[consensus_scene]['total_confidence'] / scene_votes[consensus_scene]['count']
        else:
            consensus_scene, consensus_confidence = "unknown", 0.0
        
        results = {
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
            'predictions': frame_predictions,
            'consensus': {
                'scene': consensus_scene,
                'confidence': consensus_confidence
            }
        }
        
        if show_details:
            print("\n" + "=" * 50)
            print("🎯 FINAL RESULTS")
            print("=" * 50)
            print(f"📹 Video: {os.path.basename(video_path)}")
            print(f"🏆 Consensus Prediction: {consensus_scene.upper()}")
            print(f"📊 Consensus Confidence: {consensus_confidence*100:.1f}%")
            
            # Show vote breakdown
            print(f"\n📊 Model Vote Breakdown:")
            for scene, votes in sorted(scene_votes.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"  {scene:12}: {votes['count']} models")
            
            # Show frame-level details
            print(f"\n📊 Frame-Level Analysis:")
            for model_name, pred in frame_predictions.items():
                if pred is not None:
                    scene = pred['scene']
                    frame_votes = pred['frame_votes']
                    total_frames = pred['total_frames']
                    percentage = pred['vote_percentage'] * 100
                    print(f"  {model_name:18}: {frame_votes}/{total_frames} frames ({percentage:.1f}%) voted for {scene}")
        
        return results


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Detect scene type in a single video using frame-level analysis')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--models-dir', default='models', help='Directory containing trained models')
    parser.add_argument('--quiet', action='store_true', help='Show only final results')
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = SceneDetector(models_dir=args.models_dir)
        
        # Analyze video
        results = detector.analyze_video(args.video_path, show_details=not args.quiet)
        
        # Return success
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 