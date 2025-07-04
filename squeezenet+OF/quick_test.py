#!/usr/bin/env python3
"""
Quick test script to verify the lightweight scene analysis setup
This script checks if everything is working properly before we start training
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

def check_dependencies():
    """
    Check if all required Python packages are installed
    
    What it does: Tries to import all the packages we need for the project
    Why important: If packages are missing, the training will fail
    Simple explanation: Like checking if you have all ingredients before cooking
    """
    try:
        import cv2
        import torch
        import torchvision
        import xgboost
        import sklearn
        import pandas
        import matplotlib
        import seaborn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: poetry install")
        return False

def check_dataset():
    """
    Check if video dataset folders and files exist
    
    What it does: Looks for the dataset folder and counts videos in each category
    Why important: We need videos to train the AI model
    Simple explanation: Like checking if you have the photos before sorting them
    """
    dataset_path = "dataset"
    categories = ['beach', 'chaparral', 'forest', 'intersection', 'mountain', 'port']
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder '{dataset_path}' not found")
        return False
    
    total_videos = 0
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        if not os.path.exists(category_path):
            print(f"❌ Category folder '{category}' not found")
            return False
        
        video_count = 0
        for i in range(1, 16):  # Now checking for 15 videos (01-15)
            video_name = f"{category}{i:02d}.mp4"
            video_path = os.path.join(category_path, video_name)
            if os.path.exists(video_path):
                video_count += 1
        
        print(f"📁 {category}: {video_count}/15 videos found")
        total_videos += video_count
    
    print(f"📊 Total videos: {total_videos}/90")
    return total_videos > 0

def test_video_loading():
    """
    Test if we can open and read video files properly
    
    What it does: Finds a video file and tries to open it to read frames
    Why important: If we can't read videos, we can't extract features from them
    Simple explanation: Like testing if you can open a book before reading it
    """
    print("\n🎬 Testing video loading...")
    
    # Find first available video
    dataset_path = "dataset"
    categories = ['beach', 'chaparral', 'forest', 'intersection', 'mountain', 'port']
    
    test_video = None
    for category in categories:
        for i in range(1, 16):  # Now checking for 15 videos (01-15)
            video_name = f"{category}{i:02d}.mp4"
            video_path = os.path.join(dataset_path, category, video_name)
            if os.path.exists(video_path):
                test_video = video_path
                break
        if test_video:
            break
    
    if not test_video:
        print("❌ No test video found")
        return False
    
    print(f"🎥 Testing with: {test_video}")
    
    # Test video loading
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {test_video}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"📊 Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Test frame reading
    ret, frame = cap.read()
    if ret:
        print(f"✅ Frame loaded: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ Cannot read frame")
        cap.release()
        return False

def test_pytorch():
    """
    Test if PyTorch and neural networks are working properly
    
    What it does: Checks if PyTorch can use GPU/CPU and load a pre-trained model
    Why important: We need PyTorch to extract deep learning features from videos
    Simple explanation: Like testing if your calculator works before doing math
    """
    print("\n🔥 Testing PyTorch...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    
    if device == 'cuda':
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test model loading
    try:
        from torchvision import models
        model = models.squeezenet1_1(pretrained=True)
        print("✅ SqueezeNet loaded successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: {output.shape}")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_feature_extraction():
    """
    Test if we can extract features from video frames
    
    What it does: Creates fake video frames and tests all feature extraction methods
    Why important: These features are what the AI learns from to classify scenes
    Simple explanation: Like testing if you can measure ingredients before cooking
    """
    print("\n🧠 Testing feature extraction...")
    
    try:
        # Create dummy frames (fake video frames for testing)
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
        
        # Test optical flow with proper method
        gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        
        # Use corner detection for Lucas-Kanade optical flow
        # This finds moving objects between two frames
        corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
        
        if corners is not None and len(corners) > 0:
            # Lucas-Kanade optical flow - tracks how things move in the video
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
            print("✅ Lucas-Kanade optical flow computation successful")
        else:
            print("⚠️  No corners detected, testing dense optical flow instead")
        
        # Test dense optical flow (Farneback method) as fallback
        # This calculates movement for every pixel in the image
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        print("✅ Dense optical flow (Farneback) computation successful")
        
        # Test edge detection - finds boundaries and edges in images
        edges = cv2.Canny(gray1, 50, 150)
        print("✅ Edge detection successful")
        
        # Test color histograms - analyzes what colors are in the image
        hist = cv2.calcHist([frames[0]], [0], None, [256], [0, 256])
        print("✅ Color histogram computation successful")
        
        # Test texture analysis - measures how rough or smooth surfaces look
        laplacian = cv2.Laplacian(gray1, cv2.CV_64F)
        texture_var = np.var(laplacian)
        print("✅ Texture analysis successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def main():
    """
    Run all tests to make sure everything is working
    
    What it does: Runs all the individual tests and shows a summary
    Why important: Catches problems early before we waste time training
    Simple explanation: Like doing a checklist before starting a project
    """
    print("🧪 Lightweight Scene Analysis Quick Test")
    print("=" * 45)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Dataset", check_dataset),
        ("Video Loading", test_video_loading),
        ("PyTorch", test_pytorch),
        ("Feature Extraction", test_feature_extraction)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 45)
    print("📊 TEST SUMMARY")
    print("=" * 45)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name:18}: {status}")
    
    passed = sum(results)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run training.")
        print("\nNext steps:")
        print("1. Run: poetry run python lightweight_scene_analyzer.py")
        print("   (Comprehensive model comparison with 6 algorithms)")
        print("2. Or run: poetry run python scene_analysis_trainer.py")
        print("   (Basic XGBoost + Gradient Boosting)")
        print("\nLightweight Features:")
        print("• 🔄 Advanced Optical Flow (80 features)")
        print("• 🧠 SqueezeNet CNN (256 features)")  
        print("• 🖼️  Image Processing (60 features)")
        print("• 📊 Total: 396 features")
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 