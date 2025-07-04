poetry run python single_video_detector_fixed.py video-14.mp4

# Lightweight Video Scene Analysis with SqueezeNet + Optical Flow

A lightweight, high-accuracy video scene classification system that combines advanced optical flow, SqueezeNet CNN features, and traditional image processing. The system uses a comprehensive workflow with stratified video splitting, frame-level feature extraction, and comparison of 6 machine learning models.

## 🚀 Complete Workflow (as in `lightweight_scene_analyzer.py`)

### Step-by-Step Pipeline

1. **Stratified Video Split**: Evenly split all videos into training and test sets per category (default: 12 train, 3 test per category).
2. **Frame-Level Feature Extraction**: For each video, sample frames at 2 FPS and extract features from every frame:
   - **Basic Image Features** (edges, color, texture, HSV, etc.)
   - **SqueezeNet CNN Features** (deep visual features)
   - **Combined: 143 features per frame**
3. **Data Preparation**: Aggregate all frame features, encode labels, scale features, and select relevant features.
4. **Model Training**: Train 6 models on the frame-level data:
   - XGBoost
   - Gradient Boosting
   - Random Forest
   - SVM
   - Logistic Regression
   - Ensemble (Voting of XGB, GB, RF)
5. **Cross-Validation**: Evaluate each model with 5-fold stratified cross-validation.
6. **Evaluation & Reporting**: Generate detailed classification reports, per-class and per-frame analysis.
7. **Visualization**: Create confusion matrices and accuracy comparison charts.
8. **Model Saving**: Save all trained models and preprocessing tools for later use.

### Main Function Workflow

- Initializes feature extractor, video processor, and trainer.
- Splits dataset into train/test sets (stratified by category).
- Extracts frame-level features from all training and test videos.
- Prepares data (scaling, encoding, feature selection).
- Trains and evaluates all models.
- Performs cross-validation and generates reports.
- Visualizes results and saves all models.

---

## 🎯 Key Features

- **Lightweight Design**: Only SqueezeNet CNN + optical flow + image processing (no heavy deep learning)
- **Comprehensive Model Comparison**: 6 different algorithms (XGBoost, Gradient Boosting, Random Forest, SVM, Logistic Regression, Ensemble)
- **High Performance**: Optimized for >93% accuracy with fast processing
- **Multi-modal Features**: 396 total features combining motion, appearance, and texture
- **Poetry Integration**: Modern dependency management with Poetry

## 📊 Feature Breakdown

### Frame-Level Features (per frame)

- **Basic Image Features**: 15 (edges, color, texture, HSV, etc.)
- **SqueezeNet CNN Features**: 128 (deep features)
- **Total**: 143 features per frame

### Video-Level Features (for full video, not default in main workflow)

- **Optical Flow**: 80
- **SqueezeNet CNN**: 256
- **Image Processing**: 60
- **Total**: 396 features per video

## 📁 Dataset Structure

```
dataset/
├── beach/          # Beach scenes (15 videos)
├── chaparral/      # Chaparral scenes (15 videos)
├── forest/         # Forest scenes (15 videos)
├── intersection/   # Intersection scenes (15 videos)
├── mountain/       # Mountain scenes (15 videos)
└── port/           # Port scenes (15 videos)
```

Total: 90 videos across 6 scene categories

## 🏗️ Project Files

- `lightweight_scene_analyzer.py` - Main training and evaluation script (full workflow)
- `squeezent+of/single_video_detector.py` - Single video scene detection (inference)
- `example_usage.py` - Example usage of the single video detector
- `test_detector.py` - Test suite for the single video detector
- `run_training.py` - Automated setup and training script
- `scene_analysis_trainer.py` - Basic training script
- `quick_test.py` - System test script

### Output Files

- `models/` - Saved models and preprocessing tools
- `comprehensive_model_comparison.png` - Confusion matrices for all models
- `accuracy_comparison.png` - Bar chart comparing model accuracies

## 🔄 Workflow Diagram

```mermaid
graph TD
    A[90 Videos] --> B[Stratified Train/Test Split]
    B --> C[Frame Sampling (2 FPS)]
    C --> D[Frame-Level Feature Extraction]
    D --> E[143 Features per Frame]
    E --> F[Data Preparation (Scaling, Encoding, Selection)]
    F --> G[6 Model Training]
    G --> H[Cross-Validation]
    H --> I[Evaluation & Reporting]
    I --> J[Visualization]
    J --> K[Model Saving]
```

## 🖥️ Usage

### 1. Install Poetry (if not installed)

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Run Full Training Pipeline

```bash
poetry run python lightweight_scene_analyzer.py
```

- This will split the dataset, extract frame-level features, train all models, evaluate, visualize, and save everything.

### 4. Single Video Detection (after training)

```bash
python single_video_detector_fixed.py path/to/your/video.mp4
```

## 🤖 Model Comparison

| Model                   | Type              | Strengths                                        |
| ----------------------- | ----------------- | ------------------------------------------------ |
| **XGBoost**             | Gradient Boosting | Fast, handles missing values, feature importance |
| **Gradient Boosting**   | Ensemble          | Robust, good generalization                      |
| **Random Forest**       | Ensemble          | Handles overfitting, parallel training           |
| **SVM**                 | Kernel Method     | Good with high-dimensional data                  |
| **Logistic Regression** | Linear            | Fast, interpretable baseline                     |
| **Ensemble**            | Meta-classifier   | Combines best of XGB + GB + RF                   |

## 📈 Expected Performance

- **Accuracy**: >93% on test set (frame-level)
- **Processing Speed**: ~3-4 videos per second
- **Training Time**: ~5-10 minutes for full dataset
- **Memory Usage**: <4GB RAM

## 🐛 Troubleshooting

- **Low Accuracy**: Increase frame sampling rate, check dataset balance, verify all videos are present.
- **Slow Processing**: Reduce number of workers, lower frame sampling rate.
- **Memory Issues**: Lower max frames for CNN, process sequentially.

## 📚 Example: Custom Configuration

```python
from lightweight_scene_analyzer import LightweightFeatureExtractor, OptimizedVideoProcessor

extractor = LightweightFeatureExtractor(device='cpu')
processor = OptimizedVideoProcessor('dataset', extractor)
features = processor.extract_features_parallel(n_workers=2)
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for SqueezeNet architecture
- OpenCV community for computer vision algorithms
- Scikit-learn for machine learning tools
- XGBoost team for gradient boosting implementation
