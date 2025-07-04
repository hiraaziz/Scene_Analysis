import cv2
import numpy as np
import argparse

def create_comparison_video(original_path, enhanced_path, output_path='enhancement_comparison.mp4'):
    """
    Create a side-by-side comparison video between original and enhanced versions.
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_enh = cv2.VideoCapture(enhanced_path)
    if not cap_orig.isOpened():
        raise ValueError(f"Could not open original video: {original_path}")
    if not cap_enh.isOpened():
        raise ValueError(f"Could not open enhanced video: {enhanced_path}")
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    frame_count = 0
    print("Creating side-by-side comparison video...")
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_enh, frame_enh = cap_enh.read()
        if not ret_orig or not ret_enh:
            break
        cv2.putText(frame_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_enh, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if frame_enh.shape[0] != frame_orig.shape[0] or frame_enh.shape[1] != frame_orig.shape[1]:
            frame_enh = cv2.resize(frame_enh, (frame_orig.shape[1], frame_orig.shape[0]))
        comparison_frame = np.hstack((frame_orig, frame_enh))
        out.write(comparison_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames for comparison")
    cap_orig.release()
    cap_enh.release()
    out.release()
    print(f"Comparison video saved to: {output_path}")
    return output_path

def compute_quality_metrics(original_path, enhanced_path):
    """
    Compute quality metrics comparing original and enhanced videos. No graphing, only numeric output.
    """
    cap_orig = cv2.VideoCapture(original_path)
    cap_enh = cv2.VideoCapture(enhanced_path)
    if not cap_orig.isOpened() or not cap_enh.isOpened():
        raise ValueError("Could not open one or both videos for comparison")
    orig_brightness = []
    enh_brightness = []
    orig_contrast = []
    enh_contrast = []
    orig_sharpness = []
    enh_sharpness = []
    frame_count = 0
    sample_interval = 10
    print("Computing quality metrics...")
    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_enh, frame_enh = cap_enh.read()
        if not ret_orig or not ret_enh:
            break
        frame_count += 1
        if frame_count % sample_interval == 0:
            lab_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2LAB)
            lab_enh = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2LAB)
            l_orig, _, _ = cv2.split(lab_orig)
            l_enh, _, _ = cv2.split(lab_enh)
            orig_brightness.append(np.mean(l_orig))
            enh_brightness.append(np.mean(l_enh))
            orig_contrast.append(np.std(l_orig))
            enh_contrast.append(np.std(l_enh))
            gray_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
            gray_enh = cv2.cvtColor(frame_enh, cv2.COLOR_BGR2GRAY)
            orig_sharpness.append(cv2.Laplacian(gray_orig, cv2.CV_64F).var())
            enh_sharpness.append(cv2.Laplacian(gray_enh, cv2.CV_64F).var())
    cap_orig.release()
    cap_enh.release()
    orig_brightness = np.array(orig_brightness)
    enh_brightness = np.array(enh_brightness)
    orig_contrast = np.array(orig_contrast)
    enh_contrast = np.array(enh_contrast)
    orig_sharpness = np.array(orig_sharpness)
    enh_sharpness = np.array(enh_sharpness)
    brightness_improvement = ((np.mean(enh_brightness) - np.mean(orig_brightness)) / np.mean(orig_brightness)) * 100
    contrast_improvement = ((np.mean(enh_contrast) - np.mean(orig_contrast)) / np.mean(orig_contrast)) * 100
    sharpness_improvement = ((np.mean(enh_sharpness) - np.mean(orig_sharpness)) / np.mean(orig_sharpness)) * 100
    print(f"\n📊 Enhancement Quality Analysis:")
    print(f"   Brightness: Original {np.mean(orig_brightness):.1f} → Enhanced {np.mean(enh_brightness):.1f} ({brightness_improvement:+.1f}%)")
    print(f"   Contrast:   Original {np.mean(orig_contrast):.1f} → Enhanced {np.mean(enh_contrast):.1f} ({contrast_improvement:+.1f}%)")
    print(f"   Sharpness:  Original {np.mean(orig_sharpness):.0f} → Enhanced {np.mean(enh_sharpness):.0f} ({sharpness_improvement:+.1f}%)")
    print(f"   📈 Analyzed {len(orig_brightness)} sample frames")
    return {
        'brightness_improvement': brightness_improvement,
        'contrast_improvement': contrast_improvement, 
        'sharpness_improvement': sharpness_improvement,
        'original_brightness': orig_brightness,
        'enhanced_brightness': enh_brightness,
        'original_contrast': orig_contrast,
        'enhanced_contrast': enh_contrast,
        'original_sharpness': orig_sharpness,
        'enhanced_sharpness': enh_sharpness
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video comparison and quality metrics utilities")
    parser.add_argument("--compare", action="store_true", help="Create comparison video")
    parser.add_argument("--metrics", action="store_true", help="Compute quality metrics")
    parser.add_argument("--original", type=str, help="Path to original video")
    parser.add_argument("--enhanced", type=str, help="Path to enhanced video")
    parser.add_argument("--output", type=str, default="enhancement_comparison.mp4", help="Output path for comparison video")
    args = parser.parse_args()
    if args.compare:
        if not args.original or not args.enhanced:
            print("--original and --enhanced are required for comparison video.")
        else:
            create_comparison_video(args.original, args.enhanced, args.output)
    if args.metrics:
        if not args.original or not args.enhanced:
            print("--original and --enhanced are required for metrics computation.")
        else:
            compute_quality_metrics(args.original, args.enhanced) 