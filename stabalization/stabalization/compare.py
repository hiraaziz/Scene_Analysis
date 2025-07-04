import matplotlib.pyplot as plt
import cv2
import numpy as np

def compute_jitter(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    motion = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=1, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion.append(np.mean(mag))
        prev_gray = gray

    cap.release()
    return np.mean(motion), motion

def compare_videos(original_path, stabilized_path):
    original_avg, original_motion = compute_jitter(original_path)
    stabilized_avg, stabilized_motion = compute_jitter(stabilized_path)
    improvement = ((original_avg - stabilized_avg) / original_avg) * 100

    print(f"📊 Original avg jitter: {original_avg:.4f}")
    print(f"📊 Stabilized avg jitter: {stabilized_avg:.4f}")
    print(f"✅ Improvement: {improvement:.2f}%")

    plt.figure(figsize=(10, 4))
    plt.plot(original_motion, label="Original")
    plt.plot(stabilized_motion, label="Stabilized")
    plt.title("Jitter (motion magnitude) per frame")
    plt.xlabel("Frame")
    plt.ylabel("Mean Optical Flow Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    compare_videos("test11.mp4", "orb_stabilized.avi")

if __name__ == "__main__":
    main()