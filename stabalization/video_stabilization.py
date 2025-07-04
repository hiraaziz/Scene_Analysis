import numpy as np
import cv2
from google.colab.patches import cv2_imshow

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

# Parameters
SMOOTHING_RADIUS = 50
input_path = 'test2.avi'
output_path = 'video_out.avi'

# Initialize video capture
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise ValueError(f"Could not open video file {input_path}")

# Video properties
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video info: {n_frames} frames, {w}x{h}, {fps} fps")

# Define codec and output writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))  # Side-by-side output

# Read first frame
success, prev = cap.read()
if not success:
    raise ValueError("Could not read first frame")
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Pre-allocate transform matrix
transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 2):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    success, curr = cap.read()
    if not success:
        break

    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    assert prev_pts.shape == curr_pts.shape

    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    if m is None:
        print(f"Warning: Could not estimate transform for frame {i}")
        transforms[i] = [0, 0, 0]
        continue

    dx = m[0, 2]
    dy = m[1, 2]
    da = np.arctan2(m[1, 0], m[0, 0])
    transforms[i] = [dx, dy, da]

    prev_gray = curr_gray
    print(f"Frame {i}/{n_frames} processed")

# Calculate trajectory and smooth it
trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

# Reset to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Write stabilized frames
for i in range(n_frames - 2):
    success, frame = cap.read()
    if not success:
        break

    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    frame_stabilized = cv2.warpAffine(frame, m, (w, h))
    frame_stabilized = fixBorder(frame_stabilized)

    if i % 10 == 0:
        cv2_imshow(frame_stabilized)  # Optional

    out.write(frame_stabilized)

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete! Stabilized video saved.")
