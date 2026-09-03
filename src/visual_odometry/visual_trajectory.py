import cv2
import numpy as np
import os
import glob
import csv
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FRAMES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "frames"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "visual_trajectory.csv"
)

PLOT_PATH = os.path.join(
    OUTPUT_DIR,
    "trajectory_plot.png"
)

# ============================================================
# LOAD FRAMES
# ============================================================

frame_paths = sorted(
    glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))
)

if len(frame_paths) < 2:
    print("ERROR: Need at least 2 frames.")
    print(f"Found: {len(frame_paths)}")
    raise SystemExit(1)

print("========================================")
print("VISUAL ODOMETRY")
print("========================================")
print(f"Frames found : {len(frame_paths)}")
print("========================================")

# ============================================================
# READ FIRST FRAME
# ============================================================

first_frame = cv2.imread(frame_paths[0])

if first_frame is None:
    print("ERROR: Could not read first frame.")
    raise SystemExit(1)

height, width = first_frame.shape[:2]

# Approximate camera intrinsics
fx = width * 0.9
fy = width * 0.9
cx = width / 2
cy = height / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)

print(f"Frame size  : {width} x {height}")
print(f"Camera fx   : {fx:.2f}")
print(f"Camera fy   : {fy:.2f}")

# ============================================================
# ORB FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8
)

# ============================================================
# FIRST FRAME
# ============================================================

prev_color = first_frame
prev_gray = cv2.cvtColor(
    prev_color,
    cv2.COLOR_BGR2GRAY
)

prev_keypoints, prev_descriptors = orb.detectAndCompute(
    prev_gray,
    None
)

if prev_descriptors is None:
    print("ERROR: No features found in first frame.")
    raise SystemExit(1)

# ============================================================
# CAMERA POSE
# ============================================================

R_total = np.eye(3)
t_total = np.zeros((3, 1))

trajectory = []

trajectory.append([
    0,
    0.0,
    0.0,
    0.0
])

# ============================================================
# FEATURE MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# ============================================================
# PROCESS ALL FRAMES
# ============================================================

successful_pairs = 0

for i in range(1, len(frame_paths)):

    current_color = cv2.imread(frame_paths[i])

    if current_color is None:
        print(f"WARNING: Could not read {frame_paths[i]}")
        continue

    current_gray = cv2.cvtColor(
        current_color,
        cv2.COLOR_BGR2GRAY
    )

    current_keypoints, current_descriptors = orb.detectAndCompute(
        current_gray,
        None
    )

    if current_descriptors is None:
        print(f"Frame {i}: No descriptors")
        trajectory.append([
            i,
            float(t_total[0]),
            float(t_total[1]),
            float(t_total[2])
        ])
        prev_gray = current_gray
        prev_keypoints = current_keypoints
        prev_descriptors = current_descriptors
        continue

    # --------------------------------------------------------
    # MATCH FEATURES
    # --------------------------------------------------------

    if prev_descriptors is None:
        prev_gray = current_gray
        prev_keypoints = current_keypoints
        prev_descriptors = current_descriptors
        continue

    matches = bf.knnMatch(
        prev_descriptors,
        current_descriptors,
        k=2
    )

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # --------------------------------------------------------
    # CHECK MATCH COUNT
    # --------------------------------------------------------

    if len(good_matches) < 8:

        trajectory.append([
            i,
            float(t_total[0]),
            float(t_total[1]),
            float(t_total[2])
        ])

        prev_gray = current_gray
        prev_keypoints = current_keypoints
        prev_descriptors = current_descriptors

        if i % 25 == 0:
            print(
                f"Frame {i}/{len(frame_paths)-1} "
                f"| Good matches: {len(good_matches)}"
            )

        continue

    # --------------------------------------------------------
    # GET MATCHED POINTS
    # --------------------------------------------------------

    pts_prev = np.float32([
        prev_keypoints[m.queryIdx].pt
        for m in good_matches
    ])

    pts_current = np.float32([
        current_keypoints[m.trainIdx].pt
        for m in good_matches
    ])

    # --------------------------------------------------------
    # ESSENTIAL MATRIX
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        pts_current,
        pts_prev,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is not None:

        try:

            _, R, t, pose_mask = cv2.recoverPose(
                E,
                pts_current,
                pts_prev,
                K
            )

            # ------------------------------------------------
            # UPDATE CAMERA POSITION
            # ------------------------------------------------

            t_total = t_total + R_total @ t

            R_total = R @ R_total

            successful_pairs += 1

        except cv2.error:
            pass

    # --------------------------------------------------------
    # SAVE TRAJECTORY POINT
    # --------------------------------------------------------

    trajectory.append([
        i,
        float(t_total[0]),
        float(t_total[1]),
        float(t_total[2])
    ])

    # --------------------------------------------------------
    # UPDATE PREVIOUS FRAME
    # --------------------------------------------------------

    prev_gray = current_gray
    prev_keypoints = current_keypoints
    prev_descriptors = current_descriptors

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if i % 25 == 0:

        print(
            f"Frame {i}/{len(frame_paths)-1} "
            f"| Good matches: {len(good_matches)} "
            f"| Successful poses: {successful_pairs}"
        )

# ============================================================
# SAVE CSV
# ============================================================

with open(
    CSV_PATH,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "frame",
        "x",
        "y",
        "z"
    ])

    writer.writerows(trajectory)

# ============================================================
# TRAJECTORY PLOT
# ============================================================

trajectory_np = np.array(
    trajectory,
    dtype=np.float64
)

plt.figure(figsize=(10, 7))

plt.plot(
    trajectory_np[:, 1],
    trajectory_np[:, 3],
    linewidth=2
)

plt.xlabel("X")
plt.ylabel("Z")

plt.title(
    "Drone Visual Odometry Trajectory"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=150
)

plt.close()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("")
print("========================================")
print("VISUAL ODOMETRY COMPLETE")
print("========================================")
print(f"Frames processed : {len(frame_paths)}")
print(f"Trajectory rows  : {len(trajectory)}")
print(f"Successful poses  : {successful_pairs}")
print("")
print(f"CSV  : {CSV_PATH}")
print(f"Plot : {PLOT_PATH}")
print("========================================")
print("")
print("NOTE:")
print("Monocular visual odometry does not provide")
print("absolute real-world scale by itself.")
print("GPS + IMU fusion will be used later.")