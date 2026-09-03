import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = r"C:\Users\SIDDHI GORE\Desktop\sih\data\video\drone.mp4"

OUTPUT_DIR = r"C:\Users\SIDDHI GORE\Desktop\sih\output"

CSV_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "visual_trajectory.csv"
)

PLOT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "trajectory_plot.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("=" * 60)
print("DRONE VISUAL ODOMETRY")
print("=" * 60)

print(f"Video          : {VIDEO_PATH}")
print(f"Resolution     : {width} x {height}")
print(f"FPS            : {fps:.2f}")
print(f"Total Frames   : {total_frames}")
print("=" * 60)


# ============================================================
# CAMERA INTRINSIC PARAMETERS
# ============================================================
#
# These are APPROXIMATE values.
# For final SIH accuracy, replace these with calibrated
# camera parameters if available.
#

fx = width * 0.9
fy = width * 0.9

cx = width / 2.0
cy = height / 2.0

K = np.array([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
], dtype=np.float64)

print("\nCamera Matrix:")
print(K)


# ============================================================
# ORB FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    fastThreshold=20
)


# ============================================================
# FEATURE MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# READ FIRST FRAME
# ============================================================

ret, previous_frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read first frame.")


previous_gray = cv2.cvtColor(
    previous_frame,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# DETECT FEATURES IN FIRST FRAME
# ============================================================

previous_keypoints, previous_descriptors = orb.detectAndCompute(
    previous_gray,
    None
)

if previous_descriptors is None:
    cap.release()
    raise RuntimeError("No features detected in first frame.")


# ============================================================
# CAMERA POSE
# ============================================================

R_total = np.eye(3)

t_total = np.zeros(
    (3, 1),
    dtype=np.float64
)


# ============================================================
# TRAJECTORY STORAGE
# ============================================================

trajectory = []

trajectory.append([
    0,
    0.0,
    0.0,
    0.0
])


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_index = 1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Detect features
    # --------------------------------------------------------

    current_keypoints, current_descriptors = orb.detectAndCompute(
        gray,
        None
    )

    if current_descriptors is None:
        print(
            f"Frame {frame_index}: "
            "No features detected"
        )

        frame_index += 1
        continue


    # --------------------------------------------------------
    # Match descriptors
    # --------------------------------------------------------

    matches = bf.knnMatch(
        previous_descriptors,
        current_descriptors,
        k=2
    )


    # --------------------------------------------------------
    # Lowe's ratio test
    # --------------------------------------------------------

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good_matches.append(m)


    # --------------------------------------------------------
    # Need enough matches
    # --------------------------------------------------------

    if len(good_matches) < 8:

        print(
            f"Frame {frame_index}: "
            f"Not enough matches ({len(good_matches)})"
        )

        trajectory.append([
            frame_index,
            t_total[0, 0],
            t_total[1, 0],
            t_total[2, 0]
        ])

        previous_keypoints = current_keypoints
        previous_descriptors = current_descriptors

        frame_index += 1
        continue


    # --------------------------------------------------------
    # Extract matched points
    # --------------------------------------------------------

    points_previous = np.float32([
        previous_keypoints[m.queryIdx].pt
        for m in good_matches
    ])

    points_current = np.float32([
        current_keypoints[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # Essential Matrix
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        points_previous,
        points_current,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )


    if E is None:

        print(
            f"Frame {frame_index}: "
            "Essential matrix failed"
        )

        previous_keypoints = current_keypoints
        previous_descriptors = current_descriptors

        frame_index += 1
        continue


    # --------------------------------------------------------
    # Recover camera pose
    # --------------------------------------------------------

    _, R, t, pose_mask = cv2.recoverPose(
        E,
        points_previous,
        points_current,
        K
    )


    # --------------------------------------------------------
    # Accumulate pose
    # --------------------------------------------------------

    t_total = t_total + R_total @ t

    R_total = R_total @ R


    # --------------------------------------------------------
    # Save trajectory
    # --------------------------------------------------------

    x = float(t_total[0, 0])
    y = float(t_total[1, 0])
    z = float(t_total[2, 0])

    trajectory.append([
        frame_index,
        x,
        y,
        z
    ])


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_index % 25 == 0:

        percent = (
            frame_index /
            max(total_frames - 1, 1)
        ) * 100

        print(
            f"Frame {frame_index:4d}/{total_frames} "
            f"({percent:6.2f}%) | "
            f"Features: {len(current_keypoints):4d} | "
            f"Matches: {len(good_matches):4d} | "
            f"Position: "
            f"({x:.3f}, {y:.3f}, {z:.3f})"
        )


    # --------------------------------------------------------
    # Prepare next frame
    # --------------------------------------------------------

    previous_keypoints = current_keypoints
    previous_descriptors = current_descriptors

    frame_index += 1


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# SAVE TRAJECTORY CSV
# ============================================================

df = pd.DataFrame(
    trajectory,
    columns=[
        "frame",
        "x",
        "y",
        "z"
    ]
)

df.to_csv(
    CSV_OUTPUT,
    index=False
)


# ============================================================
# TRAJECTORY PLOT
# ============================================================

plt.figure(figsize=(10, 7))

plt.plot(
    df["x"],
    df["z"],
    linewidth=1.5
)

plt.scatter(
    df["x"].iloc[0],
    df["z"].iloc[0],
    s=80,
    label="Start"
)

plt.scatter(
    df["x"].iloc[-1],
    df["z"].iloc[-1],
    s=80,
    label="End"
)

plt.xlabel("X Position")
plt.ylabel("Z Position")

plt.title(
    "Drone Visual Odometry Trajectory"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOT_OUTPUT,
    dpi=200
)

plt.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("VISUAL ODOMETRY COMPLETED")
print("=" * 60)

print(f"Processed frames : {len(df)}")
print(f"CSV output       : {CSV_OUTPUT}")
print(f"Plot output      : {PLOT_OUTPUT}")

print("\nFirst 5 trajectory points:")
print(df.head())

print("\nLast 5 trajectory points:")
print(df.tail())

print("=" * 60)
