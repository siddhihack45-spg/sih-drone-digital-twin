import cv2
import numpy as np
from pathlib import Path
import csv
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================

FRAME_DIR = Path("data/frames")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "visual_trajectory.csv"
PLOT_PATH = OUTPUT_DIR / "visual_trajectory.png"


# =========================
# LOAD FRAMES
# =========================

frame_paths = sorted(FRAME_DIR.glob("frame_*.jpg"))

if len(frame_paths) < 2:
    raise RuntimeError("Not enough frames found.")


first_frame = cv2.imread(str(frame_paths[0]))

if first_frame is None:
    raise RuntimeError("Could not read first frame.")

height, width = first_frame.shape[:2]


# =========================
# CAMERA MATRIX
# =========================
# Temporary approximation.
# We will replace this with real camera calibration later.

fx = 0.9 * width
fy = 0.9 * width

cx = width / 2
cy = height / 2

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float64)


# =========================
# ORB
# =========================

orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


# =========================
# INITIAL STATE
# =========================

R_global = np.eye(3)
position = np.zeros(3)

trajectory = []

successful_pairs = 0
failed_pairs = 0

total_matches = 0
total_inliers = 0

min_inliers = 999999
max_inliers = 0

failed_pairs_list = []


print()
print("========== VISUAL TRAJECTORY ==========")
print()


# =========================
# PROCESS FRAME PAIRS
# =========================

for i in range(len(frame_paths) - 1):

    img1 = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(frame_paths[i + 1]), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        failed_pairs += 1
        failed_pairs_list.append(i)
        continue

    # -------------------------
    # Feature detection
    # -------------------------

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        failed_pairs += 1
        failed_pairs_list.append(i)
        continue

    # -------------------------
    # KNN matching
    # -------------------------

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    total_matches += len(good_matches)

    if len(good_matches) < 8:

        failed_pairs += 1
        failed_pairs_list.append(i)

        continue

    # -------------------------
    # Matched points
    # -------------------------

    pts1 = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    )

    pts2 = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    )

    # -------------------------
    # Essential matrix
    # -------------------------

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:

        failed_pairs += 1
        failed_pairs_list.append(i)

        continue

    # -------------------------
    # RANSAC inliers
    # -------------------------

    mask = mask.ravel().astype(bool)

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if len(pts1_in) < 8:

        failed_pairs += 1
        failed_pairs_list.append(i)

        continue

    # -------------------------
    # Recover camera motion
    # -------------------------

    pose_inliers, R, t, pose_mask = cv2.recoverPose(
        E,
        pts1_in,
        pts2_in,
        K
    )

    if pose_inliers < 8:

        failed_pairs += 1
        failed_pairs_list.append(i)

        continue

    # -------------------------
    # Accumulate trajectory
    # -------------------------

    t = t.reshape(3)

    position = position + R_global @ t

    R_global = R @ R_global

    trajectory.append(
        (
            i,
            position[0],
            position[1],
            position[2]
        )
    )

    successful_pairs += 1

    total_inliers += pose_inliers

    min_inliers = min(min_inliers, pose_inliers)
    max_inliers = max(max_inliers, pose_inliers)

    # -------------------------
    # Print progress
    # -------------------------

    if i % 10 == 0:

        print(
            f"Pair {i:3d} | "
            f"matches: {len(good_matches):4d} | "
            f"inliers: {pose_inliers:3d} | "
            f"position: "
            f"({position[0]:.3f}, "
            f"{position[1]:.3f}, "
            f"{position[2]:.3f})"
        )


# =========================
# DIAGNOSTICS
# =========================

total_pairs = len(frame_paths) - 1

success_rate = (
    successful_pairs / total_pairs * 100
    if total_pairs > 0
    else 0
)

average_matches = (
    total_matches / successful_pairs
    if successful_pairs > 0
    else 0
)

average_inliers = (
    total_inliers / successful_pairs
    if successful_pairs > 0
    else 0
)


print()
print("========== VO DIAGNOSTICS ==========")
print()

print(f"Total frame pairs       : {total_pairs}")
print(f"Successful pairs        : {successful_pairs}")
print(f"Failed pairs            : {failed_pairs}")
print(f"Success rate            : {success_rate:.1f}%")

print()

print(f"Average good matches    : {average_matches:.1f}")
print(f"Average pose inliers    : {average_inliers:.1f}")
print(f"Minimum pose inliers    : {min_inliers}")
print(f"Maximum pose inliers    : {max_inliers}")

print()

print("Failed pair IDs:")

if failed_pairs_list:

    print(failed_pairs_list)

else:

    print("None")

print()


# =========================
# SAVE CSV
# =========================

with open(
    CSV_PATH,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "frame_id",
        "x",
        "y",
        "z"
    ])

    for frame_id, x, y, z in trajectory:

        writer.writerow([
            frame_id,
            x,
            y,
            z
        ])


# =========================
# PLOT
# =========================

if trajectory:

    trajectory = np.array(trajectory)

    x = trajectory[:, 1]
    z = trajectory[:, 3]

    plt.figure(figsize=(8, 6))

    plt.plot(x, z)

    plt.xlabel("X")
    plt.ylabel("Z")

    plt.title("Monocular Visual Odometry Trajectory")

    plt.grid(True)

    plt.savefig(PLOT_PATH)

    plt.close()


# =========================
# COMPLETE
# =========================

print("========== COMPLETE ==========")

print(f"CSV  : {CSV_PATH}")
print(f"Plot : {PLOT_PATH}")
