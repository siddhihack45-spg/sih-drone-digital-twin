import os
import cv2
import numpy as np
import open3d as o3d


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "video",
    "dronevideo.mp4"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "output"
)

POINT_CLOUD_PATH = os.path.join(
    OUTPUT_DIR,
    "reconstruction.ply"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# Process every Nth frame.
# 2 means every second frame.
FRAME_STEP = 2

# Maximum number of depth samples per frame.
MAX_POINTS_PER_FRAME = 1500


# ============================================================
# CHECK VIDEO
# ============================================================

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video not found:")
    print(VIDEO_PATH)
    raise SystemExit(1)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit(1)


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print("========================================")
print("3D RECONSTRUCTION")
print("========================================")
print(f"Video frames : {total_frames}")
print(f"Resolution   : {width} x {height}")
print(f"Frame step   : {FRAME_STEP}")
print("========================================")


# ============================================================
# CAMERA PARAMETERS
# ============================================================

fx = width * 0.9
fy = width * 0.9

cx = width / 2.0
cy = height / 2.0


# ============================================================
# ORB
# ============================================================

orb = cv2.ORB_create(
    nfeatures=3000
)


# ============================================================
# POINT STORAGE
# ============================================================

all_points = []
all_colors = []


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_index = 0
processed = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    if frame_index % FRAME_STEP != 0:
        frame_index += 1
        continue

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    if keypoints is None or len(keypoints) == 0:
        frame_index += 1
        continue


    # --------------------------------------------------------
    # SAMPLE FEATURES
    # --------------------------------------------------------

    count = min(
        len(keypoints),
        MAX_POINTS_PER_FRAME
    )

    selected = keypoints[:count]


    # --------------------------------------------------------
    # CONVERT IMAGE FEATURES INTO APPROXIMATE 3D POINTS
    # --------------------------------------------------------
    #
    # This creates a preliminary sparse 3D representation.
    # True metric reconstruction requires camera poses and
    # depth/triangulation from multiple views.
    #
    # --------------------------------------------------------

    for kp in selected:

        u, v = kp.pt

        # Normalized image coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy

        # Preliminary depth
        #
        # This is intentionally approximate and will later be
        # replaced by triangulated depth from camera poses.

        z = 1.0

        X = x * z
        Y = y * z

        all_points.append([
            X,
            Y,
            z
        ])


        # OpenCV uses BGR
        px = int(
            max(
                0,
                min(width - 1, u)
            )
        )

        py = int(
            max(
                0,
                min(height - 1, v)
            )
        )

        b, g, r = frame[py, px]

        all_colors.append([
            r / 255.0,
            g / 255.0,
            b / 255.0
        ])


    processed += 1

    if processed % 25 == 0:

        print(
            f"Processed frame {frame_index + 1}/"
            f"{total_frames} "
            f"| Points: {len(all_points)}"
        )


    frame_index += 1


cap.release()


# ============================================================
# CHECK POINTS
# ============================================================

if len(all_points) == 0:

    print("ERROR: No 3D points generated.")
    raise SystemExit(1)


# ============================================================
# CREATE OPEN3D POINT CLOUD
# ============================================================

points_np = np.asarray(
    all_points,
    dtype=np.float64
)

colors_np = np.asarray(
    all_colors,
    dtype=np.float64
)


point_cloud = o3d.geometry.PointCloud()

point_cloud.points = o3d.utility.Vector3dVector(
    points_np
)

point_cloud.colors = o3d.utility.Vector3dVector(
    colors_np
)


# ============================================================
# REMOVE OUTLIERS
# ============================================================

print("Removing statistical outliers...")

point_cloud, indices = (
    point_cloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0
    )
)


# ============================================================
# SAVE PLY
# ============================================================

o3d.io.write_point_cloud(
    POINT_CLOUD_PATH,
    point_cloud
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("")
print("========================================")
print("3D RECONSTRUCTION COMPLETE")
print("========================================")
print(f"Frames processed : {processed}")
print(f"Final 3D points  : {len(point_cloud.points)}")
print("")
print(f"Point cloud:")
print(POINT_CLOUD_PATH)
print("========================================")
print("")
print("NOTE:")
print("This is a preliminary sparse reconstruction.")
print("The next improvement is true multi-view")
print("triangulation using the visual trajectory.")