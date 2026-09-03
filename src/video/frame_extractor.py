import cv2
import os
import glob

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "video",
    "dronevideo.mp4"
)

FRAMES_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "frames"
)

# --------------------------------------------------
# CHECK VIDEO
# --------------------------------------------------

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video not found!")
    print(VIDEO_PATH)
    raise SystemExit(1)

# --------------------------------------------------
# CREATE / CLEAN FRAMES FOLDER
# --------------------------------------------------

os.makedirs(FRAMES_DIR, exist_ok=True)

old_frames = glob.glob(os.path.join(FRAMES_DIR, "*.jpg"))

for frame in old_frames:
    os.remove(frame)

print(f"Removed {len(old_frames)} old frames.")

# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    raise SystemExit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("----------------------------------------")
print("VIDEO INFORMATION")
print("----------------------------------------")
print(f"Video       : {VIDEO_PATH}")
print(f"Expected frames : {total_frames}")
print(f"FPS         : {fps:.2f}")
print(f"Resolution  : {width} x {height}")
print("----------------------------------------")

# --------------------------------------------------
# EXTRACT FRAMES
# --------------------------------------------------

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    filename = os.path.join(
        FRAMES_DIR,
        f"frame_{frame_number:06d}.jpg"
    )

    cv2.imwrite(filename, frame)

    frame_number += 1

    if frame_number % 50 == 0:
        print(f"Extracted {frame_number} frames...")

# --------------------------------------------------
# RELEASE VIDEO
# --------------------------------------------------

cap.release()

# --------------------------------------------------
# FINAL RESULT
# --------------------------------------------------

print("----------------------------------------")
print("FRAME EXTRACTION COMPLETE")
print("----------------------------------------")
print(f"Frames extracted : {frame_number}")
print(f"Frames folder    : {FRAMES_DIR}")

if frame_number == total_frames:
    print("SUCCESS: All video frames extracted.")
else:
    print(
        f"WARNING: Expected {total_frames}, "
        f"but extracted {frame_number}."
    )

print("----------------------------------------")