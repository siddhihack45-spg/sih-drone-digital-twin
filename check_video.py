import cv2
from pathlib import Path

VIDEO = "data/video/drone.mp4"
OUTPUT = Path("output/checks")

OUTPUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)

frame_ids = [0, 100, 200, 300, 400, 500, 600]

for frame_id in frame_ids:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    ok, frame = cap.read()

    if ok:
        path = OUTPUT / f"frame_{frame_id}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"Saved: {path}")

cap.release()

print("Done.")
