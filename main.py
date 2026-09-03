import cv2
from pathlib import Path


VIDEO_PATH = Path("data/video/drone.mp4")
FRAME_DIR = Path("data/frames")

FRAME_INTERVAL = 2


def extract_frames():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_INTERVAL == 0:
            timestamp = frame_id / fps

            filename = FRAME_DIR / f"frame_{frame_id:06d}.jpg"

            cv2.imwrite(str(filename), frame)

            print(
                f"Saved: {filename.name} "
                f"| timestamp: {timestamp:.3f}s"
            )

            saved_count += 1

        frame_id += 1

    cap.release()

    print("\n========== EXTRACTION COMPLETE ==========")
    print(f"Original frames : {frame_id}")
    print(f"Saved frames    : {saved_count}")
    print(f"Output folder   : {FRAME_DIR}")
    print("==========================================")


if __name__ == "__main__":
    extract_frames()

