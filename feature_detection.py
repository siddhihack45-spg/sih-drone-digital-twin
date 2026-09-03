import cv2
from pathlib import Path


FRAME_DIR = Path("data/frames")


def main():
    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))

    if not frame_paths:
        print("No frames found.")
        return

    # ORB feature detector
    orb = cv2.ORB_create(nfeatures=1000)

    total_features = 0

    for frame_path in frame_paths:

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read: {frame_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = orb.detectAndCompute(
            gray,
            None
        )

        feature_count = len(keypoints)
        total_features += feature_count

        print(
            f"{frame_path.name} → "
            f"{feature_count} features"
        )

    average = total_features / len(frame_paths)

    print("\n========== FEATURE DETECTION ==========")
    print(f"Frames processed : {len(frame_paths)}")
    print(f"Average features : {average:.2f}")
    print("=======================================")


if __name__ == "__main__":
    main()

    