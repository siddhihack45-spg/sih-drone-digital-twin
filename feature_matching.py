import cv2
from pathlib import Path


FRAME_DIR = Path("data/frames")


def main():
    frame_paths = sorted(FRAME_DIR.glob("*.jpg"))

    if len(frame_paths) < 2:
        print("Need at least 2 frames.")
        return

    orb = cv2.ORB_create(nfeatures=1000)

    # Hamming distance is appropriate for ORB descriptors
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    total_matches = 0
    pair_count = 0

    for i in range(len(frame_paths) - 1):

        img1 = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame_paths[i + 1]), cv2.IMREAD_GRAYSCALE)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            print(f"{frame_paths[i].name} -> {frame_paths[i + 1].name}: no descriptors")
            continue

        matches = matcher.match(des1, des2)

        # Sort from best match to worst match
        matches = sorted(matches, key=lambda m: m.distance)

        # Keep the strongest matches
        good_matches = matches[:100]

        print(
            f"{frame_paths[i].name} -> "
            f"{frame_paths[i + 1].name}: "
            f"{len(good_matches)} good matches"
        )

        total_matches += len(good_matches)
        pair_count += 1

    average = total_matches / pair_count if pair_count else 0

    print("\n========== FEATURE MATCHING ==========")
    print(f"Frame pairs processed : {pair_count}")
    print(f"Average good matches  : {average:.2f}")
    print("=======================================")


if __name__ == "__main__":
    main()

    