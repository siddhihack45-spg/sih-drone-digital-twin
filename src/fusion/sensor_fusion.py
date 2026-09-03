import os
import csv


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VISUAL_INPUT = os.path.join(
    PROJECT_ROOT,
    "output",
    "visual_trajectory.csv"
)

FUSED_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "output",
    "fused_trajectory.csv"
)


def load_visual_trajectory():

    if not os.path.exists(VISUAL_INPUT):
        print("ERROR: Visual trajectory not found.")
        print(VISUAL_INPUT)
        return []

    trajectory = []

    with open(
        VISUAL_INPUT,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:
                trajectory.append({
                    "frame": int(row["frame"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"])
                })

            except (ValueError, KeyError):
                continue

    return trajectory


def run_fusion():

    visual = load_visual_trajectory()

    if not visual:
        return

    print("========================================")
    print("SENSOR FUSION")
    print("========================================")
    print(f"Visual trajectory points: {len(visual)}")

    # --------------------------------------------------------
    # CURRENT STAGE
    # --------------------------------------------------------
    # GPS and IMU will be incorporated when actual sensor
    # recordings are available.
    #
    # For now, preserve the visual trajectory so the complete
    # 868-frame pipeline can continue working.
    # --------------------------------------------------------

    fused = visual

    with open(
        FUSED_OUTPUT,
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

        for point in fused:

            writer.writerow([
                point["frame"],
                point["x"],
                point["y"],
                point["z"]
            ])

    print(f"Fused trajectory points: {len(fused)}")
    print(f"Output: {FUSED_OUTPUT}")
    print("========================================")


if __name__ == "__main__":
    run_fusion()