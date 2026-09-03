import os
import csv


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

IMU_INPUT = os.path.join(
    PROJECT_ROOT,
    "data",
    "imu.csv"
)


def load_imu_data():
    """
    Load IMU data from data/imu.csv.

    Expected columns:
        frame,ax,ay,az,gx,gy,gz
    """

    if not os.path.exists(IMU_INPUT):
        print("IMU file not found.")
        print(f"Expected: {IMU_INPUT}")
        print("IMU fusion will be skipped until sensor data is available.")
        return []

    imu_data = []

    with open(
        IMU_INPUT,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:
                imu_data.append({
                    "frame": int(row["frame"]),
                    "ax": float(row["ax"]),
                    "ay": float(row["ay"]),
                    "az": float(row["az"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"])
                })

            except (ValueError, KeyError):
                continue

    print(f"IMU records loaded: {len(imu_data)}")

    return imu_data


if __name__ == "__main__":

    print("========================================")
    print("IMU MODULE")
    print("========================================")

    data = load_imu_data()

    if data:
        print(f"First IMU record: {data[0]}")
        print(f"Last IMU record : {data[-1]}")
    else:
        print("No IMU data available yet.")

    print("========================================")
    