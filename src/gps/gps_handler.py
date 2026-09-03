import os
import csv


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

GPS_INPUT = os.path.join(
    PROJECT_ROOT,
    "data",
    "gps.csv"
)


def load_gps_data():
    """
    Load GPS data from data/gps.csv.

    Expected columns:
        frame,latitude,longitude,altitude
    """

    if not os.path.exists(GPS_INPUT):
        print("GPS file not found.")
        print(f"Expected: {GPS_INPUT}")
        print("GPS fusion will use visual trajectory until GPS data is available.")
        return []

    gps_data = []

    with open(
        GPS_INPUT,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            try:
                gps_data.append({
                    "frame": int(row["frame"]),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "altitude": float(row["altitude"])
                })

            except (ValueError, KeyError):
                continue

    print(f"GPS records loaded: {len(gps_data)}")

    return gps_data


if __name__ == "__main__":

    print("========================================")
    print("GPS MODULE")
    print("========================================")

    data = load_gps_data()

    if data:
        print(f"First GPS record: {data[0]}")
        print(f"Last GPS record : {data[-1]}")
    else:
        print("No GPS data available yet.")

    print("========================================")
    