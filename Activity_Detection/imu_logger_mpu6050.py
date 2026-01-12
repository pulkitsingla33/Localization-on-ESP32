import serial
import csv
import time
from datetime import datetime

PORT = "COM5"          # change this if needed
BAUD = 115200
MAX_SAMPLES = 3750     # stop after this many IMU readings

def main():
    label = input("Enter activity label: ").strip()
    print(f"Saving samples under label: {label}")

    filename = f"imu_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Writing to {filename}")

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)   # allow ESP32 to reset

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_ms",
            "ax", "ay", "az",
            "gx", "gy", "gz",
            "label"
        ])

        print("\nCollecting... Will stop after 30,000 samples.\n")

        sample_count = 0

        try:
            while sample_count < MAX_SAMPLES:
                line = ser.readline().decode().strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) != 7:
                    print("Malformed:", line)
                    continue

                row = parts + [label]
                writer.writerow(row)
                sample_count += 1

                if sample_count % 100 == 0:
                    print(f"Collected {sample_count}/{MAX_SAMPLES}")

        except KeyboardInterrupt:
            print("\nStopped early by user.")

    ser.close()
    print("\nDone! Saved CSV with", sample_count, "samples.")

if __name__ == "__main__":
    main()
