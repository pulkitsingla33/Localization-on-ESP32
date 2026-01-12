import serial
import csv
from datetime import datetime
import os

port = "COM5"   # or "/dev/ttyUSB0" on Linux/Mac
baud = 115200

# Get the absolute directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file path in same directory as the script
filename = os.path.join(
    script_dir,
    f"wifi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)

ser = serial.Serial(port, baud)

with open(filename, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_ms", "mac", "rssi", "channel", "location_label"])
    
    print(f"Logging to {filename}...")
    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line and not line.startswith("timestamp"):
                parts = line.split(",")
                if len(parts) == 5:
                    writer.writerow(parts)
                    f.flush()
                    print(line)
    except KeyboardInterrupt:
        print("Logging stopped.")
