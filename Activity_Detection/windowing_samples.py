import pandas as pd
import numpy as np

WINDOW = 50
STRIDE = 25

df = pd.read_csv("merged_no_gyro.csv")

# Ensure ordering by timestamp inside each label class
df = df.sort_values(["label", "timestamp_ms"]).reset_index(drop=True)

X_windows = []
y_windows = []

for label, group in df.groupby("label"):
    group = group.reset_index(drop=True)
        # shape = (50, 6) = ax ay az gx gy gz
    data = group[["az"]].values.astype(np.float32)

    for start in range(0, len(group) - WINDOW + 1, STRIDE):
        window = data[start:start+WINDOW]
        X_windows.append(window)
        y_windows.append(label)

X_windows = np.array(X_windows)         # (N,50,6)
y_windows = np.array(y_windows)         # (N,)

np.save("X_windows_no_gyro.npy", X_windows)
np.save("y_windows_no_gyro.npy", y_windows)

print("Saved X shape", X_windows.shape)
print("Saved y shape", y_windows.shape)
