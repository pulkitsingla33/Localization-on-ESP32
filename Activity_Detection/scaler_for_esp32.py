import numpy as np

mean = np.load("scaler_mean_no_gyro.npy")
std  = np.load("scaler_std_no_gyro.npy")

with open("activity_scaler.h", "w") as f:
    f.write("#ifndef ACTIVITY_SCALER_H\n#define ACTIVITY_SCALER_H\n\n")
    
    f.write(f"static const int FEATURES = {len(mean)};\n\n")

    f.write("static const float scaler_mean[] = {")
    f.write(",".join([f"{x:.8f}" for x in mean]))
    f.write("};\n\n")

    f.write("static const float scaler_std[] = {")
    f.write(",".join([f"{x:.8f}" for x in std]))
    f.write("};\n\n")

    f.write("#endif\n")
print("Generated activity_scaler.h successfully.")

