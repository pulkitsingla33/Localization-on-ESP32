import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

CSV_FILE = "merged_clean.csv"     # change if needed

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------

df = pd.read_csv(CSV_FILE)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

required = ["ax", "ay", "az", "label"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' missing from CSV!")

# Compute acceleration magnitude
df["a_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

# ---------------------------------------------------------
# Plot magnitude timeseries for each label
# ---------------------------------------------------------

def plot_magnitude_timeseries(df, label):
    subset = df[df["label"] == label]

    if len(subset) == 0:
        print(f"No samples for label '{label}'")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(subset["a_mag"].values, linewidth=1.2)
    plt.title(f"Acceleration Magnitude – {label}")
    plt.ylabel("Acceleration (m/s²)")
    plt.xlabel("Sample Index")
    plt.grid()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# KDE distribution of magnitude
# ---------------------------------------------------------

def plot_magnitude_distribution(df, label):
    subset = df[df["label"] == label]

    plt.figure(figsize=(10, 5))
    sns.kdeplot(subset["a_mag"], fill=True, linewidth=1.5)
    plt.title(f"Acceleration Magnitude Distribution – {label}")
    plt.xlabel("Acceleration Magnitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------

labels = df["label"].unique()
print("Found labels:", labels)

for label in labels:
    if(label == 'stairs' or label == 'stationary'):
        continue
    print(f"\n--- Visualizing: {label} ---")
    plot_magnitude_timeseries(df, label)
    plot_magnitude_distribution(df, label)
