import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Load CSV
# -------------------------------------------------------------------

CSV_FILE = "merged_clean.csv"
df = pd.read_csv(CSV_FILE)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

required = ["timestamp_ms", "ax", "ay", "az", "label"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Column {col} missing!")

# Sort by time (important)
df = df.sort_values("timestamp_ms").reset_index(drop=True)

# Downsample if file is huge (for cleaner plotting)
MAX_POINTS = 5000
if len(df) > MAX_POINTS:
    df_plot = df.iloc[::len(df)//MAX_POINTS]
else:
    df_plot = df

# -------------------------------------------------------------------
# Utility – Plot Large Time Series for a Given Activity Label
# -------------------------------------------------------------------

def plot_accel_label(label, seconds=10):
    """
    Plots a clean 10-second segment of ax/ay/az for the given label.
    """
    subset = df[df["label"] == label]
    if len(subset) == 0:
        print(f"No samples for label '{label}'")
        return
    
    # Pick the first N seconds of data
    t0 = subset["timestamp_ms"].iloc[0]
    t_end = t0 + seconds * 1000
    segment = subset[(subset["timestamp_ms"] >= t0) & 
                     (subset["timestamp_ms"] <= t_end)]
    
    if len(segment) < 20:
        print(f"Not enough data for {label}")
        return
    
    plt.figure(figsize=(16, 6))
    plt.plot(segment["timestamp_ms"], segment["ax"], label="ax")
    plt.plot(segment["timestamp_ms"], segment["ay"], label="ay")
    plt.plot(segment["timestamp_ms"], segment["az"], label="az")

    plt.title(f"Accelerometer (First {seconds} seconds) – {label}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# KDE Distribution Plots
# -------------------------------------------------------------------

def plot_distributions(label):
    subset = df[df["label"] == label]
    
    plt.figure(figsize=(14, 5))
    for col in ["ax", "ay", "az"]:
        sns.kdeplot(subset[col], label=col, fill=True)
    plt.title(f"Accelerometer Distribution – {label}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Scatter Plots (Optional for Separability)
# -------------------------------------------------------------------

def plot_scatter(label):
    subset = df[df["label"] == label]

    plt.figure(figsize=(7, 6))
    plt.scatter(subset["ax"], subset["ay"], s=4, alpha=0.3)
    plt.title(f"ax vs ay – {label}")
    plt.xlabel("ax")
    plt.ylabel("ay")
    plt.grid()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Execute for each label
# -------------------------------------------------------------------

labels = df["label"].unique()
print("Found labels:", labels)

for label in labels:
    if(label == 'stationary' or label == 'stairs'):
        continue
    print(f"\n==== {label.upper()} ====")  
    plot_accel_label(label, seconds=30)
    plot_distributions(label)
    # plot_scatter(label)   # optional
