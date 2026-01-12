# summarize_rssi.py
import pandas as pd

# --- INPUT / OUTPUT PATHS ---
INPUT_FILE  = "merged_raw.csv"          # file from Step 1
OUTPUT_FILE = "fingerprint_summary.csv"

# --- LOAD RAW DATA ---
df = pd.read_csv(INPUT_FILE)

# --- BASIC CLEANUP (OPTIONAL) ---
# Drop empty RSSI rows, ensure numeric type
df = df.dropna(subset=["rssi"])
df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")

# --- GROUP & SUMMARIZE ---
summary = (
    df.groupby(["location", "mac"])
      .agg(mean_rssi   = ("rssi", "mean"),
           median_rssi = ("rssi", "median"),
           std_rssi    = ("rssi", "std"),
           samples     = ("rssi", "count"))
      .reset_index()
)

# --- SAVE ---
summary.to_csv(OUTPUT_FILE, index=False)
print(f"Saved summarized fingerprint data → {OUTPUT_FILE}")
