import pandas as pd

INPUT_CSV = "merged_raw.csv"
OUTPUT_CSV = "merged_clean.csv"

# Number of initial noisy samples to drop
CUT_FRONT = 2000

df = pd.read_csv(INPUT_CSV)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

clean_blocks = []

# Process each label separately
for label, group in df.groupby("label"):
    print(f"Processing label: {label}  |  total samples = {len(group)}")

    # Sort by timestamp (important when timestamps jump)
    group = group.sort_values("timestamp_ms").reset_index(drop=True)

    # Remove the first 2000 noisy samples
    if len(group) > CUT_FRONT:
        group = group.iloc[CUT_FRONT:].reset_index(drop=True)
    else:
        print(f" Warning: label '{label}' has fewer than {CUT_FRONT} samples, skipping trim.")

    # Reset timestamp to make it continuous (0, 20, 40, ...)
    # Approx sampling freq = 50 Hz → 20ms per sample
    group["timestamp_ms"] = group.index * 20

    clean_blocks.append(group)

# Reassemble full cleaned dataset
clean_df = pd.concat(clean_blocks, ignore_index=True)

# Save result
clean_df.to_csv(OUTPUT_CSV, index=False)

print("\nCleaning done!")
print(f"Saved cleaned CSV as {OUTPUT_CSV}")
print("Final dataset size:", len(clean_df))
