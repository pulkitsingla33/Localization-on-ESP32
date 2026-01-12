import pandas as pd

INPUT_FILE = "fingerprint_pass_matrix.csv"
OUTPUT_FILE = "fingerprint_pass_matrix_cleaned.csv"

def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return df

def compute_ap_presence(df, threshold=-90):
    """
    AP is counted as present if RSSI > threshold.
    """
    presence_mask = df > threshold
    presence_counts = presence_mask.sum(axis=0)
    return presence_counts

def remove_rare_aps(df, min_fraction=0.02, threshold=-90):
    n_rows = len(df)

    # Count how many times each AP is seen above threshold
    presence_counts = compute_ap_presence(df, threshold)

    # Keep APs seen in at least min_fraction of samples
    min_required = int(n_rows * min_fraction)
    good_aps = presence_counts[presence_counts >= min_required].index

    print(f"Total APs: {len(df.columns)}")
    print(f"APs kept (≥ {min_required} occurrences): {len(good_aps)}")
    print(f"APs removed: {len(df.columns) - len(good_aps)}")

    return df[good_aps], good_aps

def main():
    df = load_matrix(INPUT_FILE)
    cleaned_df, good_aps = remove_rare_aps(df, min_fraction=0.03, threshold=-90)

    
    cleaned_df.to_csv(OUTPUT_FILE)
    print(f"\nCleaned matrix saved to: {OUTPUT_FILE}")

    # Optional: save list of kept AP MACs
    with open("kept_aps.txt", "w") as f:
        for mac in good_aps:
            f.write(mac + "\n")
    print("Kept AP list saved to kept_aps.txt")

if __name__ == "__main__":
    main()
