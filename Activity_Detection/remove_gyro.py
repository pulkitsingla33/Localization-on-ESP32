import pandas as pd

INPUT_FILE  = "merged_no_gyro.csv"
OUTPUT_FILE = "merged_no_gyro.csv"

def remove_gyro():
    df = pd.read_csv(INPUT_FILE)

    # Normalize column names (in case they have spaces/caps)
    df.columns = [c.strip().lower() for c in df.columns]

    gyro_cols = ["ax", "ay"]

    # Check and drop only existing columns
    to_drop = [col for col in gyro_cols if col in df.columns]

    print("Dropping columns:", to_drop)

    df = df.drop(columns=to_drop)

    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    remove_gyro()
