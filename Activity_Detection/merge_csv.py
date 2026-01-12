import pandas as pd
import glob

files = glob.glob("IMU_data2/*.csv")
dfs = []

for f in files:
    df = pd.read_csv(f)
    # df["source_file"] = f     # optional tracking
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv("merged_raw.csv", index=False)

print("Merged", len(files), "files → merged_raw.csv")
