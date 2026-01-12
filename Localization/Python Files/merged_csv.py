import pandas as pd
import glob
import os

# path where all your P1.csv, P2.csv, ... files are stored
path = "./fingerprinting_data_2/"
files = glob.glob(os.path.join(path, "*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    # if the location name isn’t in a column, infer it from filename:
    if "location" not in df.columns:
        loc = os.path.splitext(os.path.basename(f))[0]
        df["location"] = loc
    
    df = df[["timestamp_ms", "mac", "rssi", "location"]]
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)
all_data.to_csv("merged_raw.csv", index=False)


