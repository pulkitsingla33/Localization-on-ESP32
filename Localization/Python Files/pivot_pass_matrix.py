# make_pass_samples.py
import pandas as pd
import os
from datetime import timedelta

df = pd.read_csv("merged_raw.csv")  # columns: timestamp_ms, mac, rssi, channel, location

df = df[pd.to_numeric(df["rssi"], errors="coerce").notna()]
df["rssi"] = df["rssi"].astype(float)

# convert timestamp if needed
if "timestamp_ms" in df.columns:
    df['ts'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
elif "timestamp" in df.columns:
    df['ts'] = pd.to_datetime(df['timestamp'])
else:
    df['ts'] = pd.Timestamp.now()

# choose window seconds
window_s = 5

samples = []
for loc, grp in df.groupby("location"):
    grp = grp.sort_values("ts")
    # create windows based on a rolling time or fixed bins
    start = grp['ts'].min()
    end = grp['ts'].max()
    cur = start
    while cur <= end:
        nxt = cur + pd.Timedelta(seconds=window_s)
        win = grp[(grp['ts'] >= cur) & (grp['ts'] < nxt)]
        if not win.empty:
            # pivot median
            med = win.groupby("mac")['rssi'].median().to_dict()
            samples.append((f"{loc}_{cur.strftime('%H%M%S')}", loc, med))
        cur = nxt
# Save per-pass pivot file for knn_overlap loader
# Build dataframe
all_macs = sorted({m for _,_,d in samples for m in d.keys()})
rows = []
index=[]
for passid, loc, d in samples:
    row = {mac: d.get(mac, -105) for mac in all_macs}
    rows.append(row)
    index.append(passid)  # you might want to include loc in passid or a separate col

import pandas as pd
df_pass = pd.DataFrame(rows, index=index, columns=all_macs)
df_pass.to_csv("fingerprint_pass_matrix.csv")
print("Saved fingerprint_pass_matrix.csv")
