import pandas as pd

fn = "fingerprint_pass_matrix_cleaned.csv"

# Read without trusting header so we can repair whatever is there
tmp = pd.read_csv(fn, header=None, dtype=str)

# first row contains the header as read by pandas
hdr = tmp.iloc[0].tolist()

# Detect patterns like: ['', 'location', 'MAC1', 'MAC2', ...]
# Or ['','Location', ...] with varying case/whitespace.
hdr0 = hdr[0].strip() if isinstance(hdr[0], str) else hdr[0]
hdr1 = hdr[1].strip().lower() if isinstance(hdr[1], str) else ""

if (hdr0 == "" or str(hdr0).lower().startswith("unnamed")) and hdr1 == "location":
    # Build corrected header: first col = 'location', rest = hdr[2:]
    new_cols = ["location"] + [c.strip() for c in hdr[2:]]
    # Data starts from next row
    data = tmp.iloc[1:].reset_index(drop=True)
    data.columns = new_cols
    df = data.copy()
    # convert RSSI columns to numeric where possible (location stays string)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
else:
    # fallback: try to detect if header already correct (first column named 'location')
    # or if pandas already parsed it fine
    df = pd.read_csv(fn)
    # ensure the first column is 'location'
    if df.columns[0] != "location":
        df = df.rename(columns={df.columns[0]: "location"})

# Strip off everything after the first underscore
df['location'] = df['location'].apply(lambda x: x.split('_')[0])


df.to_csv("fingerprint_pass_matrix_fixed.csv", index=False)

# quick sanity checks
print("Columns (first 10):", df.columns.tolist()[:10])
print("First rows:")
print(df.head())

# Now df has first column named 'location' and MAC columns after that.
# You can proceed with your preprocessing and training on 'df'
