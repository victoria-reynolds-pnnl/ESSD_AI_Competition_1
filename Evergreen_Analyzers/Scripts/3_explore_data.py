# Author: R. Amzi Jeffs, with assistance from Claude Haiku.

#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

monitoring_location_info = pd.read_csv('../Data/crbg_monitoring_location_info.csv')
monitoring_location_ids = [str(ID) for ID in monitoring_location_info['id']]

data_by_location = {
    monitoring_location_id : pd.read_csv(f'../Data/{monitoring_location_id}_data.csv')
    for monitoring_location_id in monitoring_location_ids
    if Path(f'../Data/{monitoring_location_id}_data.csv').exists()
    }

water_data = pd.concat(data_by_location.values())

df = water_data.copy()

df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df = df.dropna(subset=["monitoring_location_id", "time"]).sort_values(["monitoring_location_id", "time"])

START = pd.Timestamp("1950-01-01", tz="UTC")
END   = pd.Timestamp.now(tz="UTC").normalize()
all_days = pd.date_range(START, END, freq="D", tz="UTC")
n_days = len(all_days)
day0 = all_days[0]

# Day index since START
df["day_idx"] = ((df["time"].dt.floor("D") - day0) / pd.Timedelta(days=1)).astype("int32")
df = df[(df["day_idx"] >= 0) & (df["day_idx"] < n_days)]

# De-dup: one "present" per location per day
df = df.loc[:, ["monitoring_location_id", "day_idx"]].drop_duplicates()

# Compute "most recent missing day" per location:
# - Find the last day present
# - Then scan forward to find the first missing day after that (or treat "no missing after last present" as END+1)
g = df.groupby("monitoring_location_id", sort=False)["day_idx"]

locs_sorted = list(g.size().sort_values().index)

# Build presence matrix in sorted order
loc_to_row = {loc: i for i, loc in enumerate(locs_sorted)}
present = np.zeros((len(locs_sorted), n_days), dtype=np.uint8)

for loc, idxs in g:
    r = loc_to_row[loc]
    present[r, idxs.to_numpy(dtype=np.int32, copy=False)] = 1

# --- plot ---
fig_h = max(2.0, 0.25 * len(locs_sorted))
fig, ax = plt.subplots(figsize=(15, fig_h))

ax.imshow(present, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)

years = all_days.year
xticks = np.flatnonzero((all_days.month == 1) & (all_days.day == 1) & (years % 5 == 0))
ax.set_xticks(xticks)
ax.set_xticklabels([str(all_days[i].year) for i in xticks])

ax.set_yticks(np.arange(len(locs_sorted)))
ax.set_yticklabels(locs_sorted, fontsize=7)

ax.set_xlabel("Date")
ax.set_ylabel("monitoring_location_id (sorted by most recent missing day)")
ax.set_title("Daily measurement presence (black) and gaps (white); most complete locations at bottom")

plt.tight_layout()
plt.savefig('data_gaps.png')
plt.show()




