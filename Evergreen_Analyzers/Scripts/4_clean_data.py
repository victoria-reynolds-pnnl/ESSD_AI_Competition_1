# Author: R. Amzi Jeffs, with assistance from Claude Haiku.

#!/usr/bin/env python3
import os
import sys
import json
import time
import csv
import datetime as dt
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from itertools import islice
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

wd = water_data.copy()
wd["time"] = pd.to_datetime(wd["time"], errors="coerce", utc=True)
wd = wd.dropna(subset=["monitoring_location_id", "time"])

start = pd.Timestamp("2000-01-01", tz="UTC")
end = pd.Timestamp("2026-03-30", tz="UTC")
all_days = pd.date_range(start, end, freq="D", tz="UTC")

wd = wd[wd["time"] >= start].copy()

# require >= 80% daily coverage since 2000 (at least one entry per day counts as "present") ---
total_days = (end - start).days + 1

# Remove rows where qualifier isn't just revised and/or estimated

def qualifiers_acceptable(Q):
    """ Check if list of qualifiers is acceptable to retain in the dataset """
    if pd.isna(Q):
        return True
    
    Q = eval(Q)
    acceptable_qualifiers = [
        'ESTIMATED',
        'REVISED',
        'ZEROFLOW',
        'DISCONTINUED',
        ]
    Q = [qualifier for qualifier in Q if qualifier not in acceptable_qualifiers]

    return len(Q) == 0

acceptable_qualifiers = wd['qualifier'].apply(qualifiers_acceptable)
wd = wd[acceptable_qualifiers]

# count unique days with any measurement, per location
days_present = (
    wd.assign(day=wd["time"].dt.floor("D"))
      .groupby("monitoring_location_id")["day"]
      .nunique()
)

keep_locs = days_present[days_present >= 0.80 * total_days].index

# Get rid of locations that don't have enough data
wd_filtered = wd[wd["monitoring_location_id"].isin(keep_locs)].copy()

# Remove extra columns, and rename the flow rate column for clarity
wd_filtered = wd_filtered.drop(columns=[
    'id',
    'parameter_code', 
    'time_series_id', 
    'statistic_id', 
    'unit_of_measure', 
    'approval_status', 
    'last_modified'])
wd_filtered = wd_filtered.rename(columns={'value':'flow_rate'})

# ============================================================
# 1) Make panel complete: every location x every day
#    Fill missing rows with NA except keep (location, time, lat, lon)
# ============================================================
# Get one lat/lon per location (first non-null)
loc_latlon = (wd_filtered.sort_values("time")
                .groupby("monitoring_location_id")[["latitude", "longitude"]]
                .first())

# Index on (monitoring_location_id, time) and reindex to full grid
wd_filtered = wd_filtered.set_index(["monitoring_location_id", "time"]).sort_index()

full_index = pd.MultiIndex.from_product(
    [loc_latlon.index, all_days],
    names=["monitoring_location_id", "time"]
)

wd_filtered = wd_filtered.reindex(full_index)

# Re-fill lat/lon for created rows (and any missing)
wd_filtered = wd_filtered.join(loc_latlon, on="monitoring_location_id", rsuffix="_loc")
wd_filtered["latitude"]  = wd_filtered["latitude"].fillna(wd_filtered["latitude_loc"])
wd_filtered["longitude"] = wd_filtered["longitude"].fillna(wd_filtered["longitude_loc"])
wd_filtered = wd_filtered.drop(columns=["latitude_loc", "longitude_loc"])

# Bring columns back (optional)
wd_filtered = wd_filtered.reset_index()


# ============================================================
# 2) Seasonal sin/cos embedding (day-of-year on a yearly cycle)
# ============================================================
t = wd_filtered["time"]
doy = t.dt.dayofyear.astype("float32")
year_len = t.dt.is_leap_year.astype("int16").map({0: 365, 1: 366}).astype("float32")

phase = 2.0 * np.pi * (doy - 1.0) / year_len
wd_filtered["seasonal_sin"] = np.sin(phase).astype("float32")
wd_filtered["seasonal_cos"] = np.cos(phase).astype("float32")

# ============================================================
# 3) Trend time since START (in years)
# ============================================================
wd_filtered["trend"] = ((wd_filtered["time"] - start) / pd.Timedelta(days=365.25)).astype("float32")


# ============================================================
# 4) Rolling averages of previous values only (no leakage), per location
#    Handles NaNs: mean ignores NaNs; if all-NaN in window -> result NaN
# ============================================================
wd_filtered = wd_filtered.sort_values(["monitoring_location_id", "time"])

g = wd_filtered.groupby("monitoring_location_id", sort=False)["flow_rate"]

wd_filtered["rolling_7"]  = g.transform(lambda s: s.shift(1).rolling(7,  min_periods=1).mean())
wd_filtered["rolling_14"] = g.transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
wd_filtered["rolling_21"] = g.transform(lambda s: s.shift(1).rolling(21, min_periods=1).mean())


# Write to CSV
wd_filtered.to_csv('../Data/data_cleaned.csv')
