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
end = pd.Timestamp.now(tz="UTC").normalize()

wd = wd[wd["time"] >= start].copy()

# --- require >= 80% daily coverage since 2000 (at least one entry per day counts as "present") ---
total_days = (end - start).days + 1

# count unique days with any measurement, per location
days_present = (
    wd.assign(day=wd["time"].dt.floor("D"))
      .groupby("monitoring_location_id")["day"]
      .nunique()
)

keep_locs = days_present[days_present >= 0.80 * total_days].index

wd_filtered = wd[wd["monitoring_location_id"].isin(keep_locs)].copy()
wd_filtered = wd_filtered.drop(columns=['parameter_code', 
                                        'time_series_id', 
                                        'statistic_id', 
                                        'unit_of_measure', 
                                        'approval_status', 
                                        'last_modified'])

wd_filtered.to_csv('../Data/data_cleaned.csv')
