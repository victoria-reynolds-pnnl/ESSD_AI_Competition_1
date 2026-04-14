# Author: R. Amzi Jeffs, with assistance from Claude Haiku.
# See chat history here: https://ai-incubator-chat.pnnl.gov/s/a311ff39-37b5-4874-8dac-85150d00a087
#!/usr/bin/env python3
import os
import sys
import json
import time
import csv
import datetime as dt
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
import pandas as pd

BASE_ITEMS_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"

# Load monitoring locations
monitoring_location_info = pd.read_csv('../Data/crbg_monitoring_location_info.csv')
MONITORING_LOCATION_IDS = [str(ID) for ID in monitoring_location_info['id']]

PARAMETER_CODE = "00060"   # Discharge
STATISTIC_ID = "00003"     # Daily mean

START_DATE = "1950-01-01"
END_DATE = dt.date.today().isoformat()
TIME_CHUNK_YEARS = 50      # chunk by 50 years per request

LIMIT = 10000              # API max per page
SLEEP_BETWEEN_REQUESTS = 0.2

OUT_DIR = "../Data/"

# ---------------------------------------------------

API_KEY = os.getenv("API_USGS_PAT")
HEADERS = {"X-Api-Key": API_KEY} if API_KEY else {}

def iso_date(s):
    return dt.date.fromisoformat(s)

def daterange_chunks(start, end, years=1):
    """Yield (chunk_start, chunk_end) inclusive, as ISO strings."""
    start_d = iso_date(start)
    end_d = iso_date(end)
    cur = start_d
    while cur <= end_d:
        try:
            nxt = cur.replace(year=cur.year + years)
        except ValueError:
            nxt = cur + (dt.date(cur.year + years, 3, 1) - dt.date(cur.year, 3, 1))
        chunk_end = min(end_d, nxt - dt.timedelta(days=1))
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + dt.timedelta(days=1)

def add_api_key_to_next_link(next_href: str) -> str:
    """The 'next' link may not include your API key. Add it if we have one."""
    if not API_KEY:
        return next_href
    u = urlparse(next_href)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    new_q = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))

def fetch_all_pages(params, session: requests.Session):
    """Generator yielding GeoJSON Features across all pages for one request."""
    url = BASE_ITEMS_URL
    page_num = 0
    while True:
        page_num += 1
        try:
            r = session.get(url, params=params if url == BASE_ITEMS_URL else None,
                            headers=HEADERS, timeout=120)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  ERROR on page {page_num}: {e}", file=sys.stderr)
            break
        payload = r.json()
        for feat in payload.get("features", []):
            yield feat
        next_url = None
        for link in payload.get("links", []):
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break
        if not next_url:
            break
        url = add_api_key_to_next_link(next_url)
        params = None
        time.sleep(SLEEP_BETWEEN_REQUESTS)

def write_geojsonl(features, path):
    """Write features to newline-delimited GeoJSON."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(json.dumps(feat) + "\n")
            n += 1
    return n

def append_csv_from_geojsonl(geojsonl_path, csv_path, write_header_if_new=True):
    """Flatten GeoJSON Features -> CSV with columns from properties + lon/lat."""
    fieldnames = None
    rows_written = 0

    csv_exists = os.path.exists(csv_path)
    mode = "a" if csv_exists else "w"

    with open(geojsonl_path, "r", encoding="utf-8") as fin, \
         open(csv_path, mode, newline="", encoding="utf-8") as fout:
        writer = None
        for line in fin:
            feat = json.loads(line)
            props = feat.get("properties", {}) or {}

            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") if geom.get("type") == "Point" else None
            lon, lat = (coords[0], coords[1]) if coords and len(coords) >= 2 else (None, None)

            row = dict(props)
            row["latitude"] = lat
            row["longitude"] = lon

            if fieldnames is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
                if (not csv_exists) and write_header_if_new:
                    writer.writeheader()

            writer.writerow(row)
            rows_written += 1

    return rows_written

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    total_locations_processed = 0
    total_features_saved = 0

    # Loop over each monitoring location
    for site_num, monitoring_location_id in enumerate(MONITORING_LOCATION_IDS, 1):
        print(f"\n=== Location {site_num}/{len(MONITORING_LOCATION_IDS)}: {monitoring_location_id} ===", file=sys.stderr)

        # Output file for this location
        location_csv = os.path.join(OUT_DIR, f"{monitoring_location_id}_data.csv")

        location_total = 0

        with requests.Session() as session:
            # Chunk time ranges
            for (cstart, cend) in daterange_chunks(START_DATE, END_DATE, years=TIME_CHUNK_YEARS):
                params = {
                    "f": "json",
                    "monitoring_location_id": monitoring_location_id,
                    "parameter_code": PARAMETER_CODE,
                    "statistic_id": STATISTIC_ID,
                    "time": f"{cstart}/{cend}",
                    "limit": LIMIT,
                }

                # Temporary geojsonl file for this time chunk
                temp_geojsonl = os.path.join(
                    OUT_DIR,
                    f"_{monitoring_location_id}_temp_{cstart}_to_{cend}.geojsonl"
                )

                print(f"  Downloading {cstart} to {cend} ...", file=sys.stderr)
                feats = fetch_all_pages(params, session)
                n = write_geojsonl(feats, temp_geojsonl)
                location_total += n
                total_features_saved += n
                print(f"    saved {n} features", file=sys.stderr)

                # Append into location CSV
                if n > 0:
                    appended = append_csv_from_geojsonl(
                        temp_geojsonl,
                        location_csv,
                        write_header_if_new=(location_total == n)
                    )
                    print(f"    appended {appended} rows -> {location_csv}", file=sys.stderr)

                # Clean up temp file
                if os.path.exists(temp_geojsonl):
                    os.remove(temp_geojsonl)

        print(f"  Location total: {location_total} features", file=sys.stderr)
        total_locations_processed += 1

    print(f"\n✓ Done. Processed {total_locations_processed} locations with {total_features_saved} total features.", file=sys.stderr)