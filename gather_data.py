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

import requests

BASE_ITEMS_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"

# Load monitoring locations
with open('WA_monitoring_locations.txt', 'r') as f:
    MONITORING_LOCATION_IDS = f.read().split(',')

PARAMETER_CODE = "00060"   # Discharge
STATISTIC_ID = "00003"     # Daily mean

START_DATE = "1950-01-01"
END_DATE = dt.date.today().isoformat()
TIME_CHUNK_YEARS = 4       # chunk by year
SITE_CHUNK_SIZE = 25       # query ~25 sites at a time (API handles ~50, but be conservative)

LIMIT = 10000              # API max per page
SLEEP_BETWEEN_REQUESTS = 0.2

OUT_DIR = "usgs_daily_download"
OUT_PREFIX = f"daily_{PARAMETER_CODE}_{STATISTIC_ID}"
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

def batched(iterable, n):
    """Batch iterable into chunks of size n."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            break
        yield batch

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
            row["longitude"] = lon
            row["latitude"] = lat

            if fieldnames is None:
                fieldnames = list(row.keys())
                writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
                if (not csv_exists) and write_header_if_new:
                    writer.writeheader()

            writer.writerow(row)
            rows_written += 1

    return rows_written

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    combined_csv = os.path.join(OUT_DIR, f"{OUT_PREFIX}_ALL.csv")

    total_features = 0
    site_batch_num = 0

    # Chunk monitoring location IDs
    for site_batch in batched(MONITORING_LOCATION_IDS, SITE_CHUNK_SIZE):
        site_batch_num += 1
        site_ids_str = ",".join(site_batch)
        print(f"\n=== Site Batch {site_batch_num} ({len(site_batch)} sites) ===", file=sys.stderr)
        print(f"Sites: {site_ids_str}", file=sys.stderr)

        with requests.Session() as session:
            # Chunk time ranges
            for (cstart, cend) in daterange_chunks(START_DATE, END_DATE, years=TIME_CHUNK_YEARS):
                params = {
                    "f": "json",
                    "monitoring_location_id": site_ids_str,
                    "parameter_code": PARAMETER_CODE,
                    "statistic_id": STATISTIC_ID,
                    "time": f"{cstart}/{cend}",
                    "limit": LIMIT,
                }

                # Filename includes both batch and time chunk
                out_geojsonl = os.path.join(
                    OUT_DIR,
                    f"{OUT_PREFIX}_batch{site_batch_num:03d}_{cstart}_to_{cend}.geojsonl"
                )

                print(f"  Downloading {cstart} to {cend} ...", file=sys.stderr)
                feats = fetch_all_pages(params, session)
                n = write_geojsonl(feats, out_geojsonl)
                total_features += n
                print(f"    saved {n} features -> {out_geojsonl}", file=sys.stderr)

                # Append into combined CSV
                if n > 0:
                    appended = append_csv_from_geojsonl(out_geojsonl, combined_csv,
                                                        write_header_if_new=(total_features == n))
                    print(f"    appended {appended} rows -> {combined_csv}", file=sys.stderr)

    print(f"\n✓ Done. Total features saved: {total_features}", file=sys.stderr)

if __name__ == "__main__":
    main()
