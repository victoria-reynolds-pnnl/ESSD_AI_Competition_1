# Author: Amzi Jeffs
# AI help: GPT 5.2, see https://ai-incubator-chat.pnnl.gov/s/ebe83ad1-7b6b-4696-9e7f-996cd897938d

import requests
import pandas as pd

BASE_URL = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/items"

def fetch_monitoring_locations(monitoring_location_ids):
    """
    monitoring_location_ids: list like ["USGS-01491000", "USGS-02238500"]
    Returns: pandas DataFrame of location properties (incl. county_name when available)
    """
    params = {
        "f": "json",
        # Ask for only the fields we care about:
        # https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations/schema 
        "properties": ",".join([
            "id",
            "monitoring_location_name",
            "site_type",
            "state_name",
            "county_name",
            "county_code",
            "hydrologic_unit_code",
            "basin_code",
            "altitude",
            "altitude_accuracy",
            "vertical_datum",
            "vertical_datum_name",
            "drainage_area",
            "contributing_drainage_area",
            "aquifer_code",
            "national_aquifer_code",
            "aquifer_type_code",
        ]),
        # The API is paged; raise this if you have many sites.
        "limit": 5000,
    }

    # CQL2 JSON filter: id IN [ ... ]
    cql = {
        "op": "in",
        "args": [
            {"property": "id"},
            monitoring_location_ids
        ]
    }

    r = requests.post(BASE_URL, params=params, json=cql, timeout=60)
    r.raise_for_status()
    data = r.json()

    features = data.get("features", [])
    rows = [f.get("properties", {}) for f in features]
    return pd.DataFrame(rows)

county_signifances = {
    'core': [
        'Asotin County',
        'Chelan County',
        'Columbia County',
        'Garfield County',
        'Kittitas County',
        'Klickitat County', 
        'Okanogan County',
        'Yakima County',
        ],
    'significant': [
        'Benton County',
        'Spokane County',
        'Walla Walla County',
        'Whitman County',
        ],
    'partial': [
        'Adams County',
        'Douglas County',
        'Franklin County',
        'Grant County',
        'Lincoln County',
        ],
}
def get_county_significance(county_name):
    for significance, names in county_signifances.items():
        if county_name in names:
            return significance
    return None

if __name__ == "__main__":
    with open('WA_monitoring_locations.txt', 'r') as f:
        sites = f.read().split(',')

    df = fetch_monitoring_locations(sites)
    df = df.sort_values("id")

    df['county_significance'] = df['county_name'].apply(get_county_significance)

    df = df[df['county_significance'].notna()]

    # Optional: save results
    df.to_csv("../crbg_monitoring_location_info.csv", index=False)
