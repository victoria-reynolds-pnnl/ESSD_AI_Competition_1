# ML model selection

# ML model justification

# Data dictionary table

Cleaned and engineered dataset is in `Data/data_cleaned.csv`. The data covers
50 monitoring locations from January 1, 2000 to March 30, 2026. The columns
in the data are as follows:

| Field Name | Description | Data Type | Example |
|---|---|---|---|
| monitoring_location_id | The USGS id for the monitoring location. | string | USGS-12422500 |
| time | The date that the measurement was taken. | Pandas timestamp | 2000-12-25 00:00:00+00:00 |
| flow_rate | The average flow rate at the location on the given date. | float | 2230.0 |
| qualifier | Qualifiers regarding data quality. See data preparation paragraph for details on qualifiers and which were omitted. | string | \['ESTIMATED', 'REVISED'\] |
| latitude | Latitude of the monitoring location. | float | 47.6593354474072 |
| longitude | Longitude of the monitoring location. | float | -117.449102931018 |
| seasonal_sin | Sine embedding of day of the year, to capture seasonality while avoiding discontinuities at calendar year boundaries. | float | -0.11988093 |
| seasonal_cos | Cosine embedding of day of the year, to capture seasonality while avoiding discontinuities at calendar year boundaries. | float | 0.99278826 |
| trend | A simple linear trend from January 1, 2000 to March 30, 2026; does not depend on the values in the table. The model can weight and incorporate this to account for long-term general trends. | float | 0.98288846 |
| rolling_7 | Rolling average of flow_rate over previous 7 days. | float | 2257.1428571428600 |
| rolling_14 | Rolling average of flow_rate over previous 14 days. | float | 2260.0 |
| rolling_21 | Rolling average of flow_rate over previous 21 days. | float | 2159.0476190476200 |

# Data preparation paragraph

4-6 sentences on approach, what we are thinking while prompting AI, what we wanted to accomplish, why we chose certain feature engineering variables. Include role of AI in each step (how it performs, how we assess its performance)
