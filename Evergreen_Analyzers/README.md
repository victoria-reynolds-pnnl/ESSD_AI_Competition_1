# ML model selection

# ML model justification

# Data dictionary tables

Cleaned and engineered dataset is in `Data/data_cleaned.csv`. The data covers
50 monitoring locations from January 1, 2000 to March 30, 2026. The columns
in the data are:

'monitoring_location_id' - The USGS id for the monitoring location.
'time' - The date that the measurement was taken.
'flow_rate' - The average flow rate at the location on the given date.
'qualifier' - Qualifiers regarding data quality. See data preparation paragraph
    for details on qualifiers and which were omitted. 
'latitude' - Latitude of the monitoring location.
'longitude' - Longitude of the monitoring location.
'seasonal_sin' - Sine embedding of day of the year, to capture seasonality
    while avoiding discontinuities at calendar year boundaries.
'seasonal_cos' - Cosine embedding of day of the year, to capture seasonality
    while avoiding discontinuities at calendar year boundaries.
'trend' - A simple linear trend from January 1, 2000 to March 30, 2026, does
    not depend on the values in the table. Model will be able to weight and
    incorporate this number to account for long-term general trends that have
    occurred in the data.
'rolling_7' - Rolling average of flow_rate over previous 7 days.
'rolling_14' - Rolling average of flow_rate over previous 14 days.
'rolling_21' - Rolling average of flow_rate over previous 21 days.

# Data preparation paragraph

4-6 sentences on approach, what we are thinking while prompting AI, what we wanted to accomplish, why we chose certain feature engineering variables. Include role of AI in each step (how it performs, how we assess its performance)