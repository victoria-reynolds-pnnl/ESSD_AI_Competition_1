## Model Selection

### Modeling Approach

Our analytical approach employs a sequence of machine learning models to identify, score, and analyze the risk of extreme weather events. The pipeline is as follows:

- **Unsupervised Risk Tiering:** `HDBSCAN` is used to group events into initial risk tiers without prior labels.
- **Supervised Risk Scoring:** `XGBoost` (eXtreme Gradient Boosting) is then used to assign a specific risk score to each event.
- **Spatial Clustering Analysis:** `Moran's I` is applied to test for the statistical significance of geographic clustering of vulnerable regions.
- **Recurrence Trend Analysis:** A `Random Survival Forest` is used to analyze the time-to-event and recurrence trends of heat waves and cold snaps.

### Justification

The selection of these models is based on the specific characteristics of the dataset and the analytical goals:

- **Suitability for Tabular Data:** The dataset consists of structured tabular records in CSV format with a mix of numeric features (e.g., duration, temperature, spatial coverage) and categorical identifiers [1, 2, 3, 4, 5]. Tree-based ensembles like XGBoost and Random Survival Forest are exceptionally well-suited for this type of heterogeneous data.
- **Robustness and Interpretability:** Extreme weather events are inherently rare, leading to class imbalance. Tree-based models are robust to this issue. Furthermore, they provide strong feature importance outputs, which are crucial for interpretability and communicating findings to stakeholders.
- **Advanced Clustering:** HDBSCAN is preferred over simpler methods like k-means for the initial tiering because the NERC subregions form clusters of varying shapes and densities, which HDBSCAN is designed to handle effectively.
- **Geospatial Significance:** Moran's I is the standard statistical method to directly address the project's requirement to identify statistically significant geographic clustering of events, rather than just observing visual patterns.

## Data Dictionary Table

| Field Name | Description | Data Type(s) | Example(s) |
|---|---|---|---|
| `start_date` | Event start date in ISO format (YYYY-MM-DD). | `string` | `1980-01-06` |
| `end_date` | Event end date in ISO format (YYYY-MM-DD). | `string` | `1980-01-12` |
| `centroid_date` | Representative midpoint date for the event in ISO format (YYYY-MM-DD). | `string` | `1980-01-11` |
| `duration_days` | Event duration measured in whole days. | `integer` | `7` |
| `nerc_id` | NERC region identifier associated with the event. | `string` | `NERC` |
| `spatial_coverage` | Spatial extent metric for the event (coverage in dataset-defined units). | `number` | `48.0` |
| `highest_temperature_k` | Peak heat-wave temperature value in Kelvin. | `number` | `311.7061073825` |
| `lowest_temperature_k` | Minimum cold-snap temperature value in Kelvin. | `number` | `258.5427111111` |
| `inter_event_recovery_interval_days` | Gap in days between the current event end date and the next event start date within the same region. | `number`, `null` | `6.0`, `null` |
| `cumulative_heat_stress_index` | Running cumulative stress index within each region-year, based on event temperature multiplied by duration. | `number` | `1809.7989777778` |
| `yearly_max_heat_wave_intensity` | Maximum event temperature observed for the region in that calendar year. | `number` | `272.7129824561` |
| `yearly_max_heat_wave_duration` | Maximum event duration observed for the region in that calendar year. | `integer` | `17` |
| `yearly_max_heat_wave_intensity_trend` | Year-over-year change in yearly maximum intensity within each region. | `number`, `null` | `0.6321052632`, `null` |
| `yearly_max_heat_wave_duration_trend` | Year-over-year change in yearly maximum duration within each region. | `number`, `null` | `6.0`, `null` |

## Data Cleaning Script
- `clean_data.py` cleans and standardizes all `.zip` archives in `Data/original/` into consistent CSVs.
- Handles mixed input formats (concatenated fixed-width vs. standard CSV), removes junk/non-data rows, converts columns to appropriate data types, drops duplicates, and filters miscategorized/contaminated events (e.g., heat events in cold-snap files and vice versa).
- Outputs cleaned files to `Data/cleaned/`, organized into subfolders that mirror each source archive.

## Data Features Added Notes

- The feature-adding script was produced in VS Code using a GitHub Copilot agent.

## Feature Engineering Notes

- Gemini 2.5 Pro, GPT 5.2, and Claude Sonnet 4.6 were used to generate ideas for feature engineering, and the team selected three of these features to move forward with.