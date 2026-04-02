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

## Data Cleaning Script

### Overview

This project includes a Python script (`clean_data.py`) designed to clean and standardize a set of weather event data. The script automates the cleaning process for all `.zip` archives located in the `data/original/` directory.

The cleaning logic specifically addresses several data quality issues discovered in the source files:

- **Inconsistent Formatting**: The source data contains a mix of concatenated fixed-width lines and standard comma-separated (CSV) lines, which require parsing [1, 3, 5].
- **Data Contamination**: The files contain miscategorized events. For example, cold snap archives include summer heat events [1, 2], and heat wave archives include winter cold snaps [3, 4, 5].
- **Junk Rows**: Some files contain non-data rows, such as headers or filenames, that must be removed [1].

The script parses each format, converts columns to appropriate data types, removes duplicates, and filters out the contaminated records to produce clean, usable CSV files.

### Prerequisites

- Python 3.7 or newer
- The `pandas` library

### Directory Structure

Before running the script, ensure your project is organized with the following directory structure:

project_folder/
├── Data/
│ ├── original/
│ ├── cleaned/
│ └── cleaned_with_features/
│ ├── csv/
│ └── json/
└── Scripts/

- **`Data/original/`**: This folder must contain all the original data `.zip` files (e.g., `cold_snap_library_NERC_average.zip` [1], `heat_wave_library_NERC_average_pop.zip` [5]).
- **`Data/cleaned/`**: This folder will be created by the cleaning script if it doesn't exist. It stores cleaned CSV output files.
- **`Data/cleaned_with_features/`**: This folder is produced by the feature script and contains mirrored outputs in both CSV and JSON formats.
- **`Scripts/`**: This folder must contain the `clean_data.py` script, the `add_features.py` script, and the `requirements.txt` file.

### Setup and Execution

1.  **Place Files**:
    - Move all the provided `.zip` data archives into the `Data/original/` directory.
    - Ensure `clean_data.py`, `add_features.py`, and `requirements.txt` are located in the `Scripts/` directory.

2.  **Navigate to the Scripts Directory**:
    Open your terminal or command prompt and change your current directory to the `Scripts` folder.

    ```sh
    cd path/to/project_folder/Scripts
    ```

3.  **Install Dependencies**:
    Run the following command to install the required `pandas` library.

    ```sh
    pip install -r requirements.txt
    ```

4.  **Execute the Cleaning Script**:
    Run the cleaning script from within the `/Scripts` directory. The script will log its progress in the terminal.

    ```sh
    python clean_data.py
    ```

5.  **Execute the Feature Engineering Script**:
    After cleaning completes, run the feature script from within the same `/Scripts` directory.
    ```sh
    python add_features.py
    ```

### Output

After `clean_data.py` finishes, the cleaned data will be located in the `Data/cleaned/` directory. The output is organized into subdirectories named after the original `.zip` archives.

For example, after processing `cold_snap_library_NERC_average.zip` [1], the output structure will be:

Data/
└── cleaned/
└── cold_snap_library_NERC_average/
├── \_def1_cleaned.csv
├── \_def2_cleaned.csv
└── ... and so on for each internal CSV

After `add_features.py` finishes, engineered outputs are written under `Data/cleaned_with_features/` in two mirrored folder trees:

- **CSV output** in `Data/cleaned_with_features/csv/`
- **JSON output** in `Data/cleaned_with_features/json/`

For each cleaned input CSV, the script writes a matching CSV and JSON file while preserving the same relative folder structure.

For example, a cleaned input file at:

`Data/cleaned/cold_snap_library_NERC_average/cold_snap_library_NERC_average_def1_cleaned.csv`

produces:

`Data/cleaned_with_features/csv/cold_snap_library_NERC_average/cold_snap_library_NERC_average_def1_cleaned.csv`

and

`Data/cleaned_with_features/json/cold_snap_library_NERC_average/cold_snap_library_NERC_average_def1_cleaned.json`
