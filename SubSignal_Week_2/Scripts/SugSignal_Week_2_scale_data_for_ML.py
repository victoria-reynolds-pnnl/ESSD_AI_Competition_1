import os
import pandas as pd # 1.4.2
import numpy as np # 1.21.5

#os.chdir("Path to FTES_cleaned_pressure_with_engineered_features.csv")

#%% Drop flagged rows
cleaned = pd.read_csv("ftes_1hour_cleaned_reduced.csv", index_col = 0)
cleaned = cleaned[cleaned['flag_duplicate_timestamp']==False]
cleaned = cleaned[cleaned['flag_time_gap_gt_expected']==False]
cleaned.to_csv("ftes_1hour_cleaned_reduced_QC.csv")

#%% ML data prep workflow

# The following code to scale FTES data for GNN was generated with Co-pilot 
# and only slightly modified by the user 

# load csv
ftes_data = pd.read_csv("ftes_1hour_cleaned_reduced_QC_engineered.csv")

# Example column names (adjust to your actual naming)
monitor_wells = ["tl_interval", "tl_bottom", "tn_interval", "tn_bottom",
                 "tc_interval", "tc_bottom", "tu_interval", "tu_bottom",
                 "ts_interval", "ts_bottom"]

injection_well = "injection"

# Pressure columns per well (psi)
pressure_cols = {w: f"{w}_pressure" for w in monitor_wells}
pressure_cols[injection_well] = "injection_pressure"

# Flow (L/min) – only at injection well
inj_flow_col = "net_flow"

# Binary flags (0/1): injecting/pumping – only at injection well
# If you have a single column with values {inject=1, pump=0}, list it once.
flag_cols = ["tc_injecting", "tc_producing"]  # adjust to your case

# Ratio feature ΔP/ΔQ – if already provided per well, list them. 
# If you only have it at INJ, list just that.
#ratio_cols = {w: f"{w}_ratio_dP_dQ" for w in monitor_wells}  # if not available, you can skip
ratio_cols = {injection_well:"delta_P_delta_Q"}


#%% Prep data 

from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler # 1.0.2

df_train = ftes_data.copy()

@dataclass
class NormalizationConfig:
    pressure_cols: Dict[str, str]           # {node: col_name}
    ratio_cols: Dict[str, str]              # {node: col_name} (optional)
    inj_flow_col: Optional[str]             # single column name or None
    flag_cols: List[str]                    # list of binary columns


class Normalizer:
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.std_scalers: Dict[str, StandardScaler] = {}
        self.robust_scalers: Dict[str, RobustScaler] = {}
        # Flags: no scaler (pass-through)

    def fit(self, df_train: pd.DataFrame):
        # Fit StandardScaler for pressures
        for node, col in self.config.pressure_cols.items():
            assert col in df_train.columns, f"Missing pressure column: {col}"
            scaler = StandardScaler()
            self.std_scalers[col] = scaler.fit(df_train[[col]])

        # Fit StandardScaler for inj flow
        if self.config.inj_flow_col:
            col = self.config.inj_flow_col
            assert col in df_train.columns, f"Missing flow column: {col}"
            scaler = StandardScaler()
            self.std_scalers[col] = scaler.fit(df_train[[col]])

        # Fit RobustScaler for ratios if present
        for node, col in self.config.ratio_cols.items():
            if col in df_train.columns:
                scaler = RobustScaler(quantile_range=(10, 90))
                self.robust_scalers[col] = scaler.fit(df_train[[col]])
            else:
                # Missing ratio column -> will be imputed as 0 later
                pass

        # Verify flags exist (no fitting)
        for col in self.config.flag_cols:
            assert col in df_train.columns, f"Missing flag column: {col}"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()

        # Apply StandardScaler for pressures and flow
        for col, scaler in self.std_scalers.items():
            df_out[col] = scaler.transform(df[[col]])

        # Apply RobustScaler for ratios (or set to 0 if missing)
        for node, col in self.config.ratio_cols.items():
            if col in df.columns and col in self.robust_scalers:
                df_out[col] = self.robust_scalers[col].transform(df[[col]])
            else:
                # Ensure column exists and impute zeros
                if col not in df_out.columns:
                    df_out[col] = 0.0
                else:
                    # If present but no scaler (e.g., missing in train), standardize as zeros
                    df_out[col] = 0.0

        # Flags remain unchanged (0/1)
        # Optional: ensure flags are int 0/1
        for col in self.config.flag_cols:
            df_out[col] = df_out[col].astype(int)

        return df_out


# Instantiate the normalizer
config = NormalizationConfig(pressure_cols, ratio_cols, inj_flow_col, flag_cols)
norm = Normalizer(config)

# Fit on the TRAIN split only
norm.fit(df_train)

# Transform any split to get the normalized DataFrame
df_out = norm.transform(df_train)
df_out.to_csv("ftes_scaled_for_GNN.csv")