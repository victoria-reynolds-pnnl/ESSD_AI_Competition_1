{
  "project": {
    "name": "Yakima River Travel-Time & High-Water Modeling",
    "time_zone": "UTC",
    "sites": {
      "UnionGap": {"usgs_id": "12500450", "label": "Union Gap"},
      "Mabton": {"usgs_id": "12508990", "label": "Mabton"},
      "Kiona": {"usgs_id": "12510500", "label": "Kiona"}
    },
    "site_pairs": [
      {"id": "UG_MB", "upstream": "UnionGap", "downstream": "Mabton"},
      {"id": "MB_KI", "upstream": "Mabton", "downstream": "Kiona"},
      {"id": "UG_KI", "upstream": "UnionGap", "downstream": "Kiona"}
    ],
    "event_detection_parameters": {
      "rule1_description": "Onset of downstream rise (first timestamp where downstream derivative >= threshold after upstream rise).",
      "rule1_thresholds": {
        "initial": {"value": 0.05, "unit": "ft/hr", "notes": "Used in the first pass; applies when data are stage (ft)."},
        "retrained": {"value": 2.0, "unit": "ft/hr", "notes": "Used in retraining variant; applies when data are stage (ft)."}
      },
      "rule1_search_window_hours": 200,
      "rule2_description": "Downstream matches/exceeds upstream value at up_time.",
      "rule2_search_window_hours": 500
    }
  },

  "files": [
    {
      "file_name": "cleaned_1 Yakima River at Mabton, WA - USGS-12508990 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned discharge time series (Mabton).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "cfs", "description": "Discharge"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"],
      "notes": ["USGS discharge (cfs)"]
    },
    {
      "file_name": "cleaned_2 Yakima River at Kiona, WA - USGS-12510500 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned discharge time series (Kiona).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "cfs", "description": "Discharge"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"],
      "notes": ["USGS discharge (cfs)"]
    },
    {
      "file_name": "cleaned_3 Yakima River Above Ahtanum Creek at Union Gap, WA - USGS-12500450 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned discharge time series (Union Gap).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "cfs", "description": "Discharge"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"],
      "notes": ["USGS discharge (cfs)"]
    },
    {
      "file_name": "cleaned_1 Gage height USGS-12508990 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned gage height time series (Mabton).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "ft", "description": "Gage height (stage)"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"]
    },
    {
      "file_name": "cleaned_2 Gage height USGS-12510500 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned gage height time series (Kiona).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "ft", "description": "Gage height (stage)"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"]
    },
    {
      "file_name": "cleaned_3 Gage height USGS-12500450 010125-123125.csv",
      "type": "csv",
      "purpose": "Cleaned gage height time series (Union Gap).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "value", "type": "number", "unit": "ft", "description": "Gage height (stage)"}
      ],
      "processing": ["parsed datetime", "dropped invalid/missing", "deduped timestamps", "sorted ascending"]
    },
    {
      "file_name": "merged_resampled_yakima.csv",
      "type": "csv",
      "purpose": "Hourly-resampled merged series for the three sites (used for correlation and lag analysis).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "Mabton", "type": "number", "unit": "cfs", "description": "Resampled mean (hourly)"},
        {"name": "Kiona", "type": "number", "unit": "cfs", "description": "Resampled mean (hourly)"},
        {"name": "UnionGap", "type": "number", "unit": "cfs", "description": "Resampled mean (hourly)"}
      ],
      "processing": ["merged by timestamp", "resampled to 1H mean", "dropped rows with any missing"],
      "notes": [
        "Created from discharge (cfs) cleaned inputs in this workflow.",
        "If built from gage height files in another workflow, units would be ft."
      ]
    },
    {
      "file_name": "yakima_with_derivatives.csv",
      "type": "csv",
      "purpose": "Base merged series + first differences for each site (used for Rule 1 detection at 0.05 ft/hr in concept).",
      "schema": [
        {"name": "time", "type": "datetime", "format": "ISO-8601", "primary_key": true},
        {"name": "Mabton", "type": "number", "unit": "cfs"},
        {"name": "Kiona", "type": "number", "unit": "cfs"},
        {"name": "UnionGap", "type": "number", "unit": "cfs"},
        {"name": "deriv_Mabton", "type": "number", "unit": "cfs/hr", "description": "1-hour difference"},
        {"name": "deriv_Kiona", "type": "number", "unit": "cfs/hr", "description": "1-hour difference"},
        {"name": "deriv_UnionGap", "type": "number", "unit": "cfs/hr", "description": "1-hour difference"}
      ],
      "notes": [
        "If original data are stage (ft), derivatives would be ft/hr.",
        "Ensure unit consistency when applying thresholds."
      ]
    },
    {
      "file_name": "UG_MB_event_detection.csv",
      "type": "csv",
      "purpose": "Event detections for Union Gap → Mabton using initial Rule 1 threshold (conceptual 0.05 ft/hr) and Rule 2.",
      "schema": [
        {"name": "up_time", "type": "datetime", "format": "ISO-8601"},
        {"name": "rule1_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "rule2_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "pair", "type": "string", "nullable": true, "notes": "May be absent in this file; present in other pair outputs."}
      ],
      "notes": ["Times represent detected events; travel times are computed later."]
    },
    {
      "file_name": "MB_KI_event_detection.csv",
      "type": "csv",
      "purpose": "Event detections for Mabton → Kiona using initial Rule 1 threshold (conceptual 0.05 ft/hr) and Rule 2.",
      "schema": [
        {"name": "up_time", "type": "datetime", "format": "ISO-8601"},
        {"name": "rule1_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "rule2_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "pair", "type": "string", "example": "MB_KI"}
      ]
    },
    {
      "file_name": "UG_KI_event_detection.csv",
      "type": "csv",
      "purpose": "Event detections for Union Gap → Kiona using initial Rule 1 threshold (conceptual 0.05 ft/hr) and Rule 2.",
      "schema": [
        {"name": "up_time", "type": "datetime", "format": "ISO-8601"},
        {"name": "rule1_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "rule2_time", "type": "datetime", "format": "ISO-8601", "nullable": true},
        {"name": "pair", "type": "string", "example": "UG_KI"}
      ]
    },
    {
      "file_name": "yakima_training_matrix.csv",
      "type": "csv",
      "purpose": "Combined event detections with computed travel times and temporal features.",
      "schema": [
        {"name": "up_time", "type": "datetime"},
        {"name": "rule1_time", "type": "datetime", "nullable": true},
        {"name": "rule2_time", "type": "datetime", "nullable": true},
        {"name": "pair", "type": "string", "allowed": ["UG_MB","MB_KI","UG_KI"]},
        {"name": "travel_rule1_hr", "type": "number", "unit": "hours", "nullable": true},
        {"name": "travel_rule2_hr", "type": "number", "unit": "hours", "nullable": true},
        {"name": "month", "type": "integer", "range": [1,12]},
        {"name": "dayofyear", "type": "integer", "range": [1,366]}
      ]
    },
    {
      "file_name": "yakima_training_matrix_augmented.csv",
      "type": "csv",
      "purpose": "Training matrix augmented with downstream values and high-water labels (12 ft).",
      "schema": [
        {"name": "up_time", "type": "datetime"},
        {"name": "rule1_time", "type": "datetime", "nullable": true},
        {"name": "rule2_time", "type": "datetime", "nullable": true},
        {"name": "pair", "type": "string", "allowed": ["UG_MB","MB_KI","UG_KI"]},
        {"name": "travel_rule1_hr", "type": "number", "unit": "hours", "nullable": true},
        {"name": "travel_rule2_hr", "type": "number", "unit": "hours", "nullable": true},
        {"name": "month", "type": "integer"},
        {"name": "dayofyear", "type": "integer"},
        {"name": "downstream_at_rule1", "type": "number", "unit": "ft or cfs", "notes": "Matches base dataset unit."},
        {"name": "downstream_at_rule2", "type": "number", "unit": "ft or cfs", "notes": "Matches base dataset unit."},
        {"name": "high_water_warning_rule1", "type": "boolean", "description": "True if downstream_at_rule1 > 12 ft"},
        {"name": "high_water_warning_rule2", "type": "boolean", "description": "True if downstream_at_rule2 > 12 ft"}
      ],
      "notes": [
        "If the base merged dataset is discharge (cfs), 12 ft threshold is not directly applicable; convert to stage or define a discharge threshold."
      ]
    },
    {
      "file_name": "yakima_with_derivatives_rule1_2ft.csv",
      "type": "csv",
      "purpose": "Base merged series + derivatives used for retraining Rule 1 with 2 ft/hr threshold.",
      "schema": [
        {"name": "time", "type": "datetime", "primary_key": true},
        {"name": "Mabton", "type": "number", "unit": "cfs"},
        {"name": "Kiona", "type": "number", "unit": "cfs"},
        {"name": "UnionGap", "type": "number", "unit": "cfs"},
        {"name": "deriv_Mabton", "type": "number", "unit": "cfs/hr"},
        {"name": "deriv_Kiona", "type": "number", "unit": "cfs/hr"},
        {"name": "deriv_UnionGap", "type": "number", "unit": "cfs/hr"}
      ],
      "notes": [
        "Used as input for deriving events under Rule 1 (2 ft/hr) when the underlying dataset is stage (ft).",
        "If underlying data are discharge (cfs), adjust threshold to cfs/hr accordingly."
      ]
    },
    {
      "file_name": "model_rule1.pkl",
      "type": "pkl",
      "purpose": "RandomForestRegressor for travel_time (Rule 1) trained on augmented matrix (initial threshold pass).",
      "model_meta": {
        "estimator": "RandomForestRegressor(n_estimators=30, random_state=0)",
        "target": "travel_rule1_hr",
        "features": ["pair_code", "month", "dayofyear", "downstream_at_rule1", "downstream_at_rule2"],
        "label_space": "hours"
      }
    },
    {
      "file_name": "model_rule2.pkl",
      "type": "pkl",
      "purpose": "RandomForestRegressor for travel_time (Rule 2) trained on augmented matrix (initial threshold pass).",
      "model_meta": {
        "estimator": "RandomForestRegressor(n_estimators=30, random_state=0)",
        "target": "travel_rule2_hr",
        "features": ["pair_code", "month", "dayofyear", "downstream_at_rule1", "downstream_at_rule2"],
        "label_space": "hours"
      }
    },
    {
      "file_name": "high_model1.pkl",
      "type": "pkl",
      "purpose": "RandomForestClassifier for high-water warning at Rule 1 event time.",
      "model_meta": {
        "estimator": "RandomForestClassifier(n_estimators=30, random_state=0)",
        "target": "high_water_warning_rule1",
        "features": ["pair_code", "month", "dayofyear", "downstream_at_rule1", "downstream_at_rule2"],
        "label_space": "binary (True/False)"
      }
    },
    {
      "file_name": "high_model2.pkl",
      "type": "pkl",
      "purpose": "RandomForestClassifier for high-water warning at Rule 2 event time.",
      "model_meta": {
        "estimator": "RandomForestClassifier(n_estimators=30, random_state=0)",
        "target": "high_water_warning_rule2",
        "features": ["pair_code", "month", "dayofyear", "downstream_at_rule1", "downstream_at_rule2"],
        "label_space": "binary (True/False)"
      }
    },
    {
      "file_name": "model_rule1_deriv2.pkl",
      "type": "pkl",
      "purpose": "RandomForestRegressor retrained for Rule 1 travel time using 2 ft/hr threshold detection.",
      "model_meta": {
        "estimator": "RandomForestRegressor(n_estimators=20, random_state=0)",
        "target": "travel_rule1_hr",
        "features": ["pair_code", "month", "dayofyear"],
        "label_space": "hours"
      },
      "notes": [
        "Retraining used on-the-fly event detection from yakima_with_derivatives_rule1_2ft.csv.",
        "Feature set simplified versus initial training (no downstream_at_rule* features)."
      ]
    },
    {
      "file_name": "model_rule2_deriv2.pkl",
      "type": "pkl",
      "purpose": "RandomForestRegressor retrained for Rule 2 travel time alongside 2 ft/hr variant pipeline.",
      "model_meta": {
        "estimator": "RandomForestRegressor(n_estimators=20, random_state=0)",
        "target": "travel_rule2_hr",
        "features": ["pair_code", "month", "dayofyear"],
        "label_space": "hours"
      },
      "notes": [
        "Retraining used the same simplified feature set as Rule 1 deriv2."
      ]
    },
    {
      "file_name": "yakima_plot.png",
      "type": "png",
      "purpose": "Static visualization of time series for the three sites.",
      "schema": [],
      "notes": ["Generated from plotly; not tabular."]
    },
    {
      "file_name": "yakima_plot.json",
      "type": "json",
      "purpose": "Plotly figure JSON for time series visualization.",
      "schema": ["Plotly figure spec"],
      "notes": ["Can be rendered by Plotly to reproduce the interactive plot."]
    }
  ],

  "inference_requirements": {
    "feature_sets": {
      "initial_models": {
        "models": ["model_rule1.pkl", "model_rule2.pkl", "high_model1.pkl", "high_model2.pkl"],
        "features_required": ["pair_code", "month", "dayofyear", "downstream_at_rule1", "downstream_at_rule2"],
        "labels": ["travel_rule1_hr", "travel_rule2_hr", "high_water_warning_rule1", "high_water_warning_rule2"]
      },
      "deriv2_models": {
        "models": ["model_rule1_deriv2.pkl", "model_rule2_deriv2.pkl"],
        "features_required": ["pair_code", "month", "dayofyear"],
        "labels": ["travel_rule1_hr", "travel_rule2_hr"]
      }
    },
    "pair_code_mapping": {"UG_MB": 0, "MB_KI": 1, "UG_KI": 2},
    "alert_policy": {
      "threshold_stage_ft": 12.0,
      "alert_phrase": "high water warning",
      "trigger_when": [
        "Predicted downstream stage exceeds 12 ft at Rule 1 or Rule 2 event time",
        "Input level already exceeds 12 ft at any monitored site"
      ]
    }
  },

  "quality_and_notes": [
    "Unit consistency is critical. The merged file in this workflow was built from discharge (cfs); thresholds specified in ft/hr apply when using stage data. Convert units or set appropriate discharge thresholds if needed.",
    "Event detection windows (200h for Rule 1, 500h for Rule 2) can be tuned.",
    "Travel-time targets are non-negative in typical cases; nulls indicate no event found within the window.",
    "Consider adding rolling-window features and upstream lag features to improve predictive performance."
  ]
}