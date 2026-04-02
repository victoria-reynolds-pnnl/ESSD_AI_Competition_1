
# Sub‑Terra Tensor Tinkerers - Week 2 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Team:** Scott Unger, Julian Chesnutt, Theresa Pham, Xi Tan, Ashton Kirol, Mike Parker

1. The original un-cleaned data source 
	- Full dataset link: https://gdr.openei.org/submissions/1730
	- Example of original data:

2. The cleaned dataset with at least 3 new engineered features in csv and json format 
	- Example of cleaned data: https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_2/SubTerraTensorTinkerers/Data/FTES_1hour_cleaned.csv
	
3. A data dictionary that explains each feature, format, and context for each data source 
	- https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_2/SubTerraTensorTinkerers/Data/FTES_1hour_cleaned_data_dictionary.json

4. Any cleaning scripts used with citations in comments for sections generated with AI 
	- https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_2/SubTerraTensorTinkerers/Scripts/ftes_data_cleaning.py

5. Your selected ML Algorithms with 1-2 sentences justifying your choices 

	To evaluate model performance, we structured our approach as a progressive comparison framework — moving from a classical statistical baseline through ensemble machine learning to deep learning — allowing us to assess whether added model complexity yields meaningful forecasting improvements for injection pressure and producer temperature/flow prediction at the SURF FTES testbed.

	**Statistical Baseline (Kalman Filter):**

	A Kalman filter provides a computationally lightweight, interpretable benchmark rooted in state-space estimation, giving us a principled classical reference point to evaluate whether machine learning models offer genuine predictive improvements for this short-horizon forecasting task.

	**Random Forest/XGBoost:**

	Tree-based ensemble methods are well-suited for capturing nonlinear relationships in tabular, high-frequency sensor data and offer strong performance with interpretable feature importance, serving as a robust intermediate benchmark that several team members are already familiar with.

	**LSTM/GRU:**

	Recurrent neural network architectures like LSTM and GRU are designed to learn temporal dependencies across sequential data, making them well-suited for modeling the lagged, dynamic relationships between injection and production well telemetry.

	*<This justification was generated with the assistance of Claude Sonnet 4.5 (Anthropic).>*

6. A requirements.txt containing python environment dependencies 
	- https://github.com/victoria-reynolds-pnnl/ESSD_AI_Competition_1/blob/SubTerraTensorTinkerers_week_2/SubTerraTensorTinkerers/requirements.txt

8. A brief paragraph describing data preparation approach, including: 
	- A description of data cleaning, normalization, and feature engineering steps 

		The FTES dataset was cleaned by first aligning the telemetry to a complete hourly time index, removing duplicate timestamps, converting known placeholder values such as -500 to missing values, and dropping columns with more than 5% missing data. Remaining gaps were imputed with short-window forward fill and then column medians, while obvious anomalies were corrected by clipping negative flows, unrealistic pressures, and out-of-range temperatures; electrical conductivity negatives were flagged for review; and zero depth values were replaced with typical median depths. The script also reduced spikes using z-score capping and short rolling-median smoothing. Feature engineering added operational labels and derived predictors including Phase, Total_Production_Flow, Pressure_Differential, a Depth_Consistency_Flag, an impedance-style proxy for the injection well, and EC-based mixing-fraction features that estimate how much injected-water signature appears at each monitoring location.
		In preparation for a time-based ML model, time-series features were then added to capture temporal dynamics, such as lagged values, 24-hr offsets, and rolling-window statistics. For modeling readiness, the pipeline classifies variables by sensor type and prepares them for scaling with MinMaxScaler, while filtering the dataset to the hot-injection window used for analysis. 
		

	- The role of AI at each step

		AI was used as a workflow support tool at several stages of the pipeline rather than as a replacement for domain judgment. During data cleaning, it helped identify likely preprocessing steps, suggest handling strategies for missing values, placeholder values, and sensor anomalies, and speed up script drafting for those operations. During normalization and preparation, AI assisted in organizing variables by type, proposing scaling and filtering steps appropriate for time-series sensor data, and helping structure the preprocessing pipeline for downstream modeling. In feature engineering, AI was used to brainstorm and draft derived variables such as total production flow, pressure differential, impedance-style proxies, phase labels, and conductivity-based mixing fractions, which were then implemented and reviewed in code. Overall, AI mainly accelerated coding, documentation, and idea generation, while the team remained responsible for selecting, validating, and interpreting the final methods.