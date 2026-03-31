
# Sub‑Terra Tensor Tinkerers - Week 2 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Team:** Scott Unger, Julian Chesnutt, Theresa Pham, Xi Tan, Ashton Kirol, Mike Parker

1. The original un-cleaned data source 
	- Full dataset link: https://gdr.openei.org/submissions/1730
	- Example of original data:

2. The cleaned dataset with at least 3 new engineered features in csv and json format 
	- Example of cleaned data: **[add link]**
	
3. A data dictionary that explains each feature, format, and context for each data source 
	- **[add link]**

4. Any cleaning scripts used with citations in comments for sections generated with AI 
	- **[add link]**

5. Your selected ML Algorithms with 1-2 sentences justifying your choices 

To evaluate model performance, we structured our approach as a progressive comparison framework — moving from a classical statistical baseline through ensemble machine learning to deep learning — allowing us to assess whether added model complexity yields meaningful forecasting improvements for injection pressure and producer temperature/flow prediction at the SURF FTES testbed.

**Statistical Baseline (Kalman Filter):**
A Kalman filter provides a computationally lightweight, interpretable benchmark rooted in state-space estimation, giving us a principled classical reference point to evaluate whether machine learning models offer genuine predictive improvements for this short-horizon forecasting task.

**Random Forest/XGBoost:**
Tree-based ensemble methods are well-suited for capturing nonlinear relationships in tabular, high-frequency sensor data and offer strong performance with interpretable feature importance, serving as a robust intermediate benchmark that several team members are already familiar with.

**LSTM/GRU:**
Recurrent neural network architectures like LSTM and GRU are designed to learn temporal dependencies across sequential data, making them well-suited for modeling the lagged, dynamic relationships between injection and production well telemetry.

*This justification was generated with the assistance of Claude Sonnet 4.5 (Anthropic).*

6. A requirements.txt containing python environment dependencies 
	- **[add link]**

8. A brief paragraph describing data preparation approach, including: 
	- A description of data cleaning, normalization, and feature engineering steps 
	- The role of AI at each step
