
Problem Statement:
Climate change is projected to intensify the frequency, severity, and duration of extreme thermal events across the United States, yet empirically grounded, region-specific baselines quantifying how heat wave and cold snap characteristics have already evolved remain largely absent at the scale most relevant to energy reliability and infrastructure planning — the NERC subregion. The Extreme Thermal Event Library, comprising thousands of events across 45 years, 16 NERC subregions, three spatial aggregation methods, and 12 event definitions, offers an unprecedented opportunity to establish these baselines and to detect whether the climatic shifts anticipated by climate science are already manifest in the historical record as measurable trends, regime shifts, or seasonal timing changes. Unlocking this potential requires a systematic analytical effort that transforms raw event records into the descriptive and inferential evidence base needed to inform grid resilience planning, public health preparedness, and climate adaptation strategy across the conterminous United States.

Goal:
Communities and the power grid across the contiguous United States are facing stronger heat waves and cold snaps, yet we lack clear, region-by-region baselines that show how these extremes have already changed where reliability is managed. Our goal is to analyze the North American Electric Reliability Corporation’s Extreme Thermal Event Library to produce a descriptive profile of extreme events and apply formal trend detection, change-point analysis, and phenological shift assessment to determine whether event frequency, intensity, duration, and seasonal timing are changing. The results will equip grid operators, utilities, emergency managers, and public health agencies with actionable evidence to guide preparedness and investment from the next season through 5–30 year resilience planning across the continental U.S.
Data source: https://data.pnnl.gov/group/nodes/dataset/34393

Team Structure:
Team Member 	Role 	Responsibilities 
Joshua Hargraves	Data engineer	Data analysis
Youngjun Son	Data engineering / domain expert	Data processing and curation
Shelby Brooks	Research Analyst	Team Administration; Research/documentation reviewer
Tracy Fuentes	Research Analyst	Data analysis, data visualization
Miguel Valencia	Modeling specialist / domain expert	Results visualization, feature engineering support
Christopher Thompson	Research analyst, data processing	Data analysis

Chat history link 1: https://ai-incubator-chat.pnnl.gov/s/57766197-7282-41e3-adbc-dfc4a5dc451b
Chat history link 2: https://ai-incubator-chat.pnnl.gov/s/cbce49a2-e573-4d0f-8405-387434b845e0

Data Discovery Paragraph: A selection of the data was reviewed in MatLab to understand surface level trends: Data Set 4 (Heat Wave and Cold Snap Events) Exploration
Thursday, March 26, 2026
10:24 AM
•	Data set consists of 6 folders
o	cold_snap_library_NERC_average
o	cold_snap_library_NERC_average_area
o	cold_snap_library_NERC_average_pop
o	heat_wave_library_NERC_average
o	heat_wave_library_NERC_average_area
o	heat_wave_library_NERC_aveage_pop
•	There are 12 Temperature-based event definitions (see below). Each data folder contains 12 .csv files; one for each event type (72 .csv files total).
•	Date range is 1980-2024
 
Table 1. Temperature-based event definitions taken from the peer-reviewed literature.
 	Temperature Metric	Heat Wave Threshold	Cold Snap Threshold	Duration	References
Def 1	Daily mean 	> 90th percentile 	< 10th percentile	2+ consecutive days	Anderson and Bell, 2011
Def 2	Daily mean 	> 95th percentile	< 5th percentile	2+ consecutive days	Anderson and Bell, 2011; 
Barnett et al., 2012
Def 3	Daily mean 	> 98th percentile	< 2nd percentile	2+ consecutive days	Anderson and Bell, 2011; 
 Barnett et al., 2012
Def 4	Daily mean 	> 99th percentile	< 1st percentile	2+ consecutive days	Anderson and Bell, 2011; 
 Barnett et al., 2012
Def 5	Daily maximum	> 95th percentile	< 5th percentile 	2+ consecutive days	Anderson and Bell, 2011
Def 6	Daily maximum	T1 > 97.5th percentile;
T2 > 81st percentile	T1 < 2.5th percentile;
T2 < 19th percentile	Everyday > T2; 
3+ consecutive days > T1;
Average Tmax > T1 for whole period	Peng et al., 2011; 
Meehl and Tebaldi, 2004
Def 7	Daily maximum	T1 > 90th percentile;
T2 > 75th percentile	T1 < 10th percentile;
T2 < 25th percentile	Everyday > T2; 
3+ consecutive days > T1;
Average Tmax > T1 for whole period	Lau and Nath, 2014; 
Luo and Lau, 2017
Def 8	Daily maximum	> 90th percentile (15-day
moving window)	< 10th percentile (15-day
moving window)	3+ consecutive days	Chen and Zhai, 2017
Def 9	Daily minimum	> 95th percentile	< 5th percentile	2+ consecutive days	Anderson and Bell, 2011
Def 10	Daily minimum	T1 > 97.5th percentile;
T2 > 81st percentile	T1 < 2.5th percentile;
T2 < 19th percentile	Everyday > T2; 
3+ consecutive days > T1;
Average Tmin > T1 for whole period	Liao et al., 2018
Def 11	Daily minimum	T1 > 90th percentile;
T2 > 75th percentile	T1 < 10th percentile;
T2 < 25th percentile	Everyday > T2; 
3+ consecutive days > T1;
Average Tmin > T1 for whole period	Liao et al., 2018
Def 12	Daily minimum	> 90th percentile (15-day 
moving window)	< 10th percentile (15-day 
moving window)	3+ consecutive days	Chen and Zhai, 2017
 
From <https://pnnl-my.sharepoint.com/personal/chris_thompson_pnnl_gov/Documents/Documents/Projects/ESSD%20AI%20Competition%202026/Data%20Sets/Heat%20Wave%20and%20Cold%20Snap%20Events/37832/37832_af3564fd87db67b7d96d15563e93668d.tar/Event%20definitions.docx> 
 
There is also a map showing the subregions with mean temperatures. Note that regions 13, 14, 16, and 19 are not identified on the map.
 
 
 
Screen clipping taken: 3/26/2026 10:35 AM
 
Structure of the data files:
 
 
 
Screen clipping taken: 3/26/2026 10:39 AM
 
 
 
 
Screen clipping taken: 3/26/2026 10:45 AM
 
The .csv files for each event do not have the same number of rows.
 
Exploratory Plots
 
Histograms (number of events) by Region for Simple Average, Event1 (<10/>90 Percentile)
 
 
 
Screen clipping taken: 3/26/2026 1:50 PM
 
Histograms (number of events) by Region for Simple Average, Event10 (<2.5/>97.5 Percentile)
 
 
 
Screen clipping taken: 3/26/2026 2:07 PM
 
Cold Snap Lows and Heat Wave Highs for Simple Average, Event1 (<10/>90 Percentile)
 
 
 
Screen clipping taken: 3/26/2026 1:47 PM

LLM Reflection Paragraph:  The LLM models are very verbose in their approach to the data, as can be seen in the chat history. To maintain a concise format and keep the project in a realistic scope, prompts will have to be refined going forward. From presenting in a broad scope to comparison to our initial findings, the group was able to pare down the LLM suggestions to a manageable subset, covering the extent of Claude and Gemini’s suggestion from “Phase 1” though most of “Phase 2.” Overall, Claude was chosen as our go-to model as it formatted the process most logically and sequentially.
