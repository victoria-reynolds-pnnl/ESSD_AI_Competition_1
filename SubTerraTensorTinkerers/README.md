# Sub‑Terra Tensor Tinkerers - Week 1 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Problem Statement:**

Fracture Thermal Energy Storage (FTES) stores heat in subsurface rock by injecting warm water into one well and producing water from nearby wells. Operators need to anticipate **rising injection pressures** (which can force them to reduce pumping rate) and **changes in produced water temperature and flow** (which determine how much useful heat is recovered). The challenge is to build a model that uses high‑frequency well telemetry to **forecast injection pressure and producer temperature/flow**, enabling earlier and better operational decisions.

**Goal:** 

Develop a causal, short-horizon (1–24 hour, try different forecast horizons) forecasting model using second-resolution DEMO‑FTES Test 1 telemetry to predict **TC-well injection interval pressure** and **TN/TL-well production temperature and flow**, enabling early warning of pump-limit pressure escalation and changes in recoverable thermal output during hot-water injection and/or ambient-water recirculation operations.

Who
-   FTES/ Enhanced Geothermal Systems (EGS) operators and engineers planning and running thermal injection/production cycles.
-   Researchers developing control strategies to improve thermal recovery and avoid pump-limit interruptions.

What
-   Predict injection well pressure (operational constraint) and production well flow and temperature (thermal breakthrough, extracted energy) using high frequency injection and production well sensors.

Where
-   Sanford Underground Research Facility (SURF) meso-scale FTES testbed. Likely not generalizable to other sites given we are training on a specific single site.
-   Maybe try transfer learning for test cycle with two production wells

When
-   Likely a short forecast horizon (early warning system) 1–6 hours ahead. Can see how far we can push this out.

Why
-   Pressure escalation forecasting helps avoid exceeding pump constraints and supports proactive flow-rate adjustments.
-   Producer temperature and flow forecasting provides early indication of evolving connectivity/breakthrough and directly impacts recovered thermal energy (system value).
-   Together, these forecasts support improved operational efficiency and reliability for future FTES demonstrations and scale-up.

**Data source:** 

&emsp; DEMO-FTES Test 1: Water Flow, Pressure, Temperature, and Electrical Conductivity Data
&emsp; https://gdr.openei.org/submissions/1730

## Team Members

| Name | Role | Responsibilities |
| ---  | ---  | ---              |
| Ashton Kirol|ML lead|Organizer, Model training, evaluation metrics, interpretability, could help with GitHub|
| Scott Unger |Systems Engineer|Jack of all trades|
| Julian Chesnutt|Geologist|Subsurface SME, Define problem statement|
| Xi Tan|Computational Scientist|validation, error analysis, data quality/filtering|
| Theresa Pham|Data scientist|Model development, Github|
| Mike Parker|PM/TL|LLM prompting/benchmarking|

<br>

**Chat history link 1:**  
Claude Opus 4.6 (first half) and Gemini 2.5 Pro (second half)  
https://ai-incubator-chat.pnnl.gov/s/52fd5443-0a19-4dd3-803e-36ffc784445f

**Chat history link 2:**  
Claude Sonnet 4.6 and Gemini 2.5 Pro with added context of problem statement  
https://ai-incubator-chat.pnnl.gov/c/cb630a42-ea98-4cc9-a869-1a53a04dca7f

**Data Discovery Paragraph:**

There are two provided Fracture Thermal Energy Storage (FTES) datasets: a 1-second resolution dataset of water flow rate, electrical conductivity, pressure, and tempurature along several boreholes at multiple depths, and a lower resolution, 1-hour averaged version of the data. When considering the two resolutions of the provided data, there is a tradeoff between measurement frequency (granularity) and dataset size (indicative of training time) limitations.  
The ~3.5-month timeframe of the data ranges from December 2024 to March 2025, covering a hot water injection phase followed by an ambient temperature water injection phase, with periods of testing/maintenence in between. Data discovery in Link 1 chat history highlighted that there is no indication of the current operational phase in the data itself. If not split or labeled, measurements show drastically different values over time for seemingly no reason. The LLMs also noted inconsistent packer depths between time periods. Ideally, these depths  would be consistent to accurately monitor how conditions change over time at the same exact depth. To account for these differences, implemented models should include packer depth measurement as a feature.  
Data discovery in Link 2 cautioned the potential for sensor-related anomalies, such as sharp spikes, flat-lining, and irregular time stamps. These anomalies can be accounted for by identifying valid ranges for each variable, implementing cutoff thresholds, and checking variance. Another bias to consider is the specific time window and operational conditions of  data collection, limiting the model's training to a particular context. This bias would only become more apparent when attempting to capture seasonality, long-term trends, or different operating conditions, and is a point to keep in mind if this work is expanded.  

**LLM Reflection Paragraph:**

Potential LLM errors in the chat history links:

1. Almost all models cautioned for large time gaps or incorrectly ordered time stamps, and even provided examples of exact dates/times that should be flagged. However when manually observing the data, the time stamps seem to be correctly ordered without missing time steps.

2. Impossible negative or zero values of electrical conductivity (EC), pressure, or packer depth were flagged by the LLMs. After discussion with the team, there exist certain scenarios where these values could be valid (e.g, negative EC indicating the direction of current flow). Values in these columns will be reevaluated during data cleaning. 

When comparing the responses of the models, Claude took more time to think (Claude: ~1 minute, Gemini: a few seconds) and also provided a longer, more detailed response than Gemini (i.e., more output tokens). The two models aligned on many of the data biases/limitations presented, but differed between the two chat history links when prompted with the additional context of the problem statement. Although Claude provided more information overall, we found that the findings provided by Gemini were more useful. Claude tended to generate additional summarizations or observations that were not always needed for this particular use-case. 