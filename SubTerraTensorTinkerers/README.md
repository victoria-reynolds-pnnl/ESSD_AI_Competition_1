# Sub‑Terra Tensor Tinkerers - Week 1 Submission

**Project Title:** Signal‑to‑Storage: Early Warning Forecasts for FTES Charging Operations

**Problem Statement:**

Fracture Thermal Energy Storage (FTES) stores heat in hot rock by injecting warm water into one well and producing water from nearby wells. Operators need to anticipate **rising injection pressures** (which can force them to reduce pumping rate) and **changes in produced water temperature and flow** (which determine how much useful heat is recovered). The challenge is to build a model that uses high‑frequency well telemetry to **forecast injection pressure and producer temperature/flow**, enabling earlier and better operational decisions.

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

**Chat history link 2:** 

**Data Discovery Paragraph:** 

**LLM Reflection Paragraph:**