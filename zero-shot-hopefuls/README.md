**Problem Statement:**

Climate change is intensifying compound hazards, such as co-occurring heatwaves and droughts, or wildfires during these conditions, that threaten ecosystems, agriculture, water resources, human health, and infrastructure in the United States. Traditional forecasting systems often fail to capture complex interactions and spatio-temporal dependencies, hindering the proactive decision-making required to safe lives, environments, and infrastructure. Forecasting systems that consider these compound hazards are more likely to give first responders and the public time to prepare and react in a manner that minimizes risk and damage.

**Goal:**

Our goal is to analyze how co-occurring heatwaves and droughts across the contiguous United States elevate wildfire risk, and to build a location-aware model that forecasts compound hazard probability 1–2 weeks ahead using historical climate and fire occurrence patterns. Early prediction of these compounding conditions enables emergency managers, firefighting agencies, and local communities to transition from reactive crisis response to proactive resource allocation and risk communication. This research can improve preparedness and resilience across fire-prone U.S. regions — particularly the West and Southwest — where compound climate hazards are increasingly frequent and severe.

**Data source:**

["AI-ready Benchmark Data for US Fires, Heatwaves, and Droughts"](https://www.pnnl.gov/projects/essd-ai-challenge)

**Team Structure:**

| Name                    | Role                               | Responsibilities                                                                                                                                                                                          |
| ----------------------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Avi Joshi               | LLM Lead                           | Own the LLM solution path,build & refine prompts, benchmark LLM vs ML performance, guide use of retrieval, data prep for LLMs                                                                             |
| Sarah Saslow            | GitHub Apprentice                  | Learn Github, help with dataset cleanup, exploratory analysis, versioning, mainting the data folder, track data transformations                                                                           |
| Leah Hare               | Documentation and Workflow Owner   | Maintain the README, weekly submissions, write explanations for the project, methods, and insights, coordinate deadlines and keep track of deliverables, ensure everything aligns with the scoring rubric |
| Carolyn Pearce          | Analysis & Evaluation Specialist   | Help compare ML vs LLM results, identify insights, tradeoffs, surprises in outputs, assist in interprestation and storytelling with the leads, collaborate on visuals, figures, error analysis summaries  |
| Kevin Kautzky           | Presentation & Communications Lead | Own final presentation narrative, build overview slides explaining the problem, methods, and results, ensure the final demo is clear and compelling                                                       |
| Etienne Fluet-Chouinard | AI Lead/ML Approach Owner          | Lead the machine learning approach, decide on algorithms, evaluate performance, guide others on understanding model limitations, own reproducibility for the machine learning pipeline                    |

**Chat history link 1:** [Link](./gpt-5-mini.md) (Model Used: GPT-5 mini)

**Chat history link 2:** [Link](./claude-haiku.md) (Model used: Claude Haiku 4.5)

**Data Discovery Paragraph:**

The data set selected is extensive, spanning over 2 decades worth of compound hazard tracking and spatially resolved data. This will offer an opportunity to train, validate, and test the machine learning models used to generate the predictions outlined in our team's stated goal. The data is captured in the database (dbf) files and linked to spatial coordinates through the .shp files that are often used by ArcGIS/GIS tools to spatially map trends. This will be advantageous when identifying high risk regional areas. Included data in the subsets evaluated for the purpose of this activity include region identifiers, land and water surface areas, and total regional area.

**LLM Reflection Paragraph:**

Using the identical three question prompt across the two data‑exploration chats, both models emphasized modeling strategy over actually interrogating the dataset, leading to two clear gaps: neither enumerated or validated the available columns/variables, and both made unverified assumptions about data contents (e.g., referencing indices like SPI/SPEI/PDSI, aerosol BC/OC, ecosystem type, and distance to coastline, or defining compound‑event labels and thresholds) without confirming they exist in the files; additionally, neither grounded feature engineering in the dataset’s native spatial/temporal resolution, risking misalignment. In approach, Claude Haiku 4.5 delivered a concise, high‑level framework that was clear but generic and light on dataset specifics, while GPT‑5 mini produced a more granular, task‑oriented plan with concrete inputs (e.g., MOD14A1, MCD64A1/FIRED, CPC, gridMET, NARR/MERRA‑2, LAI/NDVI, FWI), validation metrics, calibration, and SHAP analyses—more actionable but cluttered by meta/tool‑management commentary and a self‑contradiction about “not naming models” before naming them. Given the verbosity and scope, time to first actionable data‑discovery result would likely be shorter with Claude (roughly 5–10 minutes to a usable outline) but require follow‑up to resolve missing specifics, whereas GPT‑5 mini’s detailed checklist could take longer upfront (about 15–25 minutes) yet streamline subsequent execution. Claude appears to have used slightly fewer output tokens (3,203 total); GPT‑5 mini expended more tokens (3,584 total)to add specificity and process overhead, suggesting a trade‑off where a blended strategy—start with a lean, column‑aware inventory, then layer in detailed modeling steps—balances token economy with accuracy.
