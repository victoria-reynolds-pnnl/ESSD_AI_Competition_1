**Problem Statement:**  
We aim to use the CONUS portion of the ComExDBM benchmark to predict whether a grid cell will experience fire activity on a given day or week using co-located drought, meteorological, vegetation, and lagged fire features. This could help identify elevated fire-risk conditions in a way that is practical for short-term screening and easy to compare across modeling approaches. We will compare a traditional ML classifier with an LLM-based approach on the same target, split, and evaluation metrics.

**Goal:**  
Develop and compare traditional ML and LLM-based binary classifiers that predict fire occurrence in 0.5° × 0.5° CONUS grid cells, achieving measurable performance on held-out temporal test data while maintaining rigorous validation discipline to avoid spatial and temporal leakage.

**Data source:** `An Inventory of AI-ready Benchmark Data for US Fires, Heatwaves, and Droughts`

**Team Structure:**  
| Name | Role | Responsibility |
| -------------| ------ | ------------------- |
| Haiden Deaton | Individual contributor | Lead contributor (provided chat-history-1) |
| Brian Dean | Individual contributor | Co-lead contributor (provided chat-history-2) and pull request submitter |
| Jace Olsen | Individual contributor | problem formation, investigator |
| Jason Pope | Individual contributor | problem formation, investigator |
| Habilou Ouro-Koura | Individual contributor | problem formation, investigator |
| Ilan Gonzalez-Hirshfeld | Individual contributor | problem formation, investigator |

**Chat history link 1:** Chat history can be found in the `./chat-history-1` directory.  
**Chat history link 2:** Chat history can be found in the `./chat-history-2` directory.

**Data Discovery Paragraph:**  
The ComExDBM_CONUS dataset provides daily extreme-weather data from 2001 to 2020 at 0.5° × 0.5° (~55 km × 55 km) spatial resolution across the contiguous United States, integrating heatwave labels, fire-related features (hotspot counts, fire radiative power proxies, burned area), drought indices (PDSI, SPI, SPEI), and co-located meteorological variables from multiple reanalysis sources including NARR and MERRA-2. The dataset is structured into four primary output categories (HeatWave, FireData, DroughtData, Meteorological_Variables) stored as yearly NetCDF files, making it unusually ML-ready with pre-aligned spatiotemporal grids and documented event labels. However, several biases and limitations must be acknowledged: the benchmark aggregates data from multiple source systems with different native resolutions and measurement assumptions, creating potential inconsistencies; event labels represent operational definitions rather than absolute ground truth (e.g., heatwave indices depend on percentile thresholds); fire occurrence exhibits severe class imbalance with most grid cell-days showing no fire activity; the coarse 0.5° resolution limits fine-scale risk assessment and may mask localized extreme events; and strong spatial and temporal autocorrelation between adjacent cells and consecutive days makes random row splits invalid, requiring time-based or space-time blocked validation strategies to prevent leakage and ensure realistic generalization estimates.

**LLM Reflection Paragraph:**

Overall, both models demonstrated high fidelity to the dataset documentation with **zero factual contradictions**. Codex provided rapid dataset exploration and ranked problem candidates by execution feasibility, while Claude Sonnet 4.5 formalized the goal with explicit success metrics and validation requirements. Using a third-party model, we saw an overall agreement of 82% on core decisions (CONUS selection, fire occurrence classification as primary benchmark, rejection of WUS for timeline reasons). A significant difference emerged between the two models in depth of bias analysis. Codex identified biases generically, while Claude provided a more granular treatment, which directly addressed potential downstream modeling failures and informed validation design (e.g., time-based and spatial-block cross-validation requirements).  
In terms of error detection and QC, there were two errors. First, we saw that Codex did not explicitly address temporal or spatial leakage in its initial problem ranking, while Claude addressed in specifically, with actual mitigation steps. The second error, was that neither model initially listed all available meteorological sources (NARR, MERRA-2 were initially omitted).  
As for clock time for responses, we did not record the time, however, the initial data discovery for both models was between 15-20 seconds to produce their response. And we see the token usage at the top of the chat logs in `./chat-history-2`.
