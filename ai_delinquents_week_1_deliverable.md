# AI Delinquents Week 1

## Problem Statement:
Multiple factors can influence the need to retain or remove current dams in Washington state. Using streamflow data and public resources, it is essential to evaluate which dams are likely to be removed due to social, political, engineering, and environmental factors. This system will assess risks, identify at-risk dams, and highlight gaps in data needed for reliable risk profiles, aiding decision-makers in mitigating risks and ensuring safety.

## Goal: 
Using streamflow data and other public resources, come up with a risk assessment for dams that may be at high risk of removal due to social, political, engineering, and/or environmental factors. Build a system that can assess the risk, and as well as identify missing data that is needed to compute a reasonable risk profile.

## Data source(s):
| Category                         | Link(s)                                                                                                                                               |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| USGS/WA state water data         | (USGS)(https://waterdata.usgs.gov/state/Washington/)                                                                                                  |
| FERC License docs                | [FERC Licensing](https://www.ferc.gov/licensing)                                                                                                      |
| EIA data                         | [EIA 930](https://www.eia.gov/electricity/gridmonitor/about)                                                                                          |
|                                  | [EIA 923](https://www.eia.gov/electricity/data/eia923/)                                                                                               |
|                                  | [EIA 860](https://www.eia.gov/electricity/data/eia860/)                                                                                               |
|                                  | [Catalyst Cooperative PUDL](https://catalyst.coop/pudl/)                                                                                              |
| BPA transmission and generation  | [BPA Data](https://transmission.bpa.gov/business/operations/misc/eia/)                                                                                |
| USGS flow data                   | [USGS Flow](https://api.waterdata.usgs.gov/)                                                                                                          |
| USACE hydropower data            | [USACE Hydropower](https://www.nwd-wc.usace.army.mil/dd/common/dataquery/www/)                                                                        |
| Census (social data)             | [ACS Census Data](https://www.census.gov/programs-surveys/acs.html)                                                                                   |
|                                  | [Washington State Social-Economic Data](https://ofm.wa.gov/data-research/washington-trends/social-economic-conditions/#section-families-in-poverty)   |
|                                  | [ACS 5-Year Data](https://www.census.gov/data/developers/data-sets/acs-5year.html)                                                                    |
| Environmental impacts            | [Dam Impact Metrics](https://www.usgs.gov/data/dam-impactdisturbance-metrics-conterminous-united-states-1800-2018)                                    |
| Washington state environmental   | [WA Energy Analysis](https://www.eia.gov/states/wa/analysis)                                                                                          |
|                                  | [WA Ecology Database](https://apps.ecology.wa.gov/eim/search/default.aspx)                                                                            |
| Political                        | [Columbia River Treaty](https://www.state.gov/bureau-of-western-hemisphere-affairs/columbia-river-treaty)                                             |

## Team Structure:

| Team Member         | Role                                 | Responsibilities                    |
|---------------------|--------------------------------------|-------------------------------------|
| @cameronbracken     | Domain, compute, data                | Data managment, AI techniques       |
| @stefan-rose        | Geospatial analysis, data processing | Data analysis and viz               |
| @MindyDeLong-Weetch | Research Analyst, Risk Management    | Program and project risk management |
| @mdsturtevant-pnnl  | Software Engineer                    | Flashy deliverables?                |
| @amanda-lawter      | Resarch Analyst, data processing     | Data analysis                       |
| @JanaSimo           | Subsurface Science                   | Data analysis, application          |

### Chat history link 1 (Claude):
[Claude Opus/Sonnet 4.6](idea1_claude_chat_history_2026-03-20.md)

### Chat history link 2 (GPT):
[GPT-5.3-Codex](idea1_gpt_chat_history_2026-03-23.md)

## Data Discovery Paragraph:
**A single paragraph detailing data discovery that describes the data and addresses
potential biases and data limitations.**

X


## LLM Reflection Paragraph:
**A single paragraph that: Details at least 2 LLM Errors or missing QC steps identified during data discovery
and prompt engineering; Compares the responses of each model, highlighting observed differences; Quantitative LLM vs. LLM evaluation (Time to complete QC (wall clock timing from entering prompt to result), Inter-model agreement rate – differences between LLMs on which values within the data array changed, resulting values data was changed to, any hallucinations (bad data, new data)? Output tokens – which LLM used fewer output tokens, or is there a balance between number of tokens & accuracy to consider?**

X
