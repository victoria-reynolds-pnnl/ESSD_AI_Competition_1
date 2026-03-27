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

## AI Costs

### Claude
```
❯ /cost                                                                                                                                                                                                    
  ⎿  Total cost:            $1.19
     Total duration (API):  4m 0s                                                                                                                                                                          
     Total duration (wall): 20m 27s                                                                                                                                                                      
     Total code changes:    488 lines added, 6 lines removed                                                                           
     Usage by model:                                                                                                                   
          claude-opus-4-5:  2.9k input, 11.5k output, 605.1k cache read, 93.9k cache write ($1.19)            
```

### GPT

At this time, we are unable to report the same detailed AI usage metrics from the GitHub Copilot extension in VS Code that are available in tools such as Cline. They were likley similar to that used by Claude. 

## AI chat history

### Chat history link 1 (Claude):
[Claude Opus/Sonnet 4.6](idea1_claude_chat_history_2026-03-20.md)

### Chat history link 2 (GPT):
[GPT-5.3-Codex](idea1_gpt_chat_history_2026-03-23.md)


## Data Discovery Paragraph:
**A single paragraph detailing data discovery that describes the data and addresses
potential biases and data limitations.**


### @cameronbracken
The datasets identified by claude were informed by my own bias towards quantitative datasets and water data. GPT has a more general set of datasets. Both datasets were light on implementation details about each method and dataset. 

### @stefan-rose
Key limitations include validation bias from the American Rivers Dam Removal Database, which overrepresents small, non-hydropower removals in the eastern U.S., incomplete and potentially inconsistent regulatory features from semi-manual FERC document extraction, missing Canadian coverage that leaves cross-border gaps for a transboundary system, and temporal misalignment between long-run disturbance metrics and more recent energy and Census data that may distort risk estimates for older infrastructure.

### @MindyDeLong-Weetch

### @mdsturtevant-pnnl

### @amanda-lawter
Based on the data sets we have identified for use, the category for social impact has limited data, much of which comes from self-reporting or surveys, which can present reporting bias and missing data. The data also comes from multiple agencies, each with different goals and objectives for the data they collect. This creates a potential for more diverse and inclusive data, but also creates potential for different information available for each dam we are evaluating. The data also varies from daily, annual, or as-needed data collection, creating a limitation in temporal evaluation for some data points. In addition, much of the data is higher level (state or federal) and may obscure local-level information that would be important particularly for some of the social or cultural indicators. Both LLMs used this week suggested adding two databases, the National Inventory of Dams and the American Rivers Dam Removal Database, that will help ensure the baseline data for dams and dam removals is available. 


### @JanaSimo 

## LLM Reflection Paragraph:
**A single paragraph that: Details at least 2 LLM Errors or missing QC steps identified during data discovery
and prompt engineering; Compares the responses of each model, highlighting observed differences; Quantitative LLM vs. LLM evaluation (Time to complete QC (wall clock timing from entering prompt to result), Inter-model agreement rate – differences between LLMs on which values within the data array changed, resulting values data was changed to, any hallucinations (bad data, new data)? Output tokens – which LLM used fewer output tokens, or is there a balance between number of tokens & accuracy to consider?**


### @cameronbracken
Both plans are light on implementaion details and need to be fleshed out. We were unable to find the exact token usage of GPT but Cline was good at finding general info but was biased towards my own field (water resources). 


### @stefan-rose
Claude misses field-level data provenance and quality flags during ingestion and relies on a single validation source, while GPT-4o specifies scripted ingestion with timestamps, raw snapshot preservation, per-field metadata, and a broader validation set and metric suite, reducing lineage gaps and validation bias. Claude’s run shows iterative refinement (20:27 wall time, 11.5k output tokens, heavy cache reuse, $1.19), whereas GPT’s runtime and token usage are NEED INFO TO FILL. Structurally, Claude is more exploratory, and GPT is a tighter, phase-gated plan with explicit deliverables and checks; GPT also briefly misrepresented its ability to read/write files in planning-only mode, creating a transparency risk about what was actually occurring.

### @MindyDeLong-Weetch



### @mdsturtevant-pnnl

### @amanda-lawter

The two LLMs, Claude and GPT, developed similar plans that employed comparable strategies. Both incorporated relevant and useful datasets not mentioned in the initial prompt, such as the National Inventory of Dams and the American Rivers Dam Removal Database. The latter dataset was used by both LLMs for test cases to determine whether dams that have already been removed would receive a high removal risk score under this project. Both plans also included processes to flag any critical missing data that could significantly impact the final outcomes for each dam. Additionally, both proposed starting with equal weighting for each category while keeping the weighting configurable. However, despite these broad similarities, the final products envisioned by the two LLMs diverged notably. GPT emphasized data quality and provenance, suggesting a deliverable in the form of a detailed report along with instructions for rerunning and reconfiguring the process to ensure reproducibility. Claude, on the other hand, prioritized usability by proposing a more interactive final product, including tiered dam mapping and an interactive dashboard to display rankings and other information for each dam.

### @JanaSimo 
