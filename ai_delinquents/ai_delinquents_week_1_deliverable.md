# AI Delinquents Week 1

## NOTE:
After attending the Week 1 office hours, we realize we took on a much bigger question than was intended in this challenge. Moving forward, we've narrowed down our problem statement to predicting Washington state water supply based on snow and stream flow data. The goal would be to be able to understand current water supply and predict future water supply. However, we chose not to redo last week's work; the below problem and information is based on our initial plan. 

## Problem Statement:
Multiple factors can influence the need to retain or remove current dams in Washington state. Using streamflow data and public resources, it is essential to evaluate which dams are likely to be removed due to social, political, engineering, and environmental factors. This system will assess risks, identify at-risk dams, and highlight gaps in data needed for reliable risk profiles, aiding decision-makers in mitigating risks and ensuring safety.

## Goal: 
Using streamflow data and other public resources, come up with a risk assessment for dams that may be at high risk of removal due to social, political, engineering, and/or environmental factors. Build a system that can assess the risk, and as well as identify missing data that is needed to compute a reasonable risk profile.

## Data source(s):
| Category                         | Link(s)                                                                                                                                               |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| USGS/WA state water data         | [USGS](https://waterdata.usgs.gov/state/Washington/)                                                                                                  |
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

At this time, we are unable to report the same detailed AI usage metrics from the GitHub Copilot extension in VS Code that are available in tools such as Cline. They were likely similar to that used by Claude. 

## AI chat history

### Chat history link 1 (Claude):
[Claude Opus/Sonnet 4.6](idea1_claude_chat_history_2026-03-20.md)

### Chat history link 2 (GPT):
[GPT-5.3-Codex](idea1_gpt_chat_history_2026-03-23.md)


## Data Discovery Paragraph:

The datasets identified through our LLM-assisted discovery process span streamflow and water data (USGS, USACE), energy generation and transmission (EIA Forms 860/923/930, BPA), regulatory documents (FERC licensing), social and economic indicators (Census ACS, Washington state data), and environmental impact metrics. The datasets surfaced by Claude were informed by team expertise and biased towards quantitative water resources data, while GPT identified a more general set of datasets; both were light on implementation details for each method and dataset. Both LLMs also suggested adding the National Inventory of Dams and the American Rivers Dam Removal Database to ensure baseline dam and removal data is available. Key limitations include validation bias from the American Rivers Dam Removal Database, which overrepresents small, non-hydropower removals in the eastern U.S., incomplete and potentially inconsistent regulatory features from semi-manual FERC document extraction, and missing Canadian coverage that leaves cross-border gaps for this transboundary system. Temporal misalignment between long-run disturbance metrics and more recent energy and Census data may distort risk estimates for older infrastructure. Social impact data is particularly limited, as much of it comes from self-reporting or surveys with reporting bias and missing data, is collected by multiple agencies with different goals and collection frequencies (daily to as-needed), and is often aggregated at the state or federal level, potentially obscuring local-level information important for social and cultural indicators. More broadly, the datasets have incomplete coverage due to geographic boundaries that exist on maps but not in natural systems, and intentionally adding more datasets could improve the analysis.

## LLM Reflection Paragraph:

The two LLMs, Claude and GPT, developed broadly similar plans that employed comparable strategies--both incorporated datasets not mentioned in the initial prompt (e.g., the National Inventory of Dams and the American Rivers Dam Removal Database), both proposed using the latter as test cases to validate whether already-removed dams would receive high risk scores, both included processes to flag critical missing data, and both proposed starting with equal category weighting while keeping it configurable. However, several errors and missing QC steps were identified. Claude missed field-level data provenance and quality flags during ingestion and relied on a single validation source, while GPT specified scripted ingestion with timestamps, raw snapshot preservation, per-field metadata, and a broader validation suite, reducing lineage gaps and validation bias. GPT also briefly misrepresented its ability to read and write files in planning-only mode, creating a transparency risk about what was actually occurring. Both plans were light on implementation details and need to be fleshed out. Structurally, Claude was more exploratory and biased towards the team’s domain expertise in water resources, while GPT produced a tighter, phase-gated plan with explicit deliverables and checks. The final products envisioned by the two LLMs diverged notably: GPT emphasized data quality and provenance with a detailed report and reproducibility instructions, while Claude prioritized usability with tiered dam mapping and an interactive dashboard. Quantitatively, Claude’s run took 20 minutes 27 seconds of wall time with 11.5k output tokens, heavy cache reuse, and cost $1.19; GPT’s runtime and token usage were comparable but exact metrics were unavailable due to limitations in the GitHub Copilot extension’s reporting.
