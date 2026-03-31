# Project Title: Behind the Curtain

# Problem Statement: 
Heat waves and cold snaps are not uniquely defined—published thresholds differ by temperature metric, percentile cutoff, and minimum duration. This definitional uncertainty complicates regional risk assessment. Using a multi-definition Extreme Thermal Event Library, develop models that learn relationships between event characteristics (intensity, duration, spatial coverage) and identify robust representations of events that generalize across definitions and regions. [From Mengqi; minor edits from Kofi]

# Goal: 
Provide information on relocation to a business that is sensitive to the costs associated with extreme heat wave and/or cold snap events. For example, identify regions with a low risk of severe events during the next ten years.

# Data source: https://data.pnnl.gov/group/nodes/dataset/34393

# Team Structure:
| Name | Role | Responsibilities |
| :---: | :---: | :---: |
| James Kershaw | Software Engineer | To be determined |
| Philip Meyer | Applied Research | Data analysis |
| Mengqi Zhao | Earth Scientist | To be determined |
| Kofi Poku | Generalist | Code & Compute/*TBD* |


## Chat history link 1:
https://ai-incubator-chat.pnnl.gov/s/f62a6173-bbfa-4400-b4a1-49d2966ce2a8

## Chat history link 2:
Prompted two LLMs simultaneously in AI Incubator (above link has both results).

## Data Discovery Paragraph:
The dataset was downloaded and the dataset description was read, including the table of definitions for the cold snap and heat wave events. Data files were manually examined to confirm understanding of their structure.  Two LLMs (GPT and Claude-Sonnet) were prompted with the basic data structure and asked to provide recommendations for exploratory data analysis, including looking for missing values, duplicates, or other data quality issues. The LLMs were prompted to provide a script to complete data quality checks, which were each then run on the complete dataset. Potential data issues include correlations between event definitions, spatial resolution is limited to NERC regions (the areas of which are highly variable), and the event definitions may not be closely related to variable of more direct concern (i.e., not directly related to decisions regarding risks).

## LLM Reflection Paragraph:
