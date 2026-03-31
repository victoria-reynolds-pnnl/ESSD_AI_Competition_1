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
The two LLMS were quite consistent in their recommended exploratory data analysis methods and each provided example questions to ask about the data. GPT provided bulleted lists of text while Claude provided python script for each recommended method. GPT had more thorough descriptions of the purpose of the methods and the data issues or limitations to consider. GPT also asked specific questions about the data and for follow-up tasks. Claude’s early response included some similar text but the response became just python code. Python scripts to execute the recommended exploratory data analysis methods were similar, although Claude’s was substantially longer. The reports generated upon executing the codes were similar. The dataset is very clean so few data issues were flagged. Both codes identified logical duplicates and outliers of limited relevance for our purposes. GPT identified a number of data entries with suspect hazard temperatures – it is not clear why these were flagged. Claude identified a number of data entries with low spatial coverage and long event duration, which is an interesting result. These entries were all for two specific (and related) event definitions, which might bear a closer look.
