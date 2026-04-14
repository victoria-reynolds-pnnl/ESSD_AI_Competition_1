# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AI Delinquents** team repository for the ESSD AI Competition. The project goal is to build a machine learning model for **water supply forecasting in the Columbia River Basin** using streamflow, climate indices, and snow data.

Key focus: The Dalles as the primary forecast location, using **naturalized** (not regulated) streamflow data.

## Team & Branch Strategy

- Team branch: `ai_delinquents_week_2` (week-based branches off `main`)
- 5 team members with varied expertise (hydrology, GIS, software engineering, geochemistry, subsurface science)

## Competition Deliverables (Week 2, due 4/2)

- Cleaned dataset with at least 3 engineered features
- ML algorithm selection with justification
- Data dictionary, cleaning scripts, requirements.txt
- README with model selection rationale and data preparation description

## Data Sources

- **Streamflow**: USGS Water Data (waterdata.usgs.gov)
- **Climate indices**: NOAA PSL teleconnection indices
- **Snow**: NRCS Snow Survey and Water Supply Forecasting
- **Natural flow**: BPA historical streamflow, UW digital library datasets
- **Validation**: NRCS, NWRFC, and BPA forecasts for comparison

## Key Files

- `idea_water_supply_forecast.md` — project concept, data sources, and considerations
- `ai_delinquents_week_2_deliverable.md` — week 2 requirements checklist
- `week_2_resources.md` — learning resources for the team
- `initial_prompt.md` — prompt used for initial planning
