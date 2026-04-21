# FTES Dataset

## Problem Statement

This project aims to predict the temporal evolution of temperature and flow in a multi-well fracture thermal energy storage (FTES) system using data from the Sanford Underground Research Facility (SURF). By tracking the injection of hot water in this system with temperature, pressure, and water chemistry changes at each well, we can train a machine learning (ML) model to determine where and how water flows as a proxy of the fracture network. A fracture network is an interconnected system of cracks, joints, or faults within rock formations that act as a primary pathway for fluid flow and storage, particularly in low-permeability, fractured rock. By using the model as a representative fracture network, we aim to predict the temporal evolution for different injection rates, time spans, or hot water temperatures. This research will allow for a more efficient FTES injection design to extract heat from deep underground in similar, metamorphic geologic settings and be useful for the geothermal industries.

## Goal

Predict production well temperatures based on injection conditions and historical thermal response.

## Data Source

DEMO-FTES Test 1 — Subsurface thermal energy storage monitoring at Sanford Underground Research Facility

## Team Structure

| Team Member | Role | Responsibilities |
|-------------|------|------------------|
| Heather Sabella | Data Engineer | Clean and normalize data, engineer features, manage train/test splits, and manage GitHub submissions. |
| Judy Robinson | Project Lead | Define problem statement and context for dataset, manage timelines, coordinate team workflow, and ensure rubric compliance. |
| Tse-Chun Chen | ML Engineer | Select and train ML models, tune hyperparameters, evaluate performance, and write ML scripts. |
| Michelle Li | LLM Engineer | Design prompts, run LLM benchmarks, parse LLM outputs, and compare ML vs LLM performance. |
| Dana Vesty | Documentation Lead/QA | Review code logic and deliverables, create data dictionary, human-in-the-loop validation, and facilitate team meetings. |
| Sadie Montgomery | LLM Engineer/QA | LLM prompt, failure mode identification, and bias & limitations analysis. |

## Chat Histories

- **Link 1 - O4-mini**: https://ai-incubator-chat.pnnl.gov/s/4cba5ebd-f8dc-449b-90ad-9cfd52786aa8
- **Link 2 - Haiku 4.5**: https://ai-incubator-chat.pnnl.gov/s/cc7b9816-cce6-4972-897d-0dbb082b3232
- **Link 3 - GPT 5.4**: https://ai-incubator-chat.pnnl.gov/s/03eb583c-daff-45db-8780-e70142cdfba4
- **Link 4 - Sonnet 4.5**: https://ai-incubator-chat.pnnl.gov/s/5cb0f4e6-08ac-4b8c-967b-d8df9c791e82

## Data Discovery

The FTES data contains flow rate, temperature, pressure and electrical conductivity values for a multi-well fracture configuration at the Sanford Underground Research Facility (SURF) at 1-second intervals and 1-hour averages. There were two main phases of the test. Phase 1 consisted of a hot water injection in one well and extraction/production from two adjacent wells. Phase 2 (not considered in this project) consisted of an ambient water injection. The Phase 1 experiment represents one flow field and therefore the interpretation is limited to this scenario and fracture orientation. For example, the evolution of hot water might be different from an ambient injection. Other considerations will be sensor drift and making sure to interpret pressure responses with temperature to avoid misinterpretation. The data only contains one injection and two producers, so there are limitations on how broadly the model will apply without further data inputs. Without access to the raw tabular data, the data dictionary (FTES_data_description.docx) limits the EDA to structural interpretation rather than true empirical statistical analysis.

## LLM Reflection

During EDA, three major LLM issues emerged: o4 mini hallucinated quantitative values (correlations, pressures, temperatures, EC values, RMSEs) despite having no data; Claude Sonnet 4.5 produced contradictory math by assuming 1 second sampling intervals in calculation while stating 1 minute; and all models overlooked the sampling rate mismatch as a critical QC blocker, though GPT 5.4 uniquely identified it as the primary QA issue. Performance characteristics diverged sharply: Claude Haiku 4.5 was fastest (10 s) and most accurate while using the fewest total tokens (4,270), GPT 5.4 was the most verbose (2,037 completion tokens, 9,380 characters) but avoided hallucinations, o4 mini used the most completion tokens (2,402) yet produced the shortest response (2,404 characters) and all hallucinated numbers, and Claude Sonnet 4.5 was slowest (56 s) and still made mathematical errors. Overall, models agreed on structural interpretation but diverged on quantitative claims—only o4 mini produced fabricated numbers—and neither speed nor token volume corresponded to output quality in the absence of real data.
