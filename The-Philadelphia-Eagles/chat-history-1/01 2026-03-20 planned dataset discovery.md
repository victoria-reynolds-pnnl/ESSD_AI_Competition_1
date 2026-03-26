# Planned Dataset Discovery

**Date:** 2026-03-20  
**Model:** Codex (GPT-5-based coding agent)  
**Owner:** Team The Philadelphia Eagles  
**Purpose:** Dataset discovery for the ComExDBM benchmark, `CONUS` vs `WUS` clarification, dataset-details planning, and ML problem brainstorming for the ESSD AI Challenge  
**Related Week / Deliverable:** Week 1 problem definition and data discovery

_This is a curated transcript of this thread, not a raw export. Tool execution details and repetitive scaffolding were removed for readability._

---

**User:**  
Can you help me understand the dataset that is `data/`? Whats the difference between conus and wus?

**Codex:**  
Inspected the dataset layout and the checked-in READMEs under `data/` to explain the structure from the repo itself rather than guessing.

The main points:

- `data/` is a mounted dataset workspace, not a normal app data folder.
- The two main tracked dataset packages are `ComExDBM_CONUS` and `ComExDBM_WUS`.
- Both are versions of the same benchmark for fires, heatwaves, droughts, and meteorological variables.
- Both use the same broad folder structure: `01_Input/`, `02_Code/`, `03_Analysis/`, `04_OutputData/`.

The main difference between them:

- `CONUS` = contiguous United States, nationwide coverage.
- `WUS` = western US regional subset.

This is not just a geographic crop. They also differ in resolution, sources, and output layout:

- `CONUS` is `0.5° x 0.5°` resolution, about `55 km x 55 km` per grid cell.
- `WUS` is `1 km x 1 km` resolution.
- `CONUS` heatwave data is stored as yearly NetCDF files.
- `WUS` heatwave data is stored as monthly NetCDF files, and the heatwave labels are documented only for `May-Oct`.
- `CONUS` heatwaves are built from CPC Global Unified Temperature.
- `WUS` heatwaves are built from Daymet `1 km` temperature data.
- `CONUS` includes burned area and a broader meteorological variable set.
- `WUS` focuses on finer-scale fire occurrence / FRP and a smaller meteorological set plus vegetation variables.

One extra note: the western product is described inconsistently in surrounding materials as `WUS`, `PNW`, and `WECC`, but the checked-in `ComExDBM_WUS` files are the safest source of truth.

---

**User:**  
Now consider `01.07 ML Challenge Specifics.md`.

I really just want to focus on the traditional ML model and use LLM to help me train and do the regression/classification.

Consider this dataset and look at `/wiki/02.01 Dataset Details.md`.

I would like to update Dataset details page with the contents of the data set.

then i would like to brainstorm some classification/regression model problem statements that fit this dataset.

**Codex:**  
Reviewed:

- `wiki/01.07 ML Challenge Specifics.md`
- `wiki/02.01 Dataset Details.md`
- `wiki/02 DR problem statements.md`
- `wiki/01.01 AI Challenge Submission and Scoring.md`
- the week-by-week deliverable docs

The key takeaways from that review:

- The challenge needs a clear target, task type, and evaluation metric.
- The same task has to be attempted by both a traditional ML model and an LLM.
- For this dataset, the lower-risk path is a structured tabular prediction problem rather than vague exploration.
- `CONUS` is the safer modeling scope than `WUS` because it is coarser, easier to aggregate, and easier to turn into a fair ML-vs-LLM benchmark within 5 weeks.

I proposed updating `wiki/02.01 Dataset Details.md` into an ML-focused dataset brief and then adding a ranked shortlist of classification/regression problem statements tied directly to documented variables in the dataset.

---

**User:**  
I think we should create some pros or cons, because im not an expert.

**Codex:**  
That led to a more decision-oriented framing:

- Compare `CONUS` and `WUS` directly with pros and cons.
- Rank candidate problem statements by execution ease, not just scientific ambition.
- Prefer a shortlist over a giant idea dump.
- Make the dataset-details page explicitly useful for Week 1 and Week 2 deliverables.

The working recommendation was to optimize for execution ease and use `CONUS` as the default benchmark path, while still documenting `WUS` as a stretch option.

---

**Codex Proposed Plan:**  
Update `wiki/02.01 Dataset Details.md` into a clean, ML-focused dataset page using the checked-in `CONUS` and `WUS` READMEs as the factual basis. Include:

- dataset overview and DOI
- what the benchmark contains
- a `CONUS` vs `WUS` comparison table
- output folders and major variable groups
- candidate supervised targets already supported by the benchmark
- biases, limitations, and leakage risks
- why the dataset fits the ML-plus-LLM challenge workflow

Then add a ranked shortlist of `3-5` problem statements, each with:

- task type
- target variable
- unit of analysis
- likely input features
- why it is winnable
- why an LLM can attempt the same task

Recommended ranking:

1. `CONUS` fire occurrence classification
2. `CONUS` heatwave-day classification
3. `CONUS` compound extreme classification
4. `CONUS` burned-area or fire-intensity regression
5. `WUS` fine-scale fire-risk classification as a stretch option

Recommended default problem:

- binary `CONUS` fire occurrence classification

Reasoning:

- clear target
- strong fit for tabular ML
- easier for an LLM to attempt than raw numeric regression
- standard metrics are available
- lower workflow risk than `WUS`

---

**User:**  
PLEASE IMPLEMENT THIS PLAN:

Update `Dataset Details` and Produce a Ranked ML Problem Shortlist

**Codex:**  
Implemented the dataset-page rewrite. The updated page:

- reframed `wiki/02.01 Dataset Details.md` as an ML-focused brief
- explained what the benchmark contains
- added a `CONUS` vs `WUS` comparison
- documented major variable groups across heatwave, fire, drought, and meteorology outputs
- listed candidate supervised targets already supported by the data
- called out bias, class imbalance, and leakage risks
- added a ranked shortlist of model problem statements
- ended with a recommended default: `CONUS` fire occurrence classification

The page also clarified that where the western product is described as `WUS`, `PNW`, or `WECC`, the checked-in `ComExDBM_WUS` files should be treated as the source of truth.

---

**User:**  
can you dump these messages and chat that we have created together in `saved-chats/01 planned dataset discovery.md`

**Codex:**  
Checked the repo conventions for `saved-chats/` and confirmed:

- saved chats should be curated and commit-worthy
- each file should begin with a metadata block
- metadata should include date, model, owner, purpose, and related week or deliverable

I then outlined a plan to save this thread as a curated transcript rather than a raw dump, keeping it focused on:

- the initial dataset discovery
- the `CONUS` vs `WUS` explanation
- the discussion around updating the dataset details page
- the ML problem brainstorming and ranking
- the saved-chat request itself

---

**User:**  
PLEASE IMPLEMENT THIS PLAN:

Save This Conversation as a Curated Chat Artifact

**Codex:**  
Prepared to write this file as a curated chat artifact with:

- a metadata block
- a readable transcript
- only this thread’s discussion
- light cleanup for readability

The local git config did not provide a personal username to use for the metadata owner field, so the saved artifact used the team identity from the repo context instead of inventing a personal name.

---

**Outcome Captured In This Thread**

- Established that `CONUS` is the safer first path for the challenge.
- Identified `CONUS` fire occurrence classification as the strongest default benchmark problem.
- Reframed the dataset page around ML readiness, label availability, and leakage-aware evaluation.
- Preserved this collaboration as a curated Week 1 evidence artifact.
