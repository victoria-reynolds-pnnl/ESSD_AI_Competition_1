# Saved Chat

- Date: 2026-03-20
- Model: Codex (GPT-5-based coding agent)
- Owner: Team The Philadelphia Eagles
- Purpose: Capture the decision discussion linking the wildfire benchmark choice to a possible interactive web application direction
- Related week or deliverable: Week 1 problem framing
- Related docs:
  - [../wiki/02.01 Dataset Details.md](../wiki/02.01%20Dataset%20Details.md)
  - [../wiki/02 DR problem statements.md](../wiki/02%20DR%20problem%20statements.md)
  - [../wiki/02.02 Fire occurrence decision and app direction.md](../wiki/02.02%20Fire%20occurrence%20decision%20and%20app%20direction.md)

## Summary

This chat explored whether the AI-ready fire, heatwave, and drought benchmark could support a playful web-facing application, such as a React or `react-three-fiber` experience showing predicted wildfire risk over a stylized landscape. The conclusion was that the best competition-aligned problem statement is `CONUS` fire occurrence classification, because it is easier to scope, works for both ML and LLM comparison, and can still power a visually engaging app.

The discussion also established that `CONUS` is better as the primary benchmark path than the finer-scale `WUS` option for a 5-week challenge, while still leaving room for a future interactive demo layer.

## Cleaned Transcript

### User

Consider `wiki/02.01 Dataset Details.md`.

I have an interest in building a 3D model in something like React or `react-three-fiber`, and it would be cool to have a prediction of whether a piece of land is on fire or would have a slice of land on fire given parameters. Is there any problem statement that could get us to that end goal, where we use the model we build inside a funny, cute web app like that?

### Assistant

Reviewed the dataset details and problem-statement research, then recommended `CONUS` fire occurrence classification as the strongest fit for that app direction.

Reasoning:

- It produces a simple per-cell prediction that maps naturally to a grid, map, or terrain visualization.
- It is already the recommended default problem in the current dataset notes.
- It is a better fit for both traditional ML and an LLM baseline than raw fire-magnitude regression.
- It is more manageable than the higher-resolution `WUS` workflow in the challenge timeline.

Also noted:

- `CONUS` compound extreme classification could be a good second option if the team wants a richer visual story later.
- Fire-intensity or burned-area regression is less attractive as the primary benchmark because it is harder and less LLM-friendly.

Suggested project framing:

> We aim to use the `CONUS` portion of the ComExDBM benchmark to predict whether a grid cell will experience fire activity on a given day or week based on drought, meteorological, vegetation, and lagged fire features. This could support intuitive wildfire-risk screening and interactive communication of hazardous conditions. We will compare a traditional ML classifier with an LLM-based approach on the same structured inputs and evaluation split.

Suggested app direction:

- React app with map or stylized terrain
- Optional `react-three-fiber` 3D layer
- Controls for date and environmental parameters
- Color or animation to indicate predicted fire risk

### User

I like this recommendation. Can we record this decision?

### Assistant

Recorded the decision in the repo, then after a follow-up request moved it into a dedicated wiki document rather than embedding it into the dataset-details page.

### User

I would like a new document rather than adding into `02.01`.

### Assistant

Created a standalone decision note:

- [../wiki/02.02 Fire occurrence decision and app direction.md](../wiki/02.02%20Fire%20occurrence%20decision%20and%20app%20direction.md)

Adjusted related docs:

- Removed the embedded decision note from `02.01 Dataset Details.md`
- Added a pointer in `02 DR problem statements.md` to the new decision document

## Outcome Captured

- Working benchmark choice: `CONUS` fire occurrence classification
- Application direction: lightweight wildfire-risk web experience, potentially with React and `react-three-fiber`
- Scope boundary: present results as grid-cell risk visualization, not parcel-level prediction
