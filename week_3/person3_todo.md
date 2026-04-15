# Person 3 To-Do — HITL Review, README Completion, Reproducibility, and Submission Packaging

## Main Goal
Finish the README, conduct the required HITL review with a non-model-builder reviewer, verify all required files exist, and make sure the Week 3 submission is complete and easy to run .

---

## Deliverables You Own
- Final `README.md` 
- HITL review writeup with 3 concrete answers 
- `requirements.txt` (version-pinned) 
- final submission completeness check 
- AI-use disclosure in README 

---

## Step 1: Use the existing README draft
Start from the current README draft that already includes:
- Train/Test Split Strategy
- Model Performance Summary
- Failures and Limitations
- Reproducibility Steps

Your job is to:
- remove placeholders
- add the final HITL section
- verify wording matches the files and outputs 

---

## Step 2: Conduct the HITL review
This review must be done by a **domain expert or teammate who is NOT the model builder** .

Pick 10–15 examples from model outputs, including:
- correct high-risk predictions
- incorrect high-risk predictions
- false negatives
- false positives
- examples from different `nerc_id` regions if possible

Show the reviewer:
- a few predictions
- confusion matrix or error examples
- SHAP or feature-importance outputs if available

Ask the reviewer to answer these 3 required questions :
1. Does the model behavior make sense given domain expectations?
2. Are there any concerning spurious correlations?
3. Are there any high-risk failure modes?

---

## Step 3: Write the HITL Review section in README
Use the reviewer’s actual observations.

The Week 3 grading expects:
- concrete observations
- sanity checks
- not generic statements 

So write specific comments such as:
- whether duration / temperature / trend drivers seem reasonable
- whether region variables may act as proxies
- whether false negatives are a serious concern

---

## Step 4: Verify metric justification is present
Make sure the README includes a couple of sentences explaining why:
- F1 is appropriate for balancing precision and recall in classification
- AUC-ROC is appropriate for measuring separation across thresholds 

If Person 2 gives you calibration notes, include a short sentence on Brier score / calibration if applicable .

---

## Step 5: Confirm all required files exist
Check that the following are present and correctly named :

### Data/
- `week2_clean.csv`
- `splits_indexed.csv`

### Models/
- `model_1.pkl`
- `model_2.pkl`

### Scripts/
- `train.py`
- `evaluate.py`
- `interpretability.py`

### Visualizations/
- at least 1 `.png` per model 

### Top-level
- `README.md`
- `requirements.txt`
- `ml_results.csv`

---

## Step 6: Verify reproducibility
Check that the workflow is runnable or clearly runnable .

Test:
1. `python Scripts/train.py`
2. `python Scripts/evaluate.py`
3. `python Scripts/interpretability.py`

Make sure paths work and outputs are saved where the README says they will be.

---

## Step 7: Create or finalize `requirements.txt`
Pin versions for all libraries used .

Likely packages:
- pandas
- numpy
- scikit-learn
- xgboost
- hdbscan
- shap
- matplotlib
- seaborn
- joblib

Make sure the file reflects what is actually imported in the scripts.

---

## Step 8: Add AI-use disclosure
The README must disclose AI use .

Include:
- which AI tools were used
- what they helped with
- that outputs were reviewed by the team before submission

If your course/team requires citation of AI-generated code, include that too.

---

## Step 9: Final README checklist
Make sure `README.md` contains exactly these sections :
- Train/Test Split Strategy
- Model Performance Summary
- Failures and Limitations
- HITL Review
  - Does the model behavior make sense given domain expectations?
  - Any concerning spurious correlations?
  - Any high-risk failure modes?
- Reproducibility Steps

---

## Step 10: Final submission check
Before submission, confirm:
- all files are easy to locate 
- README text matches `ml_results.csv`
- visualizations are readable
- HITL answers are concrete
- AI use is disclosed
- required filenames are correct 

---

## Reminder
The evaluation emphasizes:
- reproducibility and completeness
- correct split and leakage control
- metric clarity
- interpretability and error analysis
- substantive HITL review
- realistic failure mode awareness 