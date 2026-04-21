# ESSD AI Competition

**Project Title**: Water supply forecasts for the Columbia River Basin

**Team Name**: AI Delinquents

## 3.4.2 Deliverables

- **Train/Validation/Test split methodology**: A short paragraph describing the split strategy and data leakage prevention steps.

- **Reproducible training workflow**: This entails a runnable python script and a requirements.txt. The script should:
  - Load the cleaned dataset from week 2
  - Perform the training/validation/test split
  - Train your models
  - Tune key hyperparameters using the validation set
  - Evaluate final performance on test set

- **Data and results files**: Files of your week 2 data, the data with your train and test splits indexed properly, and the results of your models, each in csv format.

- **Model performance summary**: A short paragraph reporting metrics on validation and test sets and reflection on overall model performance. This should include a comparison of both models and justification for the metrics used to evaluate your models.

- **Visualization**: At least one data visualization that describes the model output and aids in interpretability.

- **Human-in-the-Loop Review**: A brief writeup by a team member acting as a "domain reviewer" that answers the following:
  - Does the model behavior make sense given domain expectations?
  - Any concerning spurious correlations?
  - Any high-risk failure modes?

- **Failure and limitations review**: A short paragraph detailing any observed failure conditions, edge cases, biases, and what you might do next time to address them.
