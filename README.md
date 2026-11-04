# Customer churn: model comparison

Compare a majority baseline, balanced logistic regression, and balanced random forest on the same customer data. The original exploratory work is in `main.ipynb`; the reproducible workflow is `train.py`.

## Run

Use Python 3.11 or newer. Create a virtual environment, install `requirements.txt`, then run `python train.py` from the repository root.

## What happens

1. Reserve a stratified 20% holdout with seed 42.
2. Compare models using identical five-fold stratified splits of the training portion.
3. Fit imputation, scaling, and one-hot encoding inside each fold. Unknown categories are accepted.
4. Select the model by mean validation average precision (AP), which evaluates ranking across precision/recall tradeoffs. Report variability across folds.
5. Fit the selected pipeline on all training rows and evaluate the holdout. Save metrics and the fitted pipeline in `artifacts/`.

Logistic regression learns a weighted score and transforms it into a churn probability estimate. Its numerical features are standardized. The random forest retains the original depth, leaf-size and class-weight settings, with a fixed seed. These are initial configurations, not an exhaustive tuning exercise. Classification uses the default probability threshold of 0.5; no business cost optimum is claimed. Class balancing can affect probability calibration, so treat scores as estimates requiring further calibration checks.

## Feature availability and limitations

RowNumber, CustomerId and Surname are excluded. Complain and Satisfaction Score remain excluded as in the original notebook; their measurement timing and relationship to churn require a data-source audit. All other features also need to be confirmed as available before the intended prediction date. No causal retention or savings claims are made.

The holdout split matches the original notebook and has already been inspected during development, so these results are an internal benchmark, not a pristine final evaluation. A stronger final claim requires fresh unseen data, preferably with timestamps and a defined prediction horizon. Random splitting cannot establish future performance under drift.

A majority predictor can achieve high accuracy on imbalanced data while catching no churners. Review churn precision, recall, F1, AP, ROC-AUC and the confusion matrix together. Select models and tune settings on training/validation data rather than repeatedly optimizing the holdout.

## Outputs

- `artifacts/cv_comparison.csv`: training cross-validation comparison.
- `artifacts/evaluation.json`: selected model's holdout metrics and feature list.
- `artifacts/model.joblib`: complete fitted preprocessing/model pipeline (generated locally, not committed).

Load joblib files only from trusted sources. For prediction, supply the raw feature columns listed in evaluation.json in a pandas DataFrame and call the pipeline's predict or predict_proba method.
