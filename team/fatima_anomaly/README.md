# Anomaly Detection + XAI

**Owner:** Fatima  
**Module:** Anomaly Detection + Explainable AI

## Objective

Detect anomalous EV charging sessions using the Jiaxing EV Charging dataset and provide interpretable predictions to support investigation.

## Current Implementation

- **Feature engineering:** Implemented in `src/features.py`, including timestamp-derived features and leakage-column exclusions.
- **Model training:** CatBoost training and evaluation pipeline implemented in `src/train.py`.
- **Inference:** `predict_anomaly()` supports single-record and batch predictions in `src/inference.py`.
- **Explainability:** Per-record SHAP explanations and global SHAP feature-importance export implemented in `src/explain.py`.
- **Tests:** Feature engineering and inference contract tests are configured through GitHub Actions.

## Training and Artifacts

Training requires the dataset path to be provided through the `CHARGING_DATA_PATH` environment variable.

The training pipeline is configured to save the model, model configuration, evaluation metrics, and feature importance artifacts locally.

Generated model and report files must be verified after running training. Do not assume that model artifacts or evaluation results exist until they have been produced.

## Current Limitations and Pending Work

- Run the training pipeline on the actual dataset and validate its outputs.
- Verify the selected decision threshold against the validation results and confirm final test metrics.
- Compare CatBoost against a baseline, XGBoost, and an unsupervised anomaly-detection model.
- Add Optuna hyperparameter optimization.
- Integrate MLflow experiment tracking and record the resulting tracking URI.
- Validate SHAP explanations and global feature importance using a trained model and representative data.
- Complete the final handoff documentation and verify the input/output schemas.

## Inference Output

The `predict_anomaly()` function returns:

- `charger_id`
- `anomaly`
- `score`
- `reason`
- `model_version`

**Note:** Passing automated tests confirms the tested code paths; it does not by itself establish model performance on real charging data.
