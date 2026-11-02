
"""
Model Tuning for Automated Loan Approval & Eligibility System

Purpose:
1. Tune Random Forest hyperparameters using 5-fold cross-validation.
2. Select the best model based on ROC-AUC.
3. Evaluate the tuned model on the existing test dataset.
4. Compare baseline and tuned model performance.
5. Save the tuned model and tuning results.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load processed training and test datasets
# --------------------------------------------------

print("\nLoading processed datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

# Ensure target labels are integers: 0 = Not Approved, 1 = Approved
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# --------------------------------------------------
# 3. Load the existing baseline model
# --------------------------------------------------

baseline_path = MODEL_DIR / "random_forest_model.joblib"

if not baseline_path.exists():
    raise FileNotFoundError(
        "Baseline model not found. Run src/train_model.py first."
    )

baseline_model = joblib.load(baseline_path)

print("\nBaseline model loaded successfully.")


# --------------------------------------------------
# 4. Configure cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 5. Define hyperparameter search space
# --------------------------------------------------

param_distributions = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.8],
    "class_weight": ["balanced", "balanced_subsample"],
}


# --------------------------------------------------
# 6. Initialize RandomizedSearchCV
# --------------------------------------------------

print("\nStarting Random Forest hyperparameter tuning...")
print("This may take several minutes.")

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ),
    param_distributions=param_distributions,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    refit=True,
    return_train_score=True
)


# --------------------------------------------------
# 7. Train and tune using training data only
# --------------------------------------------------

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

print("\nTuning completed successfully.")

print("\nBest cross-validation ROC-AUC:")
print(round(random_search.best_score_, 4))

print("\nBest hyperparameters:")
for parameter, value in random_search.best_params_.items():
    print(f"{parameter}: {value}")


# --------------------------------------------------
# 8. Save cross-validation results
# --------------------------------------------------

cv_results = pd.DataFrame(random_search.cv_results_)

cv_results = cv_results.sort_values(
    by="rank_test_score"
)

cv_results.to_csv(
    REPORT_DIR / "model_tuning_results.csv",
    index=False
)

best_parameters = pd.DataFrame(
    list(random_search.best_params_.items()),
    columns=["Parameter", "Best Value"]
)

best_parameters.to_csv(
    REPORT_DIR / "best_model_parameters.csv",
    index=False
)

print("\nCross-validation results saved to reports/model_tuning_results.csv")


# --------------------------------------------------
# 9. Evaluate baseline and tuned models on test data
# --------------------------------------------------

def evaluate_model(model, model_name, X, y):
    """Calculate evaluation metrics for a binary classifier."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, predictions)
    balanced_accuracy = balanced_accuracy_score(y, predictions)
    roc_auc = roc_auc_score(y, probabilities)

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Not Approved", "Approved"],
        output_dict=True,
        zero_division=0
    )

    print(f"\n{'=' * 55}")
    print(f"{model_name}")
    print(f"{'=' * 55}")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["Not Approved", "Approved"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced Accuracy": balanced_accuracy,
        "ROC-AUC": roc_auc,
        "Not Approved Precision": report["Not Approved"]["precision"],
        "Not Approved Recall": report["Not Approved"]["recall"],
        "Not Approved F1": report["Not Approved"]["f1-score"],
        "Approved Precision": report["Approved"]["precision"],
        "Approved Recall": report["Approved"]["recall"],
        "Approved F1": report["Approved"]["f1-score"],
    }


print("\nEvaluating the baseline model...")

baseline_metrics = evaluate_model(
    baseline_model,
    "Baseline Random Forest",
    X_test,
    y_test
)

print("\nEvaluating the tuned model...")

tuned_metrics = evaluate_model(
    best_model,
    "Tuned Random Forest",
    X_test,
    y_test
)


# --------------------------------------------------
# 10. Save model comparison
# --------------------------------------------------

comparison = pd.DataFrame([
    baseline_metrics,
    tuned_metrics
])

comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nModel comparison:")
print(comparison.round(4).to_string(index=False))

print("\nSaved comparison to reports/model_comparison.csv")


# --------------------------------------------------
# 11. Save the tuned model
# --------------------------------------------------

tuned_model_path = MODEL_DIR / "tuned_random_forest_model.joblib"

joblib.dump(best_model, tuned_model_path)

print(f"\nTuned model saved to: {tuned_model_path}")


# --------------------------------------------------
# 12. Save confusion matrix for the tuned model
# --------------------------------------------------

tuned_predictions = best_model.predict(X_test)

display = ConfusionMatrixDisplay.from_predictions(
    y_test,
    tuned_predictions,
    labels=[0, 1],
    display_labels=["Not Approved", "Approved"],
    cmap="Blues",
    values_format="d"
)

display.ax_.set_title("Tuned Random Forest - Confusion Matrix")
display.figure_.tight_layout()

display.figure_.savefig(
    REPORT_DIR / "tuned_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(display.figure_)


# --------------------------------------------------
# 13. Save feature importance for the tuned model
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    REPORT_DIR / "tuned_feature_importance.csv",
    index=False
)

top_features = feature_importance.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - Tuned Random Forest")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "tuned_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 14. Final summary
# --------------------------------------------------

print("\n" + "=" * 55)
print("MODEL TUNING SUMMARY")
print("=" * 55)

print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
print(f"Baseline Test ROC-AUC: {baseline_metrics['ROC-AUC']:.4f}")
print(f"Tuned Test ROC-AUC:    {tuned_metrics['ROC-AUC']:.4f}")

print("\nGenerated files:")
print("- models/tuned_random_forest_model.joblib")
print("- reports/model_tuning_results.csv")
print("- reports/best_model_parameters.csv")
print("- reports/model_comparison.csv")
print("- reports/tuned_confusion_matrix.png")
print("- reports/tuned_feature_importance.csv")
print("- reports/tuned_feature_importance.png")

print("\nModel tuning completed.")
