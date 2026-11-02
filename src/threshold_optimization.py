
"""
Threshold Optimization
Automated Loan Approval & Eligibility System

Purpose:
1. Generate cross-validated predictions using training data.
2. Compare different classification thresholds.
3. Save threshold metrics and comparison plots.
4. Select a threshold using balanced accuracy on training CV predictions.
5. Evaluate the selected threshold on the existing test set.

Note:
Threshold selection uses training data only.
The test set is used only for final evaluation in this script.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

print("\nLoading processed datasets...")

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns").astype(int)
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns").astype(int)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# --------------------------------------------------
# 3. Load the baseline Random Forest model
# --------------------------------------------------

model_path = MODEL_DIR / "random_forest_model.joblib"

if not model_path.exists():
    raise FileNotFoundError(
        "Baseline model not found. Run src/train_model.py first."
    )

model = joblib.load(model_path)

print("\nBaseline Random Forest model loaded.")


# --------------------------------------------------
# 4. Generate out-of-fold probabilities
# --------------------------------------------------

# Each training applicant receives a probability from a model
# that was not trained on that applicant.
# This helps reduce overfitting during threshold selection.

print("\nGenerating cross-validated training probabilities...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

# Locate the probability column corresponding to class 1 (Approved).
approved_column = list(model.classes_).index(1)
train_probabilities = oof_probabilities[:, approved_column]

print("Cross-validated probabilities generated.")


# --------------------------------------------------
# 5. Evaluate a range of thresholds
# --------------------------------------------------

thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)

results = []

for threshold in thresholds:

    # Predict Approved when probability >= threshold.
    train_predictions = (
        train_probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_train, train_predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_train, train_predictions
        ),
        "Precision Approved": precision_score(
            y_train, train_predictions, zero_division=0
        ),
        "Recall Approved": recall_score(
            y_train, train_predictions, zero_division=0
        ),
        "Recall Not Approved": recall_score(
            y_train, train_predictions, pos_label=0, zero_division=0
        ),
        "F1 Approved": f1_score(
            y_train, train_predictions, zero_division=0
        ),
    })

threshold_results = pd.DataFrame(results)

print("\nThreshold evaluation using cross-validated training predictions:")
print(threshold_results.round(4).to_string(index=False))


# --------------------------------------------------
# 6. Select a threshold using balanced accuracy
# --------------------------------------------------

# Balanced accuracy gives equal weight to recall for both classes.
# If multiple thresholds tie, choose the one closest to 0.50.

max_balanced_accuracy = threshold_results[
    "Balanced Accuracy"
].max()

best_candidates = threshold_results[
    np.isclose(
        threshold_results["Balanced Accuracy"],
        max_balanced_accuracy
    )
].copy()

best_candidates["Distance From 0.50"] = (
    best_candidates["Threshold"] - 0.50
).abs()

best_row = best_candidates.sort_values(
    by="Distance From 0.50"
).iloc[0]

best_threshold = float(best_row["Threshold"])

print("\nSelected threshold:", best_threshold)
print(
    "Cross-validated balanced accuracy:",
    round(float(best_row["Balanced Accuracy"]), 4)
)


# --------------------------------------------------
# 7. Save threshold results
# --------------------------------------------------

threshold_results.to_csv(
    REPORT_DIR / "threshold_optimization_results.csv",
    index=False
)

print("\nSaved threshold metrics to:")
print("reports/threshold_optimization_results.csv")


# --------------------------------------------------
# 8. Plot metrics across thresholds
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Accuracy"],
    marker="o",
    label="Accuracy"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Balanced Accuracy"],
    marker="s",
    label="Balanced Accuracy"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Recall Approved"],
    marker="^",
    label="Recall Approved"
)

plt.plot(
    threshold_results["Threshold"],
    threshold_results["Recall Not Approved"],
    marker="d",
    label="Recall Not Approved"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization - Cross-Validated Training Results")
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "threshold_optimization_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved plot to reports/threshold_optimization_plot.png")


# --------------------------------------------------
# 9. Fit a final model on all training data
# --------------------------------------------------

# The threshold was selected using out-of-fold training predictions.
# Now fit the baseline model on all training data.

print("\nFitting final model on all training data...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 10. Evaluate the selected threshold on the test set
# --------------------------------------------------

test_probabilities = model.predict_proba(X_test)[:, approved_column]

# Default threshold comparison
default_predictions = (
    test_probabilities >= 0.50
).astype(int)

# Selected threshold predictions
optimized_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_auc = roc_auc_score(y_test, test_probabilities)

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

print(f"\nROC-AUC (threshold independent): {test_auc:.4f}")

print("\n--- Default Threshold: 0.50 ---")

print(
    f"Accuracy: {accuracy_score(y_test, default_predictions):.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, default_predictions):.4f}"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, default_predictions, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        default_predictions,
        labels=[0, 1],
        target_names=["Not Approved", "Approved"],
        zero_division=0
    )
)

print(f"\n--- Selected Threshold: {best_threshold:.2f} ---")

print(
    f"Accuracy: {accuracy_score(y_test, optimized_predictions):.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy_score(y_test, optimized_predictions):.4f}"
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, optimized_predictions, labels=[0, 1]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        optimized_predictions,
        labels=[0, 1],
        target_names=["Not Approved", "Approved"],
        zero_division=0
    )
)


# --------------------------------------------------
# 11. Save test-set comparison
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "Threshold": 0.50,
        "Accuracy": accuracy_score(y_test, default_predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, default_predictions
        ),
        "Recall Not Approved": recall_score(
            y_test, default_predictions, pos_label=0, zero_division=0
        ),
        "Recall Approved": recall_score(
            y_test, default_predictions, pos_label=1, zero_division=0
        ),
        "ROC-AUC": test_auc,
    },
    {
        "Threshold": best_threshold,
        "Accuracy": accuracy_score(y_test, optimized_predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, optimized_predictions
        ),
        "Recall Not Approved": recall_score(
            y_test, optimized_predictions, pos_label=0, zero_division=0
        ),
        "Recall Approved": recall_score(
            y_test, optimized_predictions, pos_label=1, zero_division=0
        ),
        "ROC-AUC": test_auc,
    }
])

comparison.to_csv(
    REPORT_DIR / "threshold_test_comparison.csv",
    index=False
)

print("\nSaved test comparison to reports/threshold_test_comparison.csv")


# --------------------------------------------------
# 12. Save model and threshold separately
# --------------------------------------------------

# Keep the baseline model untouched.
# The threshold is saved separately for use by a future prediction module.

joblib.dump(
    model,
    MODEL_DIR / "threshold_optimized_random_forest_model.joblib"
)

joblib.dump(
    best_threshold,
    MODEL_DIR / "approval_threshold.joblib"
)

print("\nSaved model:")
print("models/threshold_optimized_random_forest_model.joblib")

print("\nSaved threshold:")
print("models/approval_threshold.joblib")


# --------------------------------------------------
# 13. Final summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION COMPLETED")
print("=" * 60)

print(f"Default threshold: 0.50")
print(f"Selected threshold: {best_threshold:.2f}")
print(
    "Training CV balanced accuracy:",
    f"{best_row['Balanced Accuracy']:.4f}"
)

print("\nGenerated files:")
print("- reports/threshold_optimization_results.csv")
print("- reports/threshold_optimization_plot.png")
print("- reports/threshold_test_comparison.csv")
print("- models/threshold_optimized_random_forest_model.joblib")
print("- models/approval_threshold.joblib")

print("\nReview the results before choosing a threshold for deployment.")
