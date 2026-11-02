
# ============================================================
# Automated Loan Approval & Eligibility System
# File: src/train_model.py
# Purpose: Train and evaluate Random Forest classifier
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD PROCESSED DATA
# ------------------------------------------------------------

print("=" * 65)
print("RANDOM FOREST MODEL TRAINING")
print("=" * 65)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

# Ensure that the feature columns match.
if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError(
        "Missing values detected in processed features."
    )

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target counts:")
print(y_train.value_counts().sort_index())


# ------------------------------------------------------------
# 3. CREATE RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Model training completed.")


# ------------------------------------------------------------
# 5. PREDICT TEST DATA
# ------------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ------------------------------------------------------------
# 6. EVALUATE MODEL
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "Not Approved (N)",
            "Approved (Y)"
        ],
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion matrix:")
print(cm)

# AUC measures ranking performance across thresholds.
if y_test.nunique() == 2:
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC score: {auc:.4f}")


# ------------------------------------------------------------
# 7. SAVE CONFUSION MATRIX PLOT
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Approved (N)",
        "Approved (Y)"
    ]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 8. SAVE ROC CURVE
# ------------------------------------------------------------

if y_test.nunique() == 2:
    RocCurveDisplay.from_predictions(y_test, y_prob)

    plt.title("Random Forest - ROC Curve")
    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ------------------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 features by importance:")
print(importance.head(10).to_string(index=False))

importance.to_csv(
    REPORTS_DIR / "feature_importance.csv",
    index=False
)

top_features = importance.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ------------------------------------------------------------
# 10. SAVE TRAINED MODEL
# ------------------------------------------------------------

model_path = MODELS_DIR / "random_forest_model.joblib"

joblib.dump(model, model_path)

print("\n" + "=" * 65)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nSaved model:", model_path)
print(
    "Confusion matrix:",
    REPORTS_DIR / "confusion_matrix.png"
)
print("ROC curve:", REPORTS_DIR / "roc_curve.png")
print(
    "Feature importance:",
    REPORTS_DIR / "feature_importance.csv"
)
print(
    "Feature importance chart:",
    REPORTS_DIR / "feature_importance.png"
)
