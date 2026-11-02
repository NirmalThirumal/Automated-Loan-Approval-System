
# ============================================================
# Automated Loan Approval & Eligibility System
# File: src/explainability.py
# Purpose: Explain Random Forest predictions using SHAP
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
X_TEST_PATH = DATA_DIR / "X_test.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD MODEL AND DATA
# ------------------------------------------------------------

def load_artifacts():
    required_files = [
        MODEL_PATH,
        FEATURE_NAMES_PATH,
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    model = joblib.load(MODEL_PATH)

    feature_names = joblib.load(FEATURE_NAMES_PATH)

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

    # Ensure the data matches the model's expected inputs.
    if list(X_train.columns) != list(feature_names):
        raise ValueError(
            "Training columns do not match saved feature names."
        )

    if list(X_test.columns) != list(feature_names):
        raise ValueError(
            "Testing columns do not match saved feature names."
        )

    if X_test.empty:
        raise ValueError("The test dataset is empty.")

    if X_test.isnull().any().any():
        raise ValueError(
            "Missing values found in the test dataset."
        )

    return model, X_train, X_test, y_test, feature_names


# ------------------------------------------------------------
# 3. INITIALIZE SHAP EXPLAINER
# ------------------------------------------------------------

def create_shap_explainer(model, X_train):
    # Use a reproducible subset of the training data
    # as the SHAP background dataset.
    background = X_train.sample(
        n=min(100, len(X_train)),
        random_state=42
    )

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability"
    )

    return explainer


# ------------------------------------------------------------
# 4. EXTRACT SHAP VALUES FOR APPROVED CLASS
# ------------------------------------------------------------

def get_approved_class_shap_values(
    explainer,
    X_test,
    model
):
    # Explain only a manageable subset of test records.
    # The first 100 test rows are used for global plots.
    X_explain = X_test.iloc[
        :min(100, len(X_test))
    ].copy()

    shap_result = explainer(X_explain)

    values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values)

    # Some SHAP versions represent binary classification as
    # (samples, features, classes).
    if values.ndim == 3:
        approved_index = list(model.classes_).index(1)
        approved_values = values[:, :, approved_index]

    # Other versions may return a two-dimensional array.
    elif values.ndim == 2:
        approved_values = values

    else:
        raise ValueError(
            f"Unexpected SHAP value shape: {values.shape}"
        )

    # Select the baseline for the approved class.
    if base_values.ndim == 2:
        approved_base_values = base_values[
            :, list(model.classes_).index(1)
        ]
    elif base_values.ndim == 1 and len(base_values) > 1:
        approved_base_values = base_values
    elif base_values.ndim == 1:
        approved_base_values = base_values
    else:
        approved_base_values = base_values

    return (
        shap_result,
        X_explain,
        approved_values,
        approved_base_values
    )


# ------------------------------------------------------------
# 5. GLOBAL SHAP SUMMARY PLOT
# ------------------------------------------------------------

def save_summary_plot(
    shap_values,
    X_explain
):
    plt.figure()

    shap.summary_plot(
        shap_values,
        X_explain,
        show=False,
        max_display=15
    )

    plt.title(
        "SHAP Summary - Loan Approval Model",
        pad=15
    )

    plt.tight_layout()

    output_path = REPORTS_DIR / "shap_summary.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved SHAP summary plot:", output_path)


# ------------------------------------------------------------
# 6. GLOBAL SHAP BAR PLOT
# ------------------------------------------------------------

def save_global_importance(
    shap_values,
    X_explain
):
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_explain.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False
    )

    csv_path = REPORTS_DIR / "shap_feature_importance.csv"

    importance_df.to_csv(
        csv_path,
        index=False
    )

    print("\nTop 10 features by mean absolute SHAP value:")
    print(importance_df.head(10).to_string(index=False))

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="bar",
        show=False,
        max_display=15
    )

    plt.title(
        "Global SHAP Feature Importance",
        pad=15
    )

    plt.tight_layout()

    plot_path = REPORTS_DIR / "shap_importance.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved SHAP importance chart:", plot_path)
    print("Saved SHAP importance data:", csv_path)


# ------------------------------------------------------------
# 7. EXPLAIN ONE TEST APPLICANT
# ------------------------------------------------------------

def explain_one_applicant(
    model,
    explainer,
    X_test,
    y_test,
    feature_names
):
    # Select the first test applicant for a reproducible example.
    applicant = X_test.iloc[[0]].copy()

    prediction = int(model.predict(applicant)[0])

    approved_probability = float(
        model.predict_proba(applicant)[0][
            list(model.classes_).index(1)
        ]
    )

    actual_status = int(y_test.iloc[0])

    print("\n" + "=" * 65)
    print("INDIVIDUAL APPLICANT EXPLANATION")
    print("=" * 65)

    print("Test applicant index:", X_test.index[0])

    print(
        "Actual status:",
        "Approved" if actual_status == 1 else "Not Approved"
    )

    print(
        "Predicted status:",
        "Approved" if prediction == 1 else "Not Approved"
    )

    print(
        f"Predicted approval probability: "
        f"{approved_probability:.4f}"
    )

    print(
        "\nNote: This probability is the model's estimate, "
        "not a guarantee of the actual lending outcome."
    )

    # Calculate SHAP values for this individual.
    single_explanation = explainer(applicant)

    single_values = np.asarray(single_explanation.values)
    single_base = np.asarray(single_explanation.base_values)

    if single_values.ndim == 3:
        approved_index = list(model.classes_).index(1)
        values = single_values[0, :, approved_index]

    elif single_values.ndim == 2:
        values = single_values[0]

    else:
        raise ValueError(
            f"Unexpected individual SHAP shape: "
            f"{single_values.shape}"
        )

    if single_base.ndim == 2:
        base_value = float(
            single_base[0, list(model.classes_).index(1)]
        )

    elif single_base.ndim == 1 and len(single_base) > 1:
        base_value = float(
            single_base[list(model.classes_).index(1)]
        )

    else:
        base_value = float(single_base.reshape(-1)[0])

    # Rank features by the magnitude of their contribution.
    contributions = pd.DataFrame({
        "Feature": feature_names,
        "Applicant_Value": applicant.iloc[0].values,
        "SHAP_Value": values
    })

    contributions["Absolute_SHAP"] = (
        contributions["SHAP_Value"].abs()
    )

    contributions = contributions.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    print("\nTop 10 feature contributions for this applicant:")
    print(
        contributions.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    contributions.to_csv(
        REPORTS_DIR / "individual_shap_explanation.csv",
        index=False
    )

    # Create a waterfall plot for the approved-class output.
    waterfall_explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=applicant.iloc[0].values,
        feature_names=feature_names
    )

    plt.figure()

    shap.plots.waterfall(
        waterfall_explanation,
        max_display=15,
        show=False
    )

    plt.title(
        "SHAP Explanation - Test Applicant",
        pad=15
    )

    plt.tight_layout()

    plot_path = REPORTS_DIR / "shap_individual_explanation.png"

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved individual explanation data:",
          REPORTS_DIR / "individual_shap_explanation.csv")

    print("Saved individual explanation plot:", plot_path)


# ------------------------------------------------------------
# 8. MAIN EXECUTION
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("SHAP MODEL EXPLAINABILITY")
    print("=" * 65)

    model, X_train, X_test, y_test, feature_names = (
        load_artifacts()
    )

    print("\nModel loaded successfully.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Features:", len(feature_names))

    if 1 not in list(model.classes_):
        raise ValueError(
            "Approved class (1) is not present in the model."
        )

    print("\nInitializing SHAP explainer...")

    explainer = create_shap_explainer(
        model,
        X_train
    )

    print("SHAP explainer initialized.")

    (
        _,
        X_explain,
        approved_shap_values,
        _
    ) = get_approved_class_shap_values(
        explainer,
        X_test,
        model
    )

    print(
        "\nSHAP values calculated for",
        len(X_explain),
        "test applicants."
    )

    save_summary_plot(
        approved_shap_values,
        X_explain
    )

    save_global_importance(
        approved_shap_values,
        X_explain
    )

    explain_one_applicant(
        model,
        explainer,
        X_test,
        y_test,
        feature_names
    )

    print("\n" + "=" * 65)
    print("SHAP EXPLAINABILITY COMPLETED")
    print("=" * 65)

    print("\nGenerated files:")
    print(REPORTS_DIR / "shap_summary.png")
    print(REPORTS_DIR / "shap_importance.png")
    print(REPORTS_DIR / "shap_feature_importance.csv")
    print(REPORTS_DIR / "shap_individual_explanation.png")
    print(REPORTS_DIR / "individual_shap_explanation.csv")


if __name__ == "__main__":
    main()
