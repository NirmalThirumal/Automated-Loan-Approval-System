
"""
Automated Loan Approval & Eligibility System
Streamlit Web Application

Features:
1. Loan Eligibility Prediction
2. Model Performance Dashboard
3. SHAP Explainability
4. Feature Importance Visualization
5. Prediction History
6. Project Information

Run:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Eligibility System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"


# --------------------------------------------------
# 2. Load saved model artifacts
# --------------------------------------------------

@st.cache_resource
def load_artifacts():
    """Load the trained model, preprocessor, and threshold."""

    model_path = (
        MODEL_DIR / "threshold_optimized_random_forest_model.joblib"
    )
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    threshold_path = MODEL_DIR / "approval_threshold.joblib"

    required_files = [
        model_path,
        preprocessor_path,
        threshold_path
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    threshold = float(joblib.load(threshold_path))

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Classification threshold must be between 0 and 1."
        )

    if 1 not in model.classes_:
        raise ValueError(
            "The model does not contain the approval class."
        )

    return model, preprocessor, threshold


# --------------------------------------------------
# 3. Page styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin: 0.5rem 0;
        }

        .approved-card {
            background-color: rgba(22, 163, 74, 0.15);
            border: 1px solid #16A34A;
        }

        .rejected-card {
            background-color: rgba(220, 38, 38, 0.15);
            border: 1px solid #DC2626;
        }

        .result-label {
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }

        .result-value {
            font-size: 1.7rem;
            font-weight: 700;
        }

        .disclaimer {
            font-size: 0.85rem;
            color: #9CA3AF;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. Load model and handle errors
# --------------------------------------------------

try:
    model, preprocessor, threshold = load_artifacts()

except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.info(
        "Check that the trained model, preprocessing pipeline, "
        "and saved threshold exist in the models folder."
    )
    st.stop()


# --------------------------------------------------
# 5. Sidebar navigation
# --------------------------------------------------

with st.sidebar:
    st.title("🏦 Loan System")

    st.caption(
        "Automated Loan Approval & Eligibility System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Loan Prediction",
            "Model Dashboard",
            "SHAP Explainability",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("**Model Information**")
    st.write("Algorithm: Random Forest")
    st.write(f"Classification threshold: {threshold:.0%}")

    st.caption(
        "Academic prototype — not a production lending system."
    )


# --------------------------------------------------
# 6. Application header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'Automated Loan Approval & Eligibility System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning-based loan eligibility prediction'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 7. Loan Prediction Page
# --------------------------------------------------

if page == "Loan Prediction":

    st.subheader("Applicant Information")

    st.write(
        "Enter the applicant details below and submit the form "
        "to generate a prediction."
    )

    with st.form("loan_application_form"):

        st.markdown("#### Personal Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

        with col2:
            married = st.selectbox(
                "Marital Status",
                ["Yes", "No"]
            )

        with col3:
            dependents = st.selectbox(
                "Number of Dependents",
                ["0", "1", "2", "3+"]
            )

        col4, col5 = st.columns(2)

        with col4:
            education = st.selectbox(
                "Education",
                ["Graduate", "Not Graduate"]
            )

        with col5:
            self_employed = st.selectbox(
                "Self Employed",
                ["No", "Yes"]
            )

        st.divider()
        st.markdown("#### Financial Information")

        col6, col7 = st.columns(2)

        with col6:
            applicant_income = st.number_input(
                "Applicant Income",
                min_value=0.0,
                value=5000.0,
                step=500.0
            )

        with col7:
            coapplicant_income = st.number_input(
                "Coapplicant Income",
                min_value=0.0,
                value=2000.0,
                step=500.0
            )

        col8, col9 = st.columns(2)

        with col8:
            loan_amount = st.number_input(
                "Loan Amount (dataset units)",
                min_value=0.0,
                value=150.0,
                step=10.0
            )

        with col9:
            loan_term = st.number_input(
                "Loan Amount Term (dataset units)",
                min_value=1.0,
                value=360.0,
                step=12.0
            )

        st.divider()
        st.markdown("#### Credit and Property Information")

        col10, col11 = st.columns(2)

        with col10:
            credit_history = st.selectbox(
                "Credit History",
                options=[1, 0],
                format_func=lambda value: (
                    "1 - Positive category"
                    if value == 1
                    else "0 - Negative category"
                )
            )

        with col11:
            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"]
            )

        submitted = st.form_submit_button(
            "Predict Loan Eligibility",
            type="primary",
            width="stretch"
        )

    # Process prediction
    if submitted:

        if applicant_income <= 0:
            st.error(
                "Applicant income must be greater than zero."
            )

        elif loan_amount <= 0:
            st.error(
                "Loan amount must be greater than zero."
            )

        else:
            applicant = pd.DataFrame([{
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Loan_Amount": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History": credit_history,
                "Property_Area": property_area
            }])

            try:
                processed_applicant = pd.DataFrame(
                    preprocessor.transform(applicant),
                    columns=preprocessor.get_feature_names_out(),
                    index=applicant.index
                )

                approved_column = list(model.classes_).index(1)

                probability = float(
                    model.predict_proba(
                        processed_applicant
                    )[0, approved_column]
                )

                predicted_class = int(
                    probability >= threshold
                )

                # Save prediction in the current session.
                if "prediction_history" not in st.session_state:
                    st.session_state["prediction_history"] = []

                st.session_state["prediction_history"].append({
                    "Applicant Income": applicant_income,
                    "Coapplicant Income": coapplicant_income,
                    "Loan Amount": loan_amount,
                    "Approval Probability": probability,
                    "Threshold": threshold,
                    "Prediction": (
                        "Approved"
                        if predicted_class == 1
                        else "Not Approved"
                    )
                })

                st.divider()
                st.subheader("Prediction Result")

                result_col, probability_col = st.columns(2)

                with result_col:

                    if predicted_class == 1:
                        st.markdown(
                            """
                            <div class="result-card approved-card">
                                <div class="result-label">
                                    Model Classification
                                </div>
                                <div class="result-value">
                                    APPROVED
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.success(
                            "The model classifies this application "
                            "as approved."
                        )

                    else:
                        st.markdown(
                            """
                            <div class="result-card rejected-card">
                                <div class="result-label">
                                    Model Classification
                                </div>
                                <div class="result-value">
                                    NOT APPROVED
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.warning(
                            "The model classifies this application "
                            "as not approved."
                        )

                with probability_col:
                    st.metric(
                        "Predicted Approval Probability",
                        f"{probability:.2%}"
                    )

                    st.progress(probability)

                    st.caption(
                        f"Classification threshold: {threshold:.0%}"
                    )

                st.info(
                    "This probability is an estimate from the model, "
                    "not a guarantee of repayment. The threshold "
                    "determines the predicted class."
                )

                with st.expander("Review Applicant Details"):
                    st.dataframe(
                        applicant,
                        width="stretch",
                        hide_index=True
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 8. Model Performance Dashboard
# --------------------------------------------------

elif page == "Model Dashboard":

    st.subheader("Model Performance Dashboard")

    comparison_path = REPORT_DIR / "model_comparison.csv"

    if comparison_path.exists():

        comparison = pd.read_csv(comparison_path)

        if not comparison.empty:

            baseline_rows = comparison[
                comparison["Model"] == "Baseline Random Forest"
            ]

            if not baseline_rows.empty:

                baseline = baseline_rows.iloc[0]

                metric1, metric2, metric3 = st.columns(3)

                metric1.metric(
                    "Baseline Accuracy",
                    f"{baseline['Accuracy']:.2%}"
                )

                metric2.metric(
                    "Balanced Accuracy",
                    f"{baseline['Balanced Accuracy']:.2%}"
                )

                metric3.metric(
                    "ROC-AUC",
                    f"{baseline['ROC-AUC']:.4f}"
                )

        st.markdown("#### Baseline vs. Tuned Model")

        st.dataframe(
            comparison.round(4),
            width="stretch",
            hide_index=True
        )

        available_metrics = [
            column for column in [
                "Accuracy",
                "Balanced Accuracy",
                "ROC-AUC"
            ]
            if column in comparison.columns
        ]

        if available_metrics and "Model" in comparison.columns:

            st.markdown("#### Metric Comparison")

            chart_data = comparison.set_index("Model")[
                available_metrics
            ]

            st.bar_chart(chart_data)

    else:
        st.warning(
            "Model comparison report was not found. "
            "Run src/model_tuning.py first."
        )

    st.divider()
    st.markdown("#### Feature Importance")

    importance_path = (
        REPORT_DIR / "tuned_feature_importance.csv"
    )

    if not importance_path.exists():
        importance_path = (
            REPORT_DIR / "feature_importance.csv"
        )

    if importance_path.exists():

        importance = pd.read_csv(importance_path)

        if {"Feature", "Importance"}.issubset(
            importance.columns
        ):

            importance = importance.sort_values(
                "Importance",
                ascending=False
            ).head(15)

            st.bar_chart(
                importance.set_index("Feature")["Importance"],
                horizontal=True
            )

            with st.expander("View Feature Importance Table"):
                st.dataframe(
                    importance,
                    width="stretch",
                    hide_index=True
                )

    else:
        st.info(
            "Feature importance report not found. "
            "Run src/train_model.py first."
        )

    st.divider()
    st.markdown("#### Confusion Matrix")

    confusion_path = REPORT_DIR / "confusion_matrix.png"

    if confusion_path.exists():
        st.image(
            str(confusion_path),
            caption="Baseline Random Forest Confusion Matrix",
            width="stretch"
        )

    else:
        st.info(
            "Confusion matrix image is not available."
        )


# --------------------------------------------------
# 9. SHAP Explainability
# --------------------------------------------------

elif page == "SHAP Explainability":

    st.subheader("SHAP Model Explainability")

    st.write(
        "SHAP (SHapley Additive exPlanations) helps explain how "
        "features influence the predictions made by the trained model."
    )

    st.info(
        "These explanations describe the model's behavior. "
        "They do not establish causation or guarantee that a lending "
        "decision is correct or fair."
    )

    # ----------------------------------------------
    # Global SHAP feature importance
    # ----------------------------------------------

    st.markdown("### Global Feature Importance")

    shap_importance_path = (
        REPORT_DIR / "shap_importance.png"
    )

    shap_summary_path = (
        REPORT_DIR / "shap_summary.png"
    )

    shap_csv_path = (
        REPORT_DIR / "shap_feature_importance.csv"
    )

    col1, col2 = st.columns(2)

    with col1:

        if shap_importance_path.exists():

            st.image(
                str(shap_importance_path),
                caption="Mean Absolute SHAP Values",
                width="stretch"
            )

        else:
            st.warning(
                "SHAP importance plot was not found."
            )

    with col2:

        if shap_summary_path.exists():

            st.image(
                str(shap_summary_path),
                caption="SHAP Summary Plot",
                width="stretch"
            )

        else:
            st.warning(
                "SHAP summary plot was not found."
            )

    if shap_csv_path.exists():

        st.markdown("### Most Influential Features")

        shap_importance = pd.read_csv(
            shap_csv_path
        )

        st.dataframe(
            shap_importance,
            width="stretch",
            hide_index=True
        )

    else:
        st.warning(
            "SHAP feature importance CSV was not found."
        )

    st.divider()

    # ----------------------------------------------
    # Individual SHAP explanation
    # ----------------------------------------------

    st.markdown("### Example Individual Explanation")

    individual_plot_path = (
        REPORT_DIR / "shap_individual_explanation.png"
    )

    individual_csv_path = (
        REPORT_DIR / "individual_shap_explanation.csv"
    )

    if individual_plot_path.exists():

        st.image(
            str(individual_plot_path),
            caption="SHAP Explanation for One Test Applicant",
            width="stretch"
        )

    else:
        st.warning(
            "Individual SHAP explanation plot was not found."
        )

    if individual_csv_path.exists():

        individual_explanation = pd.read_csv(
            individual_csv_path
        )

        st.markdown("#### Feature Contributions")

        st.dataframe(
            individual_explanation,
            width="stretch",
            hide_index=True
        )

    else:
        st.warning(
            "Individual SHAP explanation CSV was not found."
        )

    st.caption(
        "The individual explanation shown here belongs to an example "
        "from the earlier test analysis. It is not automatically the "
        "explanation for the applicant currently entered on the "
        "Loan Prediction page."
    )


# --------------------------------------------------
# 10. About Project
# --------------------------------------------------

elif page == "About Project":

    st.subheader("About the Project")

    st.write(
        "The Automated Loan Approval & Eligibility System is an "
        "academic machine learning project that predicts loan "
        "eligibility from applicant information."
    )

    st.markdown("#### Technologies Used")

    st.markdown(
        """
        - **Python** — programming language
        - **Pandas and NumPy** — data processing
        - **Scikit-learn** — preprocessing and Random Forest classification
        - **SHAP** — model explainability
        - **Matplotlib** — visualizations
        - **Streamlit** — web application interface
        """
    )

    st.markdown("#### Project Modules")

    st.markdown(
        """
        1. Exploratory Data Analysis
        2. Data Preprocessing
        3. Random Forest Model Training
        4. Model Evaluation
        5. SHAP Explainability
        6. Hyperparameter Tuning
        7. Classification Threshold Optimization
        8. Interactive Loan Prediction
        """
    )

    st.markdown("#### Current Classification Threshold")

    st.write(f"Threshold: {threshold:.2f}")

    st.warning(
        "This is an experimental prototype. It has not been validated "
        "for real lending decisions. Further testing, fairness "
        "assessment, probability calibration, and human review are "
        "necessary before any real-world use."
    )


# --------------------------------------------------
# 11. Current-session Prediction History
# --------------------------------------------------

st.divider()
st.subheader("Recent Predictions")

history = st.session_state.get(
    "prediction_history",
    []
)

if history:

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df.round(4),
        width="stretch",
        hide_index=True
    )

    if st.button("Clear Current Session History"):

        st.session_state["prediction_history"] = []

        st.rerun()

else:

    st.caption(
        "Predictions made during this session will appear here. "
        "History is temporary and is not saved to a database."
    )


# --------------------------------------------------
# 12. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Automated Loan Approval & Eligibility System | "
    "Academic project prototype | Predictions are experimental."
)
