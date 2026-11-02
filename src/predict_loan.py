
"""
Loan Eligibility Prediction
Automated Loan Approval & Eligibility System

Purpose:
1. Accept applicant details from the terminal.
2. Apply the saved preprocessing pipeline.
3. Predict the probability of loan approval.
4. Classify the application using the saved threshold.
5. Display a readable prediction result.

Important:
This is an educational ML prediction, not a final lending decision.
"""

from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# --------------------------------------------------
# 2. Load saved artifacts
# --------------------------------------------------

def load_artifacts():
    """Load the trained model, preprocessor, and threshold."""

    model_path = MODEL_DIR / "threshold_optimized_random_forest_model.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    threshold_path = MODEL_DIR / "approval_threshold.joblib"

    required_files = [
        model_path,
        preprocessor_path,
        threshold_path,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}\n"
                "Run preprocessing, training, and threshold optimization first."
            )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    threshold = float(joblib.load(threshold_path))

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The saved approval threshold must be between 0 and 1.")

    return model, preprocessor, threshold


# --------------------------------------------------
# 3. Collect applicant information
# --------------------------------------------------

def get_numeric_input(prompt, allow_zero=True):
    """Read a valid non-negative numeric value from the user."""

    while True:
        value = input(prompt).strip()

        try:
            number = float(value)

            if not pd.notna(number):
                raise ValueError

            if number < 0:
                print("Please enter a non-negative number.")
                continue

            if not allow_zero and number == 0:
                print("Please enter a value greater than zero.")
                continue

            return number

        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_choice(prompt, choices):
    """Read a value from a list of allowed choices."""

    while True:
        value = input(prompt).strip()

        for choice in choices:
            if value.casefold() == choice.casefold():
                return choice

        print("Invalid choice. Options:", ", ".join(choices))


def get_applicant_details():
    """Collect applicant details using the dataset's original columns."""

    print("\nEnter Applicant Details")
    print("-" * 40)

    gender = get_choice(
        "Gender (Male/Female): ",
        ["Male", "Female"]
    )

    married = get_choice(
        "Married (Yes/No): ",
        ["Yes", "No"]
    )

    dependents = get_choice(
        "Number of Dependents (0/1/2/3+): ",
        ["0", "1", "2", "3+"]
    )

    education = get_choice(
        "Education (Graduate/Not Graduate): ",
        ["Graduate", "Not Graduate"]
    )

    self_employed = get_choice(
        "Self Employed (Yes/No): ",
        ["Yes", "No"]
    )

    applicant_income = get_numeric_input(
        "Applicant Income: ",
        allow_zero=False
    )

    coapplicant_income = get_numeric_input(
        "Coapplicant Income (enter 0 if none): "
    )

    loan_amount = get_numeric_input(
        "Loan Amount (in dataset units): ",
        allow_zero=False
    )

    loan_term = get_numeric_input(
        "Loan Amount Term (in dataset units): ",
        allow_zero=False
    )

    credit_history = get_choice(
        "Credit History (1 = meets dataset's positive category, "
        "0 = negative category): ",
        ["0", "1"]
    )

    property_area = get_choice(
        "Property Area (Urban/Semiurban/Rural): ",
        ["Urban", "Semiurban", "Rural"]
    )

    # Use the original dataset column names.
    # Customer_ID is intentionally excluded because it was not used for training.
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
        "Credit_History": int(credit_history),
        "Property_Area": property_area,
    }])

    return applicant


# --------------------------------------------------
# 4. Predict loan eligibility
# --------------------------------------------------

def predict_loan(applicant, model, preprocessor, threshold):
    """Transform applicant data and predict loan approval."""

    # The saved preprocessor expects the original input columns.
    processed_applicant = pd.DataFrame(
    preprocessor.transform(applicant),
    columns=preprocessor.get_feature_names_out(),
    index=applicant.index
    )

    # Find the probability column corresponding to class 1 (Approved).
    approved_column = list(model.classes_).index(1)

    approval_probability = model.predict_proba(
        processed_applicant
    )[0, approved_column]

    predicted_class = int(approval_probability >= threshold)

    return approval_probability, predicted_class


# --------------------------------------------------
# 5. Display prediction
# --------------------------------------------------

def display_result(applicant, probability, predicted_class, threshold):
    """Display the applicant's details and prediction."""

    print("\n" + "=" * 55)
    print("LOAN ELIGIBILITY PREDICTION")
    print("=" * 55)

    print("\nApplicant Details:")
    print(applicant.to_string(index=False))

    print(f"\nPredicted Approval Probability: {probability:.2%}")
    print(f"Classification Threshold:      {threshold:.2f}")

    if predicted_class == 1:
        print("\nPrediction: APPROVED")
        print("The model classifies this application as approved.")
    else:
        print("\nPrediction: NOT APPROVED")
        print("The model classifies this application as not approved.")

    print("\nImportant:")
    print("- This is an experimental model prediction.")
    print("- It is not a guarantee of repayment or a final lending decision.")
    print("- Real lending decisions require further validation and review.")
    print("=" * 55)


# --------------------------------------------------
# 6. Main program
# --------------------------------------------------

def main():
    try:
        model, preprocessor, threshold = load_artifacts()

        print("Loan Eligibility Prediction System")
        print(f"Using classification threshold: {threshold:.2f}")

        while True:
            applicant = get_applicant_details()

            probability, predicted_class = predict_loan(
                applicant,
                model,
                preprocessor,
                threshold
            )

            display_result(
                applicant,
                probability,
                predicted_class,
                threshold
            )

            again = get_choice(
                "\nPredict another application? (Yes/No): ",
                ["Yes", "No"]
            )

            if again == "No":
                print("\nThank you for using the prediction system.")
                break

    except FileNotFoundError as error:
        print("\nError:", error)

    except Exception as error:
        print("\nAn unexpected error occurred:", error)
        print(
            "Check that the model, preprocessor, and input columns "
            "match your current training artifacts."
        )


if __name__ == "__main__":
    main()
