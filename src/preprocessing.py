
# ============================================================
# Automated Loan Approval & Eligibility System
# File: src/preprocessing.py
# Purpose: Prepare data for Random Forest classification
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Loan Eligibility Prediction.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "Loan_Status"
ID_COLUMN = "Customer_ID"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ------------------------------------------------------------
# 2. LOAD AND VALIDATE DATASET
# ------------------------------------------------------------

def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = [
        "Customer_ID",
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "Loan_Status",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    if df[TARGET_COLUMN].isnull().any():
        raise ValueError(
            "The target column contains missing values."
        )

    if not df[TARGET_COLUMN].isin(["Y", "N"]).all():
        raise ValueError(
            "Unexpected target values found. "
            "Expected only 'Y' and 'N'."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


# ------------------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

def prepare_features(df):
    # Remove the unique identifier from model inputs.
    X = df.drop(
        columns=[ID_COLUMN, TARGET_COLUMN]
    ).copy()

    # Encode the target: Y = 1, N = 0.
    y = df[TARGET_COLUMN].map({
        "N": 0,
        "Y": 1
    }).astype("int64")

    # Treat category-like integer columns as categorical.
    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Credit_History",
        "Property_Area",
    ]

    numerical_features = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Amount_Term",
    ]

    return (
        X,
        y,
        categorical_features,
        numerical_features,
    )


# ------------------------------------------------------------
# 4. BUILD PREPROCESSING PIPELINE
# ------------------------------------------------------------

def create_preprocessor(
    categorical_features,
    numerical_features
):
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# ------------------------------------------------------------
# 5. SPLIT DATA BEFORE FITTING PREPROCESSING
# ------------------------------------------------------------

def preprocess_dataset():
    print("=" * 65)
    print("LOAN ELIGIBILITY - DATA PREPROCESSING")
    print("=" * 65)

    df = load_dataset()

    print("\nDataset loaded successfully.")
    print("Original dataset shape:", df.shape)

    X, y, categorical_features, numerical_features = (
        prepare_features(df)
    )

    print("\nFeatures and target separated.")
    print("Customer_ID excluded from model inputs.")
    print("Number of input features before encoding:", X.shape[1])

    print("\nTarget distribution:")
    print(
        y.map({0: "N", 1: "Y"})
        .value_counts()
    )

    # Split before fitting any imputers or encoders.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain-test split completed.")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    # Fit the preprocessing pipeline only on training data.
    preprocessor = create_preprocessor(
        categorical_features,
        numerical_features
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)

    # Retrieve encoded feature names.
    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_array,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_array,
        columns=feature_names,
        index=X_test.index
    )

    # Restore a simple index for saved datasets.
    X_train_processed = X_train_processed.reset_index(
        drop=True
    )

    X_test_processed = X_test_processed.reset_index(
        drop=True
    )

    y_train_processed = y_train.reset_index(drop=True)
    y_test_processed = y_test.reset_index(drop=True)

    # Validate the processed data.
    if X_train_processed.isnull().any().any():
        raise ValueError(
            "Missing values remain in training features."
        )

    if X_test_processed.isnull().any().any():
        raise ValueError(
            "Missing values remain in testing features."
        )

    if list(X_train_processed.columns) != list(
        X_test_processed.columns
    ):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    # Create output directories if needed.
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed features.
    X_train_processed.to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    X_test_processed.to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    # Save target labels separately.
    y_train_processed.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False,
        header=["Loan_Status"]
    )

    y_test_processed.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False,
        header=["Loan_Status"]
    )

    # Save the fitted transformer for future predictions.
    joblib.dump(
        preprocessor,
        MODELS_DIR / "preprocessor.joblib"
    )

    # Save the exact feature order for model integration.
    joblib.dump(
        list(feature_names),
        MODELS_DIR / "feature_names.joblib"
    )

    print("\n" + "=" * 65)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nProcessed training features:", X_train_processed.shape)
    print("Processed testing features:", X_test_processed.shape)

    print("Missing values in training data:",
          X_train_processed.isnull().sum().sum())

    print("Missing values in testing data:",
          X_test_processed.isnull().sum().sum())

    print("\nFiles saved:")
    print(PROCESSED_DIR / "X_train.csv")
    print(PROCESSED_DIR / "X_test.csv")
    print(PROCESSED_DIR / "y_train.csv")
    print(PROCESSED_DIR / "y_test.csv")
    print(MODELS_DIR / "preprocessor.joblib")
    print(MODELS_DIR / "feature_names.joblib")

    print("\nTarget encoding:")
    print("Approved (Y) = 1")
    print("Not approved (N) = 0")


# ------------------------------------------------------------
# 6. RUN SCRIPT
# ------------------------------------------------------------

if __name__ == "__main__":
    preprocess_dataset()
