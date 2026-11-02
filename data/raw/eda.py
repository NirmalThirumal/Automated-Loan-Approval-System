
# ============================================================
# Automated Loan Approval & Eligibility System
# File: eda.py
# Purpose: Dataset Inspection and Exploratory Data Analysis
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

# eda.py and the CSV file are in the same folder: data/raw/
DATASET_PATH = (
    Path(__file__).resolve().parent
    / "Loan Eligibility Prediction.csv"
)

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATASET_PATH}\n"
        "Make sure the CSV file is inside the same folder as eda.py."
    )

df = pd.read_csv(DATASET_PATH)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("AUTOMATED LOAN APPROVAL SYSTEM - EDA")
print("=" * 65)

print("\nDataset loaded successfully!")
print("Dataset path:", DATASET_PATH)


# ------------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("1. BASIC DATASET INFORMATION")
print("=" * 65)

print("\nDataset shape:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all").T)


# ------------------------------------------------------------
# 3. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("2. COLUMN DATA TYPES")
print("=" * 65)

print(df.dtypes)

print("\nNumber of numerical columns:",
      df.select_dtypes(include="number").shape[1])

print("Number of categorical columns:",
      df.select_dtypes(include=["object", "category"]).shape[1])


# ------------------------------------------------------------
# 4. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("3. MISSING VALUE ANALYSIS")
print("=" * 65)

missing_counts = df.isnull().sum()

missing_summary = pd.DataFrame({
    "Missing_Count": missing_counts,
    "Missing_Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

print(missing_summary)

print("\nTotal missing values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# 5. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("4. DUPLICATE RECORD ANALYSIS")
print("=" * 65)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ------------------------------------------------------------
# 6. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("5. TARGET VARIABLE IDENTIFICATION")
print("=" * 65)

TARGET = "Loan_Status"
ID_COLUMN = "Customer_ID"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("Target variable:", TARGET)
print("Target classes:", df[TARGET].unique())

if ID_COLUMN in df.columns:
    print("Identifier column:", ID_COLUMN)
    print("Unique customer IDs:", df[ID_COLUMN].nunique())
    print(
        "Note: Customer_ID will be excluded "
        "from model features."
    )


# ------------------------------------------------------------
# 7. NUMERICAL FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("6. NUMERICAL FEATURE ANALYSIS")
print("=" * 65)

numerical_cols = [
    col for col in [
        "Dependents",
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Amount_Term",
        "Credit_History"
    ]
    if col in df.columns
]

print("\nNumerical summary:")
print(df[numerical_cols].describe().round(2))

print("\nNumber of unique values:")
print(df[numerical_cols].nunique())


# ------------------------------------------------------------
# 8. CATEGORICAL FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("7. CATEGORICAL FEATURE ANALYSIS")
print("=" * 65)

categorical_cols = [
    col for col in [
        "Gender",
        "Married",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status"
    ]
    if col in df.columns
]

for col in categorical_cols:
    print(f"\n--- {col} ---")

    print("\nValue counts:")
    print(df[col].value_counts(dropna=False))

    print("\nPercentage distribution:")
    print(
        (df[col].value_counts(normalize=True, dropna=False)
         * 100).round(2)
    )


# ------------------------------------------------------------
# 9. TARGET VARIABLE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("8. LOAN APPROVAL DISTRIBUTION")
print("=" * 65)

target_counts = df[TARGET].value_counts(dropna=False)

target_percentages = (
    df[TARGET].value_counts(normalize=True, dropna=False)
    * 100
).round(2)

print("\nLoan status counts:")
print(target_counts)

print("\nLoan status percentages:")
print(target_percentages)

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x=TARGET)

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 10. NUMERICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("9. NUMERICAL FEATURE DISTRIBUTIONS")
print("=" * 65)

if numerical_cols:
    df[numerical_cols].hist(
        figsize=(14, 10),
        bins=20,
        edgecolor="black"
    )

    plt.suptitle(
        "Numerical Feature Distributions",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 11. CATEGORICAL FEATURE VISUALIZATIONS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("10. CATEGORICAL FEATURE DISTRIBUTIONS")
print("=" * 65)

if categorical_cols:
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 5 * n_rows)
    )

    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        sns.countplot(data=df, x=col, ax=axes[i])

        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(
            axis="x",
            labelrotation=25
        )

    # Hide unused subplot areas
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 12. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("11. OUTLIER DETECTION (IQR METHOD)")
print("=" * 65)

outlier_cols = [
    col for col in [
        "Applicant_Income",
        "Coapplicant_Income",
        "Loan_Amount",
        "Loan_Amount_Term"
    ]
    if col in df.columns
]

outlier_results = []

for col in outlier_cols:
    series = df[col].dropna()

    if series.empty:
        continue

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[
        (series < lower_bound) |
        (series > upper_bound)
    ]

    outlier_results.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": len(outliers)
    })

    print(f"\nFeature: {col}")
    print(f"Q1: {q1:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {len(outliers)}")

if outlier_results:
    print("\nOutlier summary:")
    print(pd.DataFrame(outlier_results).round(2).to_string(index=False))


# ------------------------------------------------------------
# 13. BOXPLOTS FOR OUTLIER VISUALIZATION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("12. OUTLIER VISUALIZATION")
print("=" * 65)

if outlier_cols:
    n_cols = 2
    n_rows = (len(outlier_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 4 * n_rows)
    )

    axes = axes.flatten()

    for i, col in enumerate(outlier_cols):
        sns.boxplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")

    for i in range(len(outlier_cols), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 14. FEATURE RELATIONSHIPS WITH LOAN STATUS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("13. FEATURE VS LOAN STATUS")
print("=" * 65)

comparison_cols = [
    col for col in [
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
        "Property_Area"
    ]
    if col in df.columns
]

for col in comparison_cols:
    print(f"\n--- {col} vs {TARGET} ---")

    counts = pd.crosstab(
        df[col],
        df[TARGET],
        margins=False
    )

    print("\nApplicant counts:")
    print(counts)

    percentages = pd.crosstab(
        df[col],
        df[TARGET],
        normalize="index"
    ) * 100

    print("\nRow-wise percentages:")
    print(percentages.round(2))


# ------------------------------------------------------------
# 15. LOAN APPROVAL RATE BY PROPERTY AREA
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("14. APPROVAL RATE BY PROPERTY AREA")
print("=" * 65)

if "Property_Area" in df.columns:
    approval_data = df[df[TARGET].isin(["Y", "N"])].copy()

    approval_data["Approved"] = (
        approval_data[TARGET] == "Y"
    ).astype(int)

    approval_rates = (
        approval_data.groupby("Property_Area")["Approved"]
        .mean()
        .mul(100)
    )

    print(approval_rates.round(2))

    approval_rates.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Loan Approval Rate by Property Area")
    plt.xlabel("Property Area")
    plt.ylabel("Approval Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 16. NUMERICAL CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("15. NUMERICAL FEATURE CORRELATION")
print("=" * 65)

if len(numerical_cols) >= 2:
    correlation = df[numerical_cols].corr()

    print(correlation.round(2))

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        square=True
    )

    plt.title("Numerical Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 17. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"Dataset rows: {df.shape[0]}")
print(f"Dataset columns: {df.shape[1]}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Target variable: {TARGET}")

print("\nNext stage:")
print("Interpret EDA findings and decide preprocessing steps.")
print("Do not remove outliers before reviewing their meaning.")
