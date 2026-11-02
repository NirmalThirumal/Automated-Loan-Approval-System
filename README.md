# Automated Loan Approval & Eligibility System

A machine learning-based application that predicts loan eligibility using a **Random Forest Classifier**. The project includes data preprocessing, model training, hyperparameter tuning, classification-threshold optimization, model evaluation, and SHAP-based explainability. A Streamlit interface allows users to enter applicant information and view a prediction.

> **Disclaimer:** This project is an academic machine learning prototype. Its predictions are not financial advice and must not be used as the sole basis for real lending decisions.

## Features

- **Loan eligibility prediction:** Enter applicant details and receive a model prediction.
- **Model dashboard:** View available model evaluation results and visualizations.
- **Model explainability:** Explore feature importance and SHAP explanations.
- **Data preprocessing:** Handle missing values and encode categorical features.
- **Model development:** Train and tune a Random Forest classifier.
- **Threshold optimization:** Apply a saved classification threshold when generating predictions.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn, Random Forest |
| Model explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |

## Project Structure

```text
Automated-Loan-Approval-System/
│
├── app.py
│
├── data/
│   ├── raw/
│   │   ├── Loan Eligibility Prediction.csv
│   │   └── eda.py
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── random_forest_model.joblib
│   ├── tuned_random_forest_model.joblib
│   ├── threshold_optimized_random_forest_model.joblib
│   └── approval_threshold.joblib
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_importance.png
│   └── ... evaluation reports and plots
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── model_tuning.py
│   ├── threshold_optimization.py
│   ├── predict_loan.py
│   ├── explainability.py
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset

The project uses the `Loan Eligibility Prediction.csv` dataset.

- **Records:** 614
- **Columns:** 13
- **Target column:** `Loan_Status`
- **Target values:** `Y` (approved) and `N` (not approved)

The `Customer_ID` column is excluded from model training. The remaining applicant information is used as input features.

If you redistribute the dataset with this repository, make sure its original source and licence permit redistribution. Otherwise, provide instructions for obtaining it and leave the dataset out of the repository.


## Dataset License

This project uses the Loan Eligibility Prediction
dataset published by AVINESH Prabhakaran on Kaggle.

**Original Dataset:**  
https://www.kaggle.com/competitions/loan-eligibility-prediction/data

**Dataset License:**  
Creative Commons Attribution-NonCommercial 4.0
International (CC BY-NC 4.0).

https://creativecommons.org/licenses/by-nc/4.0/

The dataset is used for academic and non-commercial
machine learning purposes.

The original dataset and processed dataset files are
not included in this repository. Please obtain the
dataset from its original Kaggle source and comply
with the applicable licence and competition rules.

Credit goes to the original dataset creator,
AVINESH Prabhakaran.


## Machine Learning Workflow

1. **Exploratory Data Analysis:** Inspect the dataset, feature distributions, missing values, and potential outliers.
2. **Preprocessing:** Separate features and target, handle missing values, encode categorical variables, and split the data into training and testing sets.
3. **Model Training:** Train a Random Forest classifier.
4. **Model Tuning:** Search for improved Random Forest hyperparameters.
5. **Threshold Optimization:** Evaluate classification thresholds and save the selected threshold.
6. **Evaluation:** Review classification metrics, confusion matrices, ROC curves, and feature importance.
7. **Explainability:** Use SHAP to understand how input features contribute to model predictions.
8. **Application:** Load the saved model artifacts in Streamlit and predict loan eligibility from applicant inputs.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/Automated-Loan-Approval-System.git
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username.

### 2. Open the project directory

```bash
cd Automated-Loan-Approval-System
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open the local URL displayed in the terminal (usually `http://localhost:8501`).

## Required Model Files

The Streamlit application loads these files from the `models/` directory:

- `threshold_optimized_random_forest_model.joblib`
- `preprocessor.joblib`
- `approval_threshold.joblib`

Make sure these files are present before starting the application. The repository includes the trained artifacts so that the application can be run without training the model again.

## Model Evaluation and Reports

The `reports/` directory contains generated evaluation outputs, including confusion matrices, ROC curves, feature-importance visualizations, SHAP plots, and CSV result files.

Results can vary if the model is retrained with different settings, data splits, or random seeds. Refer to the saved reports in the repository for the results associated with the committed model artifacts.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The dataset may not represent current lending policies or all applicant populations.
- Model predictions can contain errors and may reflect biases in historical data.
- SHAP explanations describe model behavior; they do not establish that a prediction is fair or correct.
- This prototype does not connect to a bank, credit bureau, or live loan-processing system.

## Future Improvements

- Add automated unit and integration tests.
- Add stronger input validation and error handling.
- Evaluate fairness and performance across relevant applicant groups.
- Deploy the application using a suitable hosting platform.
- Add model monitoring and retraining workflows.

## Author

**Nirmal T**  
B.Tech, Computer Science and Engineering  
Puducherry Technological University

---

If you find this project useful, feel free to star the repository.
