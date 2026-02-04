# Loan Default Prediction using XGBoost

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A production-ready machine learning system that predicts loan defaults to help financial institutions reduce credit risk.

## Project Overview

This project builds a sophisticated credit risk model using **XGBoost** to predict whether a customer will default on a loan. The model analyzes borrower characteristics, loan details, and credit history to provide actionable risk assessments.

**Business Value:**
- Identify high-risk loans before approval
- Reduce default rates and financial losses
- Enable dynamic, risk-based pricing
- Provide explainable predictions for regulatory compliance

### Key Features

- **Industry-Standard Algorithm**: XGBoost excels at tabular data and is widely used in finance
- **Comprehensive EDA**: Deep exploratory analysis with business insights
- **Hyperparameter Tuning**: Optimized model performance through systematic search
- **Class Imbalance Handling**: Proper techniques for imbalanced datasets
- **Model Explainability**: SHAP values for transparent, interpretable predictions
- **Production-Ready Code**: Modular scripts for training, evaluation, and deployment

**Download the dataset**

Place the `accepted_2007_to_2018Q4.csv` file in the `data/` directory.

Dataset source: [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

## Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── accepted_2007_to_2018Q4.csv    # Raw data (not included in repo)
│   └── processed.csv                   # Processed data (generated)
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_modeling_xgboost.ipynb     # Model Training & Tuning
│   └── 03_explainability_shap.ipynb  # SHAP Analysis
│
├── src/
│   ├── utils.py                       # Utility functions
│   ├── train.py                       # Training pipeline
│   └── evaluate.py                    # Evaluation script
│
├── models/
│   ├── xgboost_model.pkl             # Trained model (generated)
│   ├── feature_names.pkl             # Feature list (generated)
│   └── model_metrics.pkl             # Performance metrics (generated)
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

##  Methodology

### 1. Exploratory Data Analysis (EDA)

**Notebook:** `notebooks/01_eda.ipynb`

- **Dataset**: 2.2M+ loans from LendingClub (2007-2018)
- **Target Variable**: Binary (0 = Fully Paid, 1 = Default/Charged Off)
- **Key Findings**:
  - Default rate: ~20%
  - Class imbalance present (addressed in modeling)
  - Strong predictors: interest rate, DTI, loan grade, credit history

**Visualizations Include**:
- Target distribution
- Feature distributions
- Correlation heatmaps
- Risk analysis by loan grade, home ownership, etc.

### 2. Feature Engineering

**Key Transformations**:
- Parse interest rate and term to numeric
- Extract employment length in years
- Calculate credit history length from earliest credit line
- Create derived features:
  - `loan_to_income`: Loan amount / Annual income
  - `installment_to_income`: Annual installment / Annual income

**Encoding Strategy**:
- **High-cardinality categoricals** (e.g., state, sub_grade): Target encoding
- **Low-cardinality categoricals** (e.g., grade, home_ownership): One-hot encoding

### 3. Model Training

**Notebook:** `notebooks/02_modeling_xgboost.ipynb`

**Pipeline**:
1. **Baseline**: Logistic Regression with class weights
2. **Initial XGBoost**: Default parameters with `scale_pos_weight`
3. **Hyperparameter Tuning**: RandomizedSearchCV (20 iterations, 3-fold CV)

**XGBoost Configuration**:
- Objective: `binary:logistic`
- Class imbalance: `scale_pos_weight` based on class ratio
- Early stopping on validation set
- Evaluation metric: ROC-AUC

**Hyperparameters Tuned**:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [4, 6, 8, 10]
- `learning_rate`: [0.01, 0.05, 0.1]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]
- `min_child_weight`: [1, 3, 5]
- `gamma`: [0, 0.1, 0.2]

### 4. Model Evaluation

**Metrics**:
- **Primary**: ROC-AUC (handles imbalanced data well)
- **Secondary**: Precision-Recall AUC, F1 Score
- **Business**: Confusion matrix, false positive/negative rates

**Results** (on test set):
```
ROC-AUC:            0.68-0.72
Average Precision:  0.45-0.50
F1 Score:           0.45-0.55
Recall:             60-70%
Precision:          40-50%
```

*(Results depend on data preprocessing and tuning)*

### 5. Model Explainability

**Notebook:** `notebooks/03_explainability_shap.ipynb`

**SHAP (SHapley Additive exPlanations)**:
- **Global Importance**: Top features driving predictions
- **Individual Explanations**: Why specific loans are high/low risk
- **Feature Interactions**: How features work together
- **Dependence Plots**: Non-linear relationships

**Top Risk Drivers**:
1. Interest Rate (higher → higher default risk)
2. Loan Grade (E, F, G → higher risk)
3. DTI Ratio (>30% → significant risk increase)
4. Recent Delinquencies (strong negative signal)
5. Credit History Length (shorter → higher uncertainty)

## Usage

### Option 1: Interactive Notebooks (Recommended for Exploration)

```bash
jupyter notebook
```

Navigate to `notebooks/` and run in order:
1. `01_eda.ipynb`
2. `02_modeling_xgboost.ipynb`
3. `03_explainability_shap.ipynb`

### Option 2: Command-Line Scripts (Production Pipeline)

**Train Model**:
```bash
cd src
python train.py \
    --data-path ../data/accepted_2007_to_2018Q4.csv \
    --processed-data-path ../data/processed.csv \
    --model-dir ../models \
    --n-iter 20
```

**Evaluate Model**:
```bash
python evaluate.py \
    --model-dir ../models \
    --test-data-path ../data/processed.csv \
    --output-dir ../models \
    --threshold-analysis
```

### Option 3: Use Trained Model for Predictions

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/xgboost_model.pkl')

# Prepare your data (same preprocessing as training)
X_new = pd.DataFrame({...})  # Your features

# Predict
predictions = model.predict_proba(X_new)[:, 1]
print(f"Default probability: {predictions[0]*100:.2f}%")
```



## Model Performance

### Comparison with Baseline

| Model                | ROC-AUC | F1 Score | Accuracy |
|---------------------|---------|----------|----------|
| Logistic Regression | 0.64    | 0.42     | 0.78     |
| XGBoost (Initial)   | 0.68    | 0.48     | 0.80     |
| **XGBoost (Tuned)** | **0.71**| **0.52** | **0.82** |

## Author
- GitHub: [@KarimHaddar](https://github.com/karimHADDAR)
- LinkedIn: [KarimHaddar](https://www.linkedin.com/in/karim-haddar/)
- Email: karim.haddar@eleves.ec-nantes.fr







