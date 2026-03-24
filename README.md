# Credit Risk Model - ML Hackathon Fix

## Problem Statement

Given a deliberately broken ML model and a credit risk dataset, the task was to:

- **Identify** all issues causing poor or misleading model performance
- **Fix** the pipeline using proper ML practices  
- **Improve** evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Document** methodology clearly for judges

---

## Repository Structure

```
.
├── credit_risk_fixed.py                # Full fixed pipeline (Colab-ready)
├── credit_risk_dataset.csv             # Dataset (13,266 rows × 20 columns)
├── Broken_Credit_Risk_Model.ipynb      # Original broken notebook (provided)
├── before_vs_after.png                 # Metrics comparison visualization
├── feature_importance.png              # Top features analysis
└── README.md                           # This file
```

---

## Dataset Overview

| Property | Details |
|----------|---------|
| **Rows** | 13,266 |
| **Columns** | 20 |
| **Target** | `target_flag` (0 = No Default, 1 = Default) |
| **Class Imbalance** | 96% No Default vs 4% Default |
| **Missing Values** | `annual_inc`, `employment_length`, `loan_amt`, `interest_rate`, `credit_score` |
| **Task Type** | Binary Classification |

---

## Issues Identified: 12 Critical Flaws

The original notebook contained systematic errors across three categories:

### 1. Data Leakage (Most Critical)

| Issue | Description |
|-------|-------------|
| **Post-loan outcome features** | `loan_status_final`, `repayment_flag`, `last_payment_status` encode what happens *after* a loan defaults — information unavailable at prediction time |
| **Engineered leakage features** | `payment_behavior_score`, `risk_indicator`, `default_likelihood_score` derived from leaked columns |
| **Preprocessor fit on combined data** | `fit_transform(X_combined)` leaks test set statistics (mean, std) into the scaler and imputer |
| **Correlation computed on full dataset** | Feature selection via `df.corr()` before train/test split contaminated feature selection |

### 2. Invalid Evaluation Protocol

| Issue | Description |
|-------|-------------|
| **Hyperparameter tuning on test set** | 15 model combinations evaluated directly on `X_test` (p-hacking) |
| **Threshold optimized on test set** | Best threshold searched across values using test F1 — all downstream metrics invalid |
| **No cross-validation** | Single train-test split leads to high variance and unreliable estimates |

### 3. Data Quality Problems

| Issue | Description |
|-------|-------------|
| **Noise columns** | `random_score_1`, `random_score_2`, `duplicate_feature` added zero business value |
| **Inconsistent categorical values** | `'self_emp'` vs `'Self-employed'`, `'RURAL'` vs `'Rural'` treated as separate classes |
| **Wrong column name** | Code referenced `annual_income`; actual column is `annual_inc` |

### 4. Modeling Mistakes

| Issue | Description |
|-------|-------------|
| **RandomOverSampler instead of SMOTE** | Duplicates minority rows → overfitting rather than learning generalizable patterns |
| **Feature importance misattribution** | After one-hot encoding, indexing `[:len(original_features)]` misaligned importance scores |

> **⚠️ Critical Insight:** The broken model achieved 1.0 ROC-AUC—a red flag, not success. This perfect score resulted from leakage columns that directly encoded the target variable, making predictions trivial.

---

## Solutions Implemented

### 1. Data Cleaning

- ✅ Removed all 3 leakage columns
- ✅ Removed 3 noise/duplicate columns  
- ✅ Standardized inconsistent categorical labels before encoding
- ✅ Corrected column name references

### 2. Legitimate Feature Engineering

Created 5 new features using only information available at **loan application time**:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| **loan_to_income** | `loan_amt / (annual_inc + 1)` | Measures affordability relative to income |
| **debt_to_income** | `(loan_amt × interest_rate) / (monthly_income + 1)` | Estimates monthly debt burden |
| **credit_loan_ratio** | `credit_score / (loan_amt + 1)` | Creditworthiness vs. loan amount requested |
| **young_borrower** | `age < 25 AND employment_length < 2` | Flags higher-risk demographic profile |
| **high_interest** | `interest_rate > median_rate` | Loan tier risk indicator |

### 3. Correct Preprocessing Pipeline

**Strict separation of concerns:**

```python
# ✅ Correct order
train_indices, test_indices = train_test_split(...)
X_train, X_test = X[train_indices], X[test_indices]

# ✅ Fit ONLY on training data
preprocessor.fit(X_train)

# ✅ Transform both (X_test never influences fit)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ✅ SMOTE applied ONLY to training set
X_train_resampled, y_train_resampled = SMOTE(...).fit_resample(X_train_transformed, y_train)
```

**Preprocessing steps:**
- Train/test split **first** (before any statistics computed)
- Median imputation (robust to outliers)
- StandardScaler normalization
- SMOTE applied only to training set

### 4. Model Training & Evaluation

**Hyperparameter Tuning:**
- Algorithm: `RandomizedSearchCV` (30 iterations)
- Cross-validation: 5-fold StratifiedKFold  
- Primary metric: ROC-AUC (appropriate for imbalanced classification)
- Models: XGBoost + LightGBM ensemble

**Threshold Optimization:**
- Set at 0.30 (below default 0.5)
- **Rationale:** In credit risk, missing a default (false negative) is far more costly than a false alarm. Lower threshold increases recall on minority class.

---

## Results: Before vs. After

| Metric | Broken Model | Fixed Model | Status |
|--------|--------------|-------------|--------|
| **ROC-AUC** | 1.0000 | *Real score* | Leakage → Valid |
| **F1-Score** | 1.0000 | *Real score* | Leakage → Valid |
| **Precision** | 1.0000 | *Real score* | Leakage → Valid |
| **Recall** | 1.0000 | *Real score* | Leakage → Valid |

> **Important:** The broken model's perfect scores are **evidence of leakage**, not evidence of success. The fixed model's scores represent true out-of-sample performance on unseen data.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `credit_risk_fixed.py` or paste cells manually
3. Run Cell 1 → Restart Runtime
4. Run Cells 2–11 in sequence
5. Upload `credit_risk_dataset.csv` when prompted in Cell 3

#### Cell Execution Guide

| Cell | Purpose | Estimated Time |
|------|---------|-----------------|
| 1 | Install required libraries | ~2 min (run once) |
| 2 | Import dependencies | 5 sec |
| 3 | Upload and load dataset | 10 sec |
| 4 | Run broken model → capture BEFORE metrics | 30 sec |
| 5 | Diagnostic audit (identify 12 issues) | 2 sec |
| 6 | Data cleaning | 5 sec |
| 7 | Feature engineering | 5 sec |
| 8 | Preprocessing + SMOTE | 15 sec |
| 9 | XGBoost + LightGBM + RandomizedSearchCV | ~8 min |
| 10 | Generate before/after comparison | 10 sec |
| 11 | Generate methodology report | 2 sec |

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/Deep-keni/cognitia_ai_hackathon
cd cognitia_ai_hackathon

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn

# Run fixed pipeline
python credit_risk_fixed.py
```

---

## Technology Stack

| Library | Purpose |
|---------|---------|
| **pandas / numpy** | Data manipulation and numerical operations |
| **scikit-learn** | Preprocessing, model evaluation, cross-validation |
| **XGBoost** | Gradient boosted decision trees |
| **LightGBM** | Fast gradient boosting framework |
| **imbalanced-learn** | SMOTE for class imbalance handling |
| **matplotlib / seaborn** | Data visualization |

---

## Methodology: Key Decisions Explained

### Why ROC-AUC Over Accuracy?

With 96:4 class imbalance, a naive model predicting "No Default" for every row achieves 96% accuracy—misleading. ROC-AUC measures discriminative power across all probability thresholds, making it appropriate for imbalanced binary classification.

### Why SMOTE Over RandomOverSampler?

- **RandomOverSampler:** Duplicates existing minority samples → model memorizes training data → poor generalization
- **SMOTE:** Generates synthetic samples along decision boundaries → teaches model to recognize diverse minority patterns → better generalization

### Why Threshold = 0.30?

In credit risk applications:
- **False Negative (missed default):** High cost—bank loses money
- **False Positive (false alarm):** Low cost—customer slightly inconvenienced

Lowering the decision threshold from 0.5 to 0.3 increases recall on the minority default class, catching more actual defaults at the cost of more false alarms.

### Why Fit Preprocessor on Training Data Only?

If the scaler sees test data during `fit()`:
- It incorporates test statistics (mean, std) into transformation
- This leaks future information into the training pipeline
- All evaluation metrics become artificially inflated
- Model performance on truly unseen data will be worse

**Correct approach:** Fit preprocessor on training data only, then apply same transformation to test data.

---

## Project Structure

```
Fixed Pipeline Process:
  1. Load raw data
  2. Remove leakage + noise columns
  3. Train-test split (FIRST)
  4. Clean categorical variables
  5. Engineer legitimate features
  6. Fit preprocessor on train ONLY
  7. Transform train + test
  8. Apply SMOTE to train
  9. Hyperparameter tune via 5-fold CV
  10. Evaluate on test set
  11. Generate comparison report
```

---

## Author

**Deep-keni**

- GitHub: [@Deep-keni](https://github.com/Deep-keni)
- LinkedIn: [Connect](https://linkedin.com/in/your-profile)

---

## License

This project was submitted as part of a competitive ML Hackathon.

- **Dataset & broken model code:** Provided by hackathon committee
- **Fixed pipeline & methodology:** Original work by the author

© 2026. All rights reserved.