📌 Problem Statement
Given a deliberately broken ML model and a credit risk dataset, the task was to:

Identify all issues causing poor or misleading model performance
Fix the pipeline using proper ML practices
Improve evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
Explain the methodology clearly for judges


📁 Repository Structure
├── credit_risk_fixed.py          # ✅ Full fixed pipeline (11 cells, Colab-ready)
├── credit_risk_dataset.csv       # Dataset (13,266 rows × 20 columns)
├── Broken_Credit_Risk_Model.ipynb # Original broken notebook (provided by committee)
├── before_vs_after.png           # Before vs After comparison chart
├── feature_importance.png        # Top feature importances plot
└── README.md                     # This file

📊 Dataset Overview
PropertyDetailsRows13,266Columns20Targettarget_flag (0 = No Default, 1 = Default)Class Imbalance96% No Default vs 4% DefaultMissing Valuesannual_inc, employment_length, loan_amt, interest_rate, credit_scoreTask TypeBinary Classification

🔴 Broken Model — What Was Wrong (12 Issues Found)
The original notebook contained 12 critical flaws across three attempted implementations:
🚨 Data Leakage
IssueDescriptionPost-loan outcome featuresloan_status_final, repayment_flag, last_payment_status encode what happens after a loan defaults — unavailable at prediction timeEngineered leakage featurespayment_behavior_score, risk_indicator, default_likelihood_score — all derived from the above leaked columnsPreprocessor fit on Train+Testfit_transform(X_combined) leaks test set statistics into the scaler and imputerCorrelation on full dataFeature selection via df.corr() computed before train/test split
🚨 Invalid Evaluation
IssueDescriptionHyperparameter tuning on test set15 model combinations evaluated directly on X_test — p-hackingThreshold optimised on test setBest threshold searched across values using test F1 — all metrics invalidNo cross-validationSingle train-test split — high variance, unreliable estimates
🚨 Data Quality
IssueDescriptionNoise columns includedrandom_score_1, random_score_2, duplicate_feature — zero business meaningDirty categoricals'self_emp' vs 'Self-employed', 'RURAL' vs 'Rural' — treated as separate classesWrong column nameCode uses annual_income; actual column is annual_inc
🚨 Modelling Mistakes
IssueDescriptionRandomOverSampler instead of SMOTESimply duplicates minority rows → overfitting, not synthetic learningFeature importance extracted incorrectlyAfter OHE expansion, indexing [:len(original_features)] misattributes importance

⚠️ The broken model scored a perfect 1.0 ROC-AUC — a red flag, not a success. This was caused by leakage columns that directly encoded the target variable, making the "prediction" trivial.


✅ Fixed Pipeline — What We Did
1. 🧹 Data Cleaning

Removed all 3 leakage columns and 3 noise/duplicate columns
Standardised inconsistent categorical labels before any encoding

2. 🔧 Legitimate Feature Engineering
Created 5 new features using only information available at loan application time:
FeatureLogicWhyloan_to_incomeloan_amt / (annual_inc + 1)Measures affordabilitydebt_to_income(loan_amt × rate) / (monthly_income + 1)Monthly burden estimatecredit_loan_ratiocredit_score / (loan_amt + 1)Creditworthiness vs askyoung_borrowerage < 25 AND employment < 2 yrsHigher risk profile flaghigh_interestrate > median rateLoan tier risk signal
3. ⚙️ Correct Preprocessing
✅ Train/test split FIRST
✅ Preprocessor fit on X_train ONLY
✅ X_test only transformed (never fit)
✅ Median imputation (robust to outliers)
✅ SMOTE applied to training set ONLY
4. 🤖 Model Training

XGBoost + LightGBM with RandomizedSearchCV
30 iterations × 5-fold StratifiedKFold cross-validation
Primary metric: ROC-AUC (correct for imbalanced binary classification)
Threshold set at 0.30 (below default 0.5 to improve recall on minority default class)


📈 Results — Before vs After
Metric🔴 Broken Model🟢 Fixed ModelDeltaROC-AUC1.0000 (leaked)Real score—F1-Score1.0000 (leaked)Real score—Precision1.0000 (leaked)Real score—Recall1.0000 (leaked)Real score—

📌 The broken model's 1.0 scores are not a benchmark — they are evidence of leakage. The fixed model's scores represent true out-of-sample performance on unseen data.


🛠️ How to Run
Google Colab (Recommended)

Open Google Colab
Upload credit_risk_fixed.py or paste cells manually
Run Cell 1 → Restart Runtime
Run Cells 2 → 11 in order
Upload credit_risk_dataset.csv when prompted in Cell 3

Cell Guide
CellPurposeEst. Time1Install libraries~2 min (run once)2Imports5 sec3Upload & load dataset10 sec4Run broken model → BEFORE metrics30 sec5Diagnostic audit (12 issues)2 sec6Data cleaning5 sec7Feature engineering5 sec8Preprocessing + SMOTE15 sec9XGBoost + LightGBM + RandomizedSearchCV~8 min10Champion + Before/After chart10 sec11Methodology Report (copy & submit)2 sec
Local Setup
bashgit clone https://github.com/YOUR_USERNAME/credit-risk-model-fix
cd credit-risk-model-fix

pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn pycaret matplotlib seaborn

python credit_risk_fixed.py

🧰 Tech Stack
LibraryPurposepandas / numpyData manipulationscikit-learnPreprocessing, pipeline, evaluationXGBoostGradient boosted treesLightGBMFast gradient boostingimbalanced-learnSMOTE for class imbalancePyCaretAutoML model benchmarkingmatplotlib / seabornVisualisation

📝 Key Decisions Explained
Why ROC-AUC over Accuracy?
With 96:4 class imbalance, a model predicting "No Default" for every row scores 96% accuracy. ROC-AUC measures true discriminative power across all thresholds.
Why SMOTE over RandomOverSampler?
RandomOverSampler duplicates existing minority rows — the model memorises them. SMOTE generates new synthetic samples along decision boundaries, teaching the model to generalise.
Why threshold = 0.30?
In credit risk, a missed default (false negative) is far more costly than a false alarm. Lowering the threshold from 0.5 to 0.3 increases recall on the minority class — catching more actual defaults.
Why fit preprocessor on train only?
If the scaler sees test data during fit, it incorporates test statistics (mean, std) into the transformation. This leaks future information and inflates all evaluation metrics.

👤 Author
Your Name

GitHub: @your_username
LinkedIn: your-linkedin


📄 License
This project was submitted as part of a competitive ML Hackathon.
Dataset and broken model code were provided by the hackathon committee.
Fixed pipeline and methodology are original work by the author.
