# build_prod_pipeline.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# CONFIG
MODELS_DIR = Path("models")
TRAINED_MODEL_FILE = MODELS_DIR / "xgboost_tuned.joblib"   # change if your tuned model filename differs
PROD_PIPELINE_FILE = MODELS_DIR / "prod_pipeline.joblib"
THRESHOLD = 0.35   # change to your preferred threshold

# Load one of your trained models (tuned)
model = joblib.load(str(TRAINED_MODEL_FILE))

# We must save feature names and the preprocessing steps needed in production:
# (1) Which columns are required in order. We'll infer from training saved file or re-derive from train.csv
df = pd.read_csv("train.csv")
# Re-do exactly the preprocessing pipeline (no SMOTE) so prod pipeline reproduces training preprocessing
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df['person_income_log'] = np.log1p(df['person_income'])
df['loan_amnt_log'] = np.log1p(df['loan_amnt'])
df.drop(['id','person_income','loan_amnt'], axis=1, inplace=True)

feature_names = df.drop('loan_status', axis=1).columns.tolist()
print("Feature names for production pipeline:", feature_names)

# Create a StandardScaler fitted on the original training data (re-fit here)
from sklearn.preprocessing import StandardScaler
X = df[feature_names].values
scaler = StandardScaler().fit(X)

# Create a wrapper pipeline object that has scaler and model and metadata
prod = {
    "feature_names": feature_names,
    "scaler": scaler,
    "model": model,
    "threshold": THRESHOLD
}

joblib.dump(prod, PROD_PIPELINE_FILE)
print("Saved production pipeline to", PROD_PIPELINE_FILE)
