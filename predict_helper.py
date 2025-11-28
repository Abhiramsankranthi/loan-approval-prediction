# predict_helper.py
import joblib
import numpy as np
import pandas as pd

PROD_PIPE = "models/prod_pipeline.joblib"

def load_prod():
    return joblib.load(PROD_PIPE)

def preprocess_input(raw: dict, prod):
    """
    raw: dict of raw inputs similar to original columns (includes person_income, loan_amnt, categorical strings)
    Returns a numpy array of shape (1, n_features) matching prod['feature_names'] ordering.
    """
    fnames = prod['feature_names']
    # Build a single-row DataFrame with expected columns
    # Start with all zeros
    row = pd.Series({c: 0 for c in fnames})
    # Fill numeric and log-transformed fields
    # if user provided person_income and loan_amnt, compute logs
    if 'person_income' in raw:
        row['person_income_log'] = np.log1p(float(raw['person_income']))
    if 'loan_amnt' in raw:
        row['loan_amnt_log'] = np.log1p(float(raw['loan_amnt']))
    # numeric fields that might be present directly:
    for num in ['loan_int_rate','person_emp_length','loan_percent_income','person_age','cb_person_cred_hist_length']:
        if num in raw:
            row[num] = float(raw[num])
    # Categorical one-hot: for each cat column, mark corresponding one-hot if present
    # e.g., if raw['person_home_ownership'] == 'RENT' then set 'person_home_ownership_RENT' = 1
    for col in fnames:
        if '_' in col:   # heuristic for one-hot columns like person_home_ownership_RENT
            # find prefix (original col) and value
            prefix = col.rsplit('_', 1)[0]
            # raw may supply prefix as a string, like raw['person_home_ownership'] = 'RENT'
            if prefix in raw:
                val = str(raw[prefix])
                if col.endswith('_' + val):
                    row[col] = 1
    # convert to numpy and scale
    arr = row.values.reshape(1, -1).astype(float)
    arr_scaled = prod['scaler'].transform(arr)
    return arr_scaled

def predict(raw_input: dict):
    prod = load_prod()
    X = preprocess_input(raw_input, prod)
    proba = prod['model'].predict_proba(X)[:,1][0]
    pred = int(proba >= prod['threshold'])
    # Also return top feature contributions roughly using feature_importances_ (approx)
    import numpy as np
    importances = getattr(prod['model'], 'feature_importances_', None)
    if importances is not None:
        idx = np.argsort(importances)[-5:][::-1]
        top = [(prod['feature_names'][i], float(importances[i])) for i in idx]
    else:
        top = []
    decision = "Approved" if pred == 0 else "Rejected"

    return  {
        "probability": float(proba),
        "prediction": pred,
        "decision": decision,
        "threshold": prod['threshold'],
        "top_features": top
    }
