# checkpoint2_models.py
import pandas as pd
import numpy as np
import os
import joblib
from time import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------
# 1. Load + Preprocess
# -----------------------
print("Loading data...")
df = pd.read_csv("train.csv")

# Impute medians (same as before)
df['loan_int_rate'].fillna(df['loan_int_rate'].median(), inplace=True)
df['person_emp_length'].fillna(df['person_emp_length'].median(), inplace=True)

# One-hot encode categorical cols (same as your earlier step)
df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)

# Log-transform skewed numeric features
df['person_income_log'] = np.log1p(df['person_income'])
df['loan_amnt_log'] = np.log1p(df['loan_amnt'])

# drop id and original skewed columns
df.drop(['id', 'person_income', 'loan_amnt'], axis=1, inplace=True)

# Separate X and y and get feature names
X = df.drop('loan_status', axis=1)
y = df['loan_status']
feature_names = X.columns.tolist()
print(f"Number of features: {len(feature_names)}")

# -----------------------
# 2. Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train/test split done. Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------
# 3. Scale numeric features
# -----------------------
# We'll scale all features (you previously scaled all) — okay for these models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(scaler, "models/scaler.joblib")

# --- Before SMOTE ---
print("Before SMOTE class counts:")
print(y_train.value_counts())
print("As percentages:")
print(y_train.value_counts(normalize=True) * 100)

# -----------------------
# 4. SMOTE on training set
# -----------------------
print("\nApplying SMOTE to training data to fix class imbalance...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
print("After SMOTE, counts:", pd.Series(y_train_res).value_counts().to_dict())

# -----------------------
# 5. Train baseline Logistic Regression (retrain on resampled)
# -----------------------
print("\nTraining Logistic Regression (baseline) on resampled data...")
lr = LogisticRegression(max_iter=1000, random_state=42)
t0 = time()
lr.fit(X_train_res, y_train_res)
t1 = time()
print(f"Trained Logistic Regression in {t1-t0:.2f}s")
joblib.dump(lr, "models/logistic_regression.joblib")

# Evaluate helper
def evaluate_model(name, model, X_test_arr, y_test_arr):
    y_pred = model.predict(X_test_arr)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test_arr)[:, 1]
    except Exception:
        # some models may not have predict_proba
        y_proba = model.decision_function(X_test_arr)
    acc = accuracy_score(y_test_arr, y_pred)
    roc = roc_auc_score(y_test_arr, y_proba)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_arr, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_arr, y_pred, target_names=['Not Defaulted (0)', 'Defaulted (1)']))
    return {"accuracy": acc, "roc_auc": roc, "y_pred": y_pred}

results = {}
results["Logistic Regression"] = evaluate_model("Logistic Regression", lr, X_test_scaled, y_test)

# -----------------------
# 6. Random Forest (default hyperparams)
# -----------------------
print("\nTraining Random Forest (default hyperparameters) on resampled data...")
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
t0 = time()
rf.fit(X_train_res, y_train_res)
t1 = time()
print(f"Trained Random Forest in {t1-t0:.2f}s")
joblib.dump(rf, "models/random_forest_default.joblib")
results["Random Forest (default)"] = evaluate_model("Random Forest (default)", rf, X_test_scaled, y_test)

# Feature importances (RF)
rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Random Forest feature importances:")
print(rf_importances.head(10))

# -----------------------
# 7. XGBoost (default)
# -----------------------
print("\nTraining XGBoost (default hyperparameters) on resampled data...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
t0 = time()
xgb.fit(X_train_res, y_train_res)
t1 = time()
print(f"Trained XGBoost in {t1-t0:.2f}s")
joblib.dump(xgb, "models/xgboost_default.joblib")
results["XGBoost (default)"] = evaluate_model("XGBoost (default)", xgb, X_test_scaled, y_test)

# Feature importances (XGB)
xgb_importances = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop 10 XGBoost feature importances:")
print(xgb_importances.head(10))

# -----------------------
# 8. Quick Hyperparameter Tuning (RandomizedSearchCV)
# -----------------------
print("\nStarting randomized hyperparameter search for Random Forest and XGBoost (short runs)...")
# Random Forest param grid
rf_param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [6, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, None]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_dist,
    n_iter=12,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
t0 = time()
rf_search.fit(X_train_res, y_train_res)
t1 = time()
print(f"Random Forest randomized search done in {t1-t0:.2f}s. Best ROC-AUC: {rf_search.best_score_:.4f}")
best_rf = rf_search.best_estimator_
joblib.dump(best_rf, "models/random_forest_tuned.joblib")
results["Random Forest (tuned)"] = evaluate_model("Random Forest (tuned)", best_rf, X_test_scaled, y_test)

# XGBoost param grid
xgb_param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
    xgb_param_dist,
    n_iter=12,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
t0 = time()
xgb_search.fit(X_train_res, y_train_res)
t1 = time()
print(f"XGBoost randomized search done in {t1-t0:.2f}s. Best ROC-AUC: {xgb_search.best_score_:.4f}")
best_xgb = xgb_search.best_estimator_
joblib.dump(best_xgb, "models/xgboost_tuned.joblib")
results["XGBoost (tuned)"] = evaluate_model("XGBoost (tuned)", best_xgb, X_test_scaled, y_test)

# -----------------------
# 9. Compile results summary
# -----------------------
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        "model": name,
        "accuracy": res["accuracy"],
        "roc_auc": res["roc_auc"]
    })
summary_df = pd.DataFrame(summary_rows).sort_values(by="roc_auc", ascending=False)
summary_df.to_csv("models/model_results_summary.csv", index=False)
print("\nModel comparison summary saved to models/model_results_summary.csv")
print(summary_df)

# -----------------------
# 10. Save top feature importance CSVs
# -----------------------
rf_importances.head(50).to_csv("models/rf_feature_importances.csv")
xgb_importances.head(50).to_csv("models/xgb_feature_importances.csv")
print("\nSaved feature importances for RF and XGB to models/ folder.")

print("\nCheckpoint 2 script finished. Models + artifacts saved in the 'models' folder.")
print("Next: inspect model_results_summary.csv and the classification reports above to choose the best model for final tuning / deploy.")
