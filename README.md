# Automated Loan Approval Prediction System

This repository contains the complete machine learning system developed for predicting loan default risk using the Kaggle Playground Series S4E10 dataset.  
The system includes data preprocessing, model development, evaluation, threshold tuning, a FastAPI backend, and a Streamlit user interface.

---

## 🚀 Project Overview

This project builds an end-to-end machine learning pipeline to predict whether a loan applicant will default (1) or not default (0).  
Key steps include:

- Exploratory Data Analysis (EDA)
- Missing value handling
- One-hot encoding of categorical features
- Log transformation of skewed features
- Feature scaling
- Handling class imbalance with SMOTE
- Training multiple models (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning
- Threshold optimization
- Deployment using FastAPI + Streamlit

The final production model uses **XGBoost (tuned)** with threshold tuning for improved recall.

---

## 📂 Repository Structure

├── app.py # FastAPI backend
├── streamlit_app.py # Streamlit UI
├── predict_helper.py # Preprocessing + model inference
├── build_prod_pipeline.py # Builds production pipeline
├── checkpoint2.py # Training & evaluation script
├── explore_data.py # EDA visualizations
├── models/
│ ├── prod_pipeline.joblib # Final pipeline (scaler + model + features + threshold)
│ ├── xgboost_tuned.joblib # Final trained model
│ ├── scaler.joblib # StandardScaler
│ ├── roc_pr_curves.png # ROC & PR figure
│ └── threshold_tuning.png # Precision–Recall vs Threshold
├── train.csv # Training data
└── README.md # This file


---

## 🧪 How to Run the System

### 1. Install dependencies

pip install -r requirements.txt

python build_prod_pipeline.py

python -m uvicorn app:app --reload

streamlit run streamlit_app.py

http://localhost:8501



---

## 🧠 Model Info

- **Final Model:** XGBoost (tuned)
- **Threshold:** 0.35  
- **Balanced using SMOTE**
- **Best ROC–AUC:** ~0.95
- **Key Features:**
  - loan_percent_income  
  - loan_int_rate  
  - loan_grade  
  - home_ownership  

---

## 📊 Explainability

The Streamlit UI displays:

- Default probability  
- Final “Approved/Rejected” loan decision  
- Top contributing features (via model feature importance)

---

## 📄 Report

This code accompanies the final report submitted for  
**Data Mining – Final Project**.

The full IEEE-format report text is included in the project submission PDF.

---

## 🔗 Code Access for Submission

You may access all source code here:  
**(Insert your GitHub repo link here)**

---

## 👥 Team Members

- **Abhiram Sankranthi** – Modeling, API, Technical Development  
- **Yagnitha Challagurugula** – EDA, UI, Baseline Modeling, Documentation

---

## 📜 License

This project is for academic purposes only.

