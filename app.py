# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predict_helper import predict

app = FastAPI(title="Loan Approval Predictor")

class LoanInput(BaseModel):
    person_income: Optional[float]
    loan_amnt: Optional[float]
    loan_int_rate: Optional[float]
    person_emp_length: Optional[float]
    loan_percent_income: Optional[float]
    person_age: Optional[float]
    person_home_ownership: Optional[str]
    loan_intent: Optional[str]
    loan_grade: Optional[str]
    cb_person_default_on_file: Optional[str]
    cb_person_cred_hist_length: Optional[float]

@app.post("/predict")
def predict_loan(data: LoanInput):
    raw = data.dict()
    res = predict(raw)
    return res

# Run with uvicorn: uvicorn app:app --reload --port 8000
