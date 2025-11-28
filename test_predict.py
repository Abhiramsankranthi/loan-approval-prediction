from predict_helper import predict

sample = {
    "person_income": 50000,
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "person_emp_length": 5,
    "loan_percent_income": 0.2,
    "person_home_ownership": "RENT",
    "loan_intent": "EDUCATION",
    "loan_grade": "C"
}

print(predict(sample))
