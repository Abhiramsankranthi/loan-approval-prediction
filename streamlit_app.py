# streamlit_app.py
import streamlit as st
from predict_helper import predict

st.set_page_config(page_title="Loan Approval Demo", layout="centered")

st.title("Loan Approval Prediction Demo")
st.write("Enter applicant details and get a predicted probability of default (1 = default).")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    person_income = st.number_input("Person income (annual)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
    loan_amnt = st.number_input("Loan amount", min_value=0.0, value=10000.0, step=500.0, format="%.2f")
    loan_int_rate = st.number_input("Loan interest rate (%)", min_value=0.0, value=10.5, step=0.1, format="%.2f")
    person_emp_length = st.number_input("Employment length (years)", min_value=0.0, value=3.0, step=1.0, format="%.1f")
    loan_percent_income = st.number_input("Loan percent of income (0-1)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, format="%.3f")

with col2:
    person_age = st.number_input("Applicant age", min_value=18, max_value=100, value=30)
    person_home_ownership = st.selectbox("Home ownership", ["OWN", "RENT", "OTHER"])
    loan_intent = st.selectbox("Loan intent", ["PERSONAL", "EDUCATION", "HOMEIMPROVEMENT", "VENTURE", "MEDICAL"])
    loan_grade = st.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
    cb_person_default_on_file = st.selectbox("Prior default on file?", ["N", "Y"])

st.markdown("---")

if st.button("Predict"):
    payload = {
        "person_income": person_income,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "person_emp_length": person_emp_length,
        "loan_percent_income": loan_percent_income,
        "person_age": person_age,
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "cb_person_default_on_file": cb_person_default_on_file,
        # optional: add cb_person_cred_hist_length if you want
    }

    try:
        res = predict(payload)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    else:
        prob = res.get("probability", None)
        pred = res.get("prediction", None)
        top = res.get("top_features", [])

        st.subheader("Prediction")
        if prob is not None:
            st.metric("Probability of default", f"{prob:.3f}", delta=None)
        if pred is not None:
            decision = res.get("decision", "N/A")

            if decision == "Approved":
                st.success(f"Loan Decision: {decision}")
            else:
                st.error(f"Loan Decision: {decision}")
            st.write(f"(Model probability = {prob:.3f}, threshold = {res.get('threshold')})")

        if top:
            st.subheader("Top contributing features (approx.)")
            # top is list of tuples (feature, importance)
            df_top = { "Feature": [f[0] for f in top], "Importance": [f[1] for f in top] }
            import pandas as _pd
            st.table(_pd.DataFrame(df_top))
        else:
            st.write("No feature importances available for this model.")

st.markdown("---")
st.write("Notes: This demo uses the local production pipeline produced by `build_prod_pipeline.py` and `predict_helper.py`.")
