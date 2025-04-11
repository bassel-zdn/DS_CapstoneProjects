import streamlit as st

st.set_page_config(page_title="Employee Churn Predictor", layout="centered")

import pandas as pd
import numpy as np
import pickle

# Cache the model loading function
@st.cache_resource
def load_model():
    with open("best_rf_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("👥 Employee Churn Prediction App")

st.markdown("""
Use this app to predict whether an employee is likely to leave the company based on their profile.
""")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Employee Information")

    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5, step=0.01)
    number_project = st.slider("Number of Projects", 0, 7, 3)
    average_montly_hours = st.slider("Average Monthly Hours", 96, 310, 160, step=1)
    time_spend_company = st.slider("Years at Company", 2, 10, step=1)

    submit = st.form_submit_button("Predict")

# Prediction logic
if submit:
    user_input = [
        satisfaction_level, last_evaluation, number_project,
        average_montly_hours, time_spend_company
    ]

    feature_names = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company'
    ]

    input_df = pd.DataFrame([user_input], columns=feature_names)

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("This employee is likely to leave.")
    else:
        st.success("This employee is likely to stay.")

    st.caption("Model: Tuned Random Forest")
