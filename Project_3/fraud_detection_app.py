import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import xgboost as xgb

@st.cache_resource
def load_model():
    with open("balanced/xgb_model_balanced.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

with open("feature_stats.json", "r") as f:
    feature_stats = json.load(f)

# --- Example values for testing ---
fraud_example = {
    'V1': -2.3122265423263,
    'V2': 1.95199201064158,
    'V3': -1.60985073229769,
    'V4': 3.9979055875468,
    'V5': -0.522187864667764,
    'V6': -1.42654531920595,
    'V7': -2.53738730624579,
    'V8': 1.39165724829804,
    'V9': -2.77008927719433,
    'V10': -2.77227214465915,
    'V11': 3.20203320709635,
    'V12': -2.89990738849473,
    'V13': -0.595221881324605,
    'V14': -4.28925378244217,
    'V15': 0.389724120274487,
    'V16': -1.14074717980657,
    'V17': -2.83005567450437,
    'V18': -0.0168224681808257,
    'V19': 0.416955705037907,
    'V20': 0.126910559061474,
    'V21': 0.517232370861764,
    'V22': -0.0350493686052974,
    'V23': -0.465211076182388,
    'V24': 0.320198198514526,
    'V25': 0.0445191674731724,
    'V26': 0.177839798284401,
    'V27': 0.261145002567677,
    'V28': -0.143275874698919,
    'Amount': 256.0,
    'Hour': 0.0
}

legit_example = {
    'V1': 0.1, 'V2': 0.2, 'V3': -0.1, 'V4': 0.3, 'V5': -0.2, 'V6': 0.1, 'V7': -0.3,
    'V8': 0.2, 'V9': -0.2, 'V10': 0.1, 'V11': -0.1, 'V12': 0.3, 'V13': -0.3,
    'V14': 0.1, 'V15': -0.2, 'V16': 0.1, 'V17': -0.1, 'V18': 0.2, 'V19': -0.2,
    'V20': 0.1, 'V21': -0.1, 'V22': 0.2, 'V23': -0.1, 'V24': 0.1, 'V25': -0.1,
    'V26': 0.1, 'V27': -0.2, 'V28': 0.2, 'Amount': 30.0, 'Hour': 12.0
}

# --- Streamlit UI ---
st.title("Credit Card Fraud Detector")

st.markdown("Set values manually using sliders or use one of the test buttons to fill inputs.")

# --- Preset Buttons ---
col1, col2 = st.columns(2)
with col1:
    if st.button("⚠️ Set Fraud Example"):
        st.session_state["values"] = fraud_example.copy()
with col2:
    if st.button("✅ Set Legit Example"):
        st.session_state["values"] = legit_example.copy()

# --- Initialize input values ---
st.session_state.setdefault("values", {})
input_values = []

st.markdown("### Input Features")

# Sliders for input
for feature, stats in feature_stats.items():
    min_val = float(stats["Min"])
    max_val = float(stats["Max"])
    mean_val = float(stats["Mean"])

    default_val = st.session_state["values"].get(feature, mean_val)

    val = st.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val - min_val) / 100,
    )
    input_values.append(val)

# --- Prediction ---
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_values], columns=feature_stats.keys())
    prediction = model.predict_proba(input_df)[0][1]  # Probability for class 1

    st.subheader("Prediction Result")

    if prediction >= 0.5:
        st.error("🚨 This transaction is likely **fraudulent**.")
    else:
        st.success("✅ This transaction is likely **legitimate**.")
