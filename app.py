
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Starbucks Decision Tree",
    page_icon="☕",
    layout="centered"
)

# App title
st.title("☕ Starbucks ML Prediction App")
st.write("Machine Learning Model using Decision Tree")
st.divider()

# Load trained model
MODEL_PATH = "decision_tree_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.success("Decision Tree model loaded successfully!")

    st.subheader("Make a Prediction")
    st.info(
        "Model load ho gaya hai. Prediction inputs "
        "aapke training dataset ke columns ke mutabiq "
        "set karne honge."
    )

else:
    st.error(
        "Model file nahi mili! "
        "decision_tree_model.pkl ko app.py ke "
        "same folder mein rakhein."
    )

st.divider()
st.caption("Starbucks Data Analysis | Decision Tree ML Project")
