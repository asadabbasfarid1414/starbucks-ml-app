import os
import joblib
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "starbucks_decision_tree(1).pkl"
)

try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()
