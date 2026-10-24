
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Starbucks Beverage Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Beverage Category Predictor")
st.write(
    "Enter the nutritional information to predict "
    "the beverage category."
)

# Load model
MODEL_PATH = Path(__file__).parent / "starbucks_decision_tree (1).pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# Exact feature names and order used during training
features = [
    "Calories",
    "Trans Fat (g) ",
    "Saturated Fat (g)",
    " Sodium (mg)",
    " Total Carbohydrates (g) ",
    "Cholesterol (mg)",
    " Dietary Fibre (g)",
    " Sugars (g)",
    " Protein (g) "
]

st.subheader("Nutritional Information")

with st.form("prediction_form"):
    calories = st.number_input("Calories", min_value=0.0, value=150.0)
    trans_fat = st.number_input("Trans Fat (g)", min_value=0.0, value=0.0)
    saturated_fat = st.number_input("Saturated Fat (g)", min_value=0.0, value=0.0)
    sodium = st.number_input("Sodium (mg)", min_value=0.0, value=0.0)
    carbohydrates = st.number_input("Total Carbohydrates (g)", min_value=0.0, value=20.0)
    cholesterol = st.number_input("Cholesterol (mg)", min_value=0.0, value=0.0)
    fibre = st.number_input("Dietary Fibre (g)", min_value=0.0, value=0.0)
    sugars = st.number_input("Sugars (g)", min_value=0.0, value=15.0)
    protein = st.number_input("Protein (g)", min_value=0.0, value=5.0)

    submitted = st.form_submit_button("Predict Beverage Category")

if submitted:
    input_data = pd.DataFrame(
        [[
            calories,
            trans_fat,
            saturated_fat,
            sodium,
            carbohydrates,
            cholesterol,
            fibre,
            sugars,
            protein
        ]],
        columns=features
    )

    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Beverage Category: {prediction[0]}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_
            results = pd.DataFrame({
                "Beverage Category": classes,
                "Probability (%)": probabilities * 100
            }).sort_values("Probability (%)", ascending=False)

            st.subheader("Prediction probabilities")
            st.dataframe(
                results.head(5).style.format(
                    {"Probability (%)": "{:.2f}%"}
                ),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check whether the model was trained with these exact "
            "feature names and the same preprocessing."
        )
