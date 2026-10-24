
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

st.set_page_config(
    page_title="Starbucks Coffee EDA",
    page_icon="☕",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.title("☕ Starbucks Coffee Analysis")
st.write("Explore Starbucks coffee nutrition data and predictions.")

# Load CSV dataset
@st.cache_data
def load_data():
    csv_path = os.path.join(BASE_DIR, "starbucks.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

coffee_df = load_data()

if coffee_df is None:
    st.error("starbucks.csv file nahi mili. GitHub par filename check karein.")
    st.stop()

# Dataset overview
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Total Records", coffee_df.shape[0])
col2.metric("Total Columns", coffee_df.shape[1])

st.subheader("Coffee Dataset")
st.dataframe(coffee_df, use_container_width=True)

st.subheader("Missing Values")
st.dataframe(coffee_df.isnull().sum().rename("Missing Values"))

# EDA
numeric_df = coffee_df.select_dtypes(include="number")
numeric_columns = numeric_df.columns.tolist()

if numeric_columns:
    st.header("📈 Exploratory Data Analysis")

    selected_column = st.selectbox(
        "Select a nutrition column",
        numeric_columns
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Average", f"{numeric_df[selected_column].mean():.2f}")
    col2.metric("Minimum", f"{numeric_df[selected_column].min():.2f}")
    col3.metric("Maximum", f"{numeric_df[selected_column].max():.2f}")

    st.subheader(f"{selected_column} Distribution")
    fig, ax = plt.subplots()
    numeric_df[selected_column].dropna().hist(bins=20, ax=ax)
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Number of drinks")
    st.pyplot(fig)
    plt.close(fig)

    if len(numeric_columns) >= 2:
        st.subheader("Nutrition Correlation")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = numeric_df.corr()
        im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Summary Statistics")
    st.dataframe(numeric_df.describe())

# Machine learning model
st.header("🤖 Coffee Nutrition Prediction")

model_path = os.path.join(
    BASE_DIR,
    "starbucks_decision_tree(1).pkl"
)

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Decision Tree model loaded successfully!")

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            features = []

        if features and all(f in coffee_df.columns for f in features):
            inputs = {}

            for feature in features:
                values = pd.to_numeric(
                    coffee_df[feature], errors="coerce"
                )
                default_value = values.median()
                if pd.isna(default_value):
                    default_value = 0.0

                inputs[feature] = st.number_input(
                    feature, value=float(default_value)
                )

            if st.button("Predict"):
                input_df = pd.DataFrame([inputs], columns=features)
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")
        else:
            st.info(
                "Model ke training features identify nahi huay. "
                "Prediction ke liye original training columns check karein."
            )

    except Exception as e:
        st.error(f"Model error: {e}")
else:
    st.warning(
        "Model file nahi mili. GitHub par "
        "starbucks_decision_tree(1).pkl ka exact naam check karein."
    )

st.caption("Starbucks Coffee EDA and Machine Learning Project")
