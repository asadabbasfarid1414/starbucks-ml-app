
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

st.title("☕ Starbucks Coffee Analysis")
st.write("Explore coffee nutrition data and machine learning predictions.")

# Load dataset
@st.cache_data
def load_data():
    possible_files = [
        "starbucks.csv",
        "coffee.csv",
        "starbucks_drinkMenu_expanded.csv"
    ]

    for file in possible_files:
        if os.path.exists(file):
            return pd.read_csv(file)

    return None

coffee_df = load_data()

if coffee_df is None:
    st.warning(
        "Dataset CSV file nahi mili. "
        "Apni Starbucks CSV file GitHub repository mein upload karein."
    )
    st.stop()

# Dataset overview
st.header("📊 Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Total Records", coffee_df.shape[0])
col2.metric("Total Columns", coffee_df.shape[1])

st.subheader("Coffee Dataset")
st.dataframe(coffee_df, use_container_width=True)

st.subheader("Dataset Information")
st.write("Dataset shape:", coffee_df.shape)
st.write("Missing values per column:")
st.dataframe(coffee_df.isnull().sum().rename("Missing Values"))

# Numeric columns
numeric_df = coffee_df.select_dtypes(include="number")
numeric_columns = numeric_df.columns.tolist()

if numeric_columns:
    st.header("📈 Exploratory Data Analysis")

    selected_column = st.selectbox(
        "Select a nutrition column",
        numeric_columns
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Average",
        f"{numeric_df[selected_column].mean():.2f}"
    )
    col2.metric(
        "Minimum",
        f"{numeric_df[selected_column].min():.2f}"
    )
    col3.metric(
        "Maximum",
        f"{numeric_df[selected_column].max():.2f}"
    )

    st.subheader(f"{selected_column} Distribution")
    fig, ax = plt.subplots()
    numeric_df[selected_column].dropna().hist(
        bins=20, ax=ax
    )
    ax.set_xlabel(selected_column)
    ax.set_ylabel("Number of drinks")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Nutrition Correlation")
    if len(numeric_columns) >= 2:
        fig, ax = plt.subplots(figsize=(9, 5))
        corr = numeric_df.corr()
        image = ax.imshow(corr, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)
        fig.colorbar(image, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Summary Statistics")
    st.dataframe(numeric_df.describe())

# Decision Tree prediction
st.header("🤖 Coffee Nutrition Prediction")

model_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "starbucks_decision_tree(1).pkl"
)

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Decision Tree model loaded successfully!")

        st.write(
            "Prediction ke liye apne trained model ke "
            "required input features select karein."
        )

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        else:
            features = numeric_columns

        available_features = [
            col for col in features if col in coffee_df.columns
        ]

        if len(available_features) == len(features) and features:
            inputs = {}
            for feature in features:
                default_value = float(
                    pd.to_numeric(
                        coffee_df[feature], errors="coerce"
                    ).median()
                )
                if pd.isna(default_value):
                    default_value = 0.0

                inputs[feature] = st.number_input(
                    feature,
                    value=default_value
                )

            if st.button("Predict"):
                input_df = pd.DataFrame([inputs])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {prediction[0]}")
        else:
            st.info(
                "Model ke required features dataset columns se "
                "match nahi kar rahe. Training features check karein."
            )

    except Exception as e:
        st.error(f"Model load/prediction error: {e}")
else:
    st.warning(
        "Model file nahi mili. GitHub par "
        "'starbucks_decision_tree(1).pkl' ka exact naam check karein."
    )

st.caption("Starbucks Coffee EDA and Machine Learning Project")
