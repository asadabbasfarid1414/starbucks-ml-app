
# Machine Learning Model
st.header("🤖 Coffee Nutrition Prediction")

model_path = os.path.join(
    BASE_DIR,
    "starbucks_decision_tree(1).pkl"
)

# Original training features
TRAINING_FEATURES = [
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

if not os.path.exists(model_path):
    st.error(
        "Model file nahi mili. GitHub par check karein ke file ka naam "
        "starbucks_decision_tree(1).pkl hai."
    )
else:
    try:
        loaded_object = joblib.load(model_path)
        model = loaded_object

        # Handle a model saved inside a dictionary
        if isinstance(loaded_object, dict):
            model = None

            for key in [
                "model", "pipeline", "classifier",
                "decision_tree", "estimator", "best_model"
            ]:
                candidate = loaded_object.get(key)
                if hasattr(candidate, "predict"):
                    model = candidate
                    break

            if model is None:
                candidates = [
                    value for value in loaded_object.values()
                    if hasattr(value, "predict")
                ]
                if len(candidates) == 1:
                    model = candidates[0]

            if model is None:
                st.error(
                    "PKL file ek dictionary hai, lekin us mein prediction "
                    "model identify nahi hua."
                )
                st.write("PKL dictionary keys:", list(loaded_object.keys()))

        if model is not None:
            if not hasattr(model, "predict"):
                st.error(
                    "Loaded object mein predict() method nahi hai. "
                    "Sahi trained model ya pipeline PKL file mein save karein."
                )
            else:
                st.success("Decision Tree model loaded successfully!")

                # Detect original model input columns if available
                if hasattr(model, "feature_names_in_"):
                    features = list(model.feature_names_in_)
                elif isinstance(loaded_object, dict) and isinstance(
                    loaded_object.get("feature_names"), (list, tuple)
                ):
                    features = list(loaded_object["feature_names"])
                else:
                    features = TRAINING_FEATURES

                st.write(
                    "Prediction ke liye nutrition values enter karein."
                )

                with st.form("coffee_prediction_form"):
                    inputs = {}

                    for feature in features:
                        inputs[feature] = st.number_input(
                            feature.strip(),
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"prediction_{feature}"
                        )

                    predict_button = st.form_submit_button(
                        "Predict Beverage Category"
                    )

                if predict_button:
                    input_df = pd.DataFrame(
                        [inputs],
                        columns=features
                    )

                    try:
                        prediction = model.predict(input_df)
                        st.success(
                            f"Predicted Beverage Category: {prediction[0]}"
                        )
                    except Exception as prediction_error:
                        st.error(
                            f"Prediction failed: {prediction_error}"
                        )

    except Exception as e:
        st.error(f"Model loading error: {e}")

st.caption("Starbucks Coffee EDA and Machine Learning Project")
