
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Paris Housing Price Predictor", layout="centered")
st.title("🏡 Paris Housing Price Predictor")

st.markdown(
    "Provide property details and get an estimated price using the **trained pipeline** "
    "(`paris_price_model.pkl`) which includes preprocessing and a tree-based model."
)

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

# Path to saved pipeline (exported from notebook)
model_path = st.sidebar.text_input("Model file path", "paris_price_model.pkl")
try:
    pipe = load_pipeline(model_path)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.sidebar.error(f"Could not load model at '{model_path}': {e}")
    st.stop()

st.subheader("Enter Property Details")

col1, col2, col3 = st.columns(3)
with col1:
    squareMeters = st.number_input("Square Meters", min_value=10.0, max_value=200000.0, value=1000.0, step=10.0)
    hasPool = st.selectbox("Has Pool?", [0, 1])
    area_per_room = st.number_input("Area per Room", min_value=1.0, max_value=100000.0, value=30.0, step=1.0)
with col2:
    rooms_per_floor = st.number_input("Rooms per Floor", min_value=0.1, max_value=100.0, value=3.0, step=0.1)
    hasYard = st.selectbox("Has Yard?", [0, 1])
    isNewBuilt = st.selectbox("Is New Built?", [0, 1])
with col3:
    luxury_score = st.slider("Luxury Score (0–4)", 0, 4, 1)
    hasStorageRoom = st.selectbox("Has Storage Room?", [0, 1])
    hasStormProtector = st.selectbox("Has Storm Protector?", [0, 1])

# Keep column names consistent with training pipeline
input_df = pd.DataFrame([{
    "squareMeters": squareMeters,
    "rooms_per_floor": rooms_per_floor,
    "luxury_score": luxury_score,
    "hasPool": hasPool,
    "hasYard": hasYard,
    "area_per_room": area_per_room,
    "isNewBuilt": isNewBuilt,
    "hasStorageRoom": hasStorageRoom,
    "hasStormProtector": hasStormProtector
}])

st.write("**Model input preview**")
st.dataframe(input_df)

def get_processed_feature_names(pipeline, input_df):
    try:
        prep = pipeline.named_steps["prep"]
    except Exception:
        return input_df.columns.to_numpy()

    cat_names = np.array([])
    num_names = np.array([])

    try:
        ohe = prep.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = prep.transformers_[0][2]
        cat_names = ohe.get_feature_names_out(cat_cols)
    except Exception:
        pass

    try:
        num_names = np.array(prep.transformers_[1][2])
    except Exception:
        # fallback length check
        tx = prep.transform(input_df)
        n = tx.shape[1] - len(cat_names)
        num_names = np.array([f"num_{i}" for i in range(n)])

    return np.r_[cat_names, num_names]

def predict_and_explain(pipeline, row_df):
    # Transform
    try:
        Xt = pipeline.named_steps["prep"].transform(row_df)
        model = pipeline.named_steps["model"]
    except Exception:
        Xt = row_df.values
        model = pipeline

    # Predict
    yhat = float(model.predict(Xt)[0])

    # SHAP (tree explainer)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xt)
    base_value = explainer.expected_value

    feature_names = get_processed_feature_names(pipeline, row_df)
    row_sv = shap_values[0]
    row_data = Xt[0].toarray().ravel() if hasattr(Xt, "toarray") else np.array(Xt[0]).ravel()

    exp = shap.Explanation(values=row_sv,
                           base_values=base_value,
                           data=row_data,
                           feature_names=feature_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()
    return yhat, fig

if st.button("Predict & Explain"):
    try:
        price, shap_fig = predict_and_explain(pipe, input_df)
        st.success(f"💰 Predicted Price: €{price:,.2f}")
        st.markdown("### 🔎 SHAP Explanation (Waterfall)")
        st.pyplot(shap_fig)
        st.caption("Positive bars push the prediction up relative to the model baseline; negative bars push it down.")
    except Exception as e:
        st.error(f"Prediction/Explanation failed: {e}")
