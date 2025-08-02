# Paris Housing Price Predictor (Streamlit)

This app loads a **trained scikit-learn pipeline** (`paris_price_model.pkl`) that already includes preprocessing
and a tree-based model, and serves predictions with SHAP explanations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place your saved pipeline in the same folder:

```python
# in your notebook
import joblib
joblib.dump(best_pipe, "paris_price_model.pkl")
```

## Deploy on Streamlit Cloud

1. Push to GitHub: `app.py`, `requirements.txt`, and `paris_price_model.pkl`
2. Create a new app on https://streamlit.io/cloud
3. Select your repo and set **Main file** to `app.py`