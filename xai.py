import streamlit as st
import numpy as np
import pandas as pd
import lime.lime_tabular
import joblib
from sklearn.model_selection import train_test_split

# Load trained RandomForest model
from cibil_model import rf_model  

# # Load dataset for LIME
# X = pd.read_csv("X_data.csv")  # Your feature dataset
# y = pd.read_csv("y_data.csv")  # Target variable

# # Page Configuration
# st.set_page_config(page_title="CIBIL Score Predictor with XAI", layout="wide")

# Define feature names based on your app inputs
feature_names = [
    "Annual Revenue", "Loan Amount", "GST Compliance",
    "Repayment History", "Credit Utilization Ratio", "Credit Age"
]

# Function to explain prediction with LIME
def explain_with_lime(user_input):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.random.rand(100, len(feature_names)),  # Dummy data for LIME
        feature_names=feature_names,  # Use manually defined feature names
        mode='regression',
        random_state=42
    )

    explanation = explainer.explain_instance(
        data_row=user_input.values[0],
        predict_fn=rf_model.predict
    )

    explanation_html = "lime_explanation.html"
    explanation.save_to_file(explanation_html)
    
    return explanation_html

# Predict Button
if st.sidebar.button("Generate CIBIL Score"):
    user_input = np.array([
        annual_revenue, loan_amount, gst_billing, repayment_history, credit_utilization_manual, credit_age
    ]).reshape(1, -1)

    user_input_df = pd.DataFrame(user_input, columns=X.columns)
    predicted_score = rf_model.predict(user_input_df)[0]
    predicted_score = np.clip(predicted_score, 300, 900)

    st.subheader("Predicted CIBIL Score")
    st.markdown(f"<h2 style='color:blue;'>{predicted_score:.2f}</h2>", unsafe_allow_html=True)

    # Explain with LIME
    explanation_file = explain_with_lime(user_input_df)
    st.subheader("Feature Importance (LIME)")
    st.markdown("🔍 Download LIME Explanation:")
    st.download_button(label="Download Explanation", data=open(explanation_file, "rb"), file_name="LIME_CIBIL_Explanation.html")

