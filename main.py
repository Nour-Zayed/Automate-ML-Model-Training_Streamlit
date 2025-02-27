import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml_utility import (read_data, preprocess_data, train_model, evaluate_model)

st.set_page_config(page_title="Automate ML", page_icon="🧠", layout="centered")

st.title("ML Model Training No Code ")

# 🆕 Only allow file upload
uploaded_file = st.file_uploader("📂 **Upload Your CSV file**", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    st.warning("⚠️ Please upload a CSV file to continue.")
    df = None

# Display the data if available
if df is not None:
    st.dataframe(df.head())

    col1, col2, col3, col4 = st.columns(4)

    scaler_type_list = ["standard", "minmax"]

    model_dictionary = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Classifier": SVC(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XGBoost Classifier": XGBClassifier()
    }

    with col1:
        target_column = st.selectbox("🎯 Select Target Column", list(df.columns))
    with col2:
        scaler_type = st.selectbox("⚙️ Select Scaler", scaler_type_list)
    with col3:
        selected_model = st.selectbox("🧠 Select Model", list(model_dictionary.keys()))
    with col4:
        model_name = st.text_input("✍️ Model Name")

    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
        model_to_be_trained = model_dictionary[selected_model]
        model = train_model(X_train, y_train, model_to_be_trained, model_name)
        accuracy = evaluate_model(model, X_test, y_test)
        st.success(f"🏆 Model Accuracy: {accuracy:.2%}")
