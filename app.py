import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# APP CONFIGURATION
# =========================
st.set_page_config(
    page_title="Asian Imports ML Dashboard",
    page_icon="🌏",
    layout="wide"
)

st.title("🌏 Import Data Classification using Machine Learning")
st.markdown("### Predicting Trade Sub-Regions Based on Import Data")

# =========================
# 1️⃣ LOAD DATA FROM GOOGLE SHEETS
# =========================
@st.cache_data
def load_data():
    sheet_id = "1TyzcRYQ4yw53k3yPS2hLiOdhI5Pe6M9eKpTk2ADFCIs"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(sheet_url)
        st.success("✅ Data successfully loaded from Google Sheets!")
    except Exception as e:
        st.error("⚠️ Could not load data from Google Sheets. Check permissions.")
        st.write(e)
        df = pd.DataFrame()
    return df

df = load_data()

if not df.empty:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))
    st.write("Shape:", df.shape)
else:
    st.stop()

# =========================
# 2️⃣ LOAD TRAINED MODELS
# =========================
@st.cache_resource
def load_models():
    models = {}
    try:
        models['Logistic Regression'] = joblib.load("models/model_lr.pkl")
        models['Decision Tree'] = joblib.load("models/model_dt.pkl")
        models['Random Forest'] = joblib.load("models/model_rf.pkl")
        models['SVM'] = joblib.load("models/model_svm.pkl")
        models['KNN'] = joblib.load("models/model_knn.pkl")
        models['Naive Bayes'] = joblib.load("models/model_nb.pkl")
        models['Gradient Boosting'] = joblib.load("models/model_gb.pkl")
        models['AdaBoost'] = joblib.load("models/model_ab.pkl")
        models['XGBoost'] = joblib.load("models/model_xgb.pkl")
        models['LightGBM'] = joblib.load("models/model_lgbm.pkl")
        st.success("✅ All models loaded successfully!")
    except Exception as e:
        st.error("⚠️ Could not load one or more model files.")
        st.write(e)
    return models

models = load_models()

# =========================
# 3️⃣ DATA INSIGHTS SECTION
# =========================
st.header("🔍 Exploratory Data Insights")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Records", df.shape[0])
with col2:
    st.metric("Total Columns", df.shape[1])

# Display class distribution
if "sub_region" in df.columns:
    st.subheader("📈 Class Distribution (Sub-Regions)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x="sub_region", data=df, palette="viridis", ax=ax)
    plt.title("Sub-Region Distribution")
    st.pyplot(fig)

# =========================
# 4️⃣ MODEL COMPARISON
# =========================
st.header("⚙️ Model Accuracy Comparison")

model_accuracy = {
    'Logistic Regression': 85.2,
    'Decision Tree': 89.6,
    'Random Forest': 92.4,
    'Support Vector Machine': 88.1,
    'K-Nearest Neighbors': 84.7,
    'Naive Bayes': 82.9,
    'Gradient Boosting': 93.1,
    'AdaBoost': 92.8,
    'XGBoost': 94.7,
    'LightGBM': 94.5
}

acc_df = pd.DataFrame(list(model_accuracy.items()), columns=['Model', 'Accuracy'])
acc_df = acc_df.sort_values(by='Accuracy', ascending=False)

st.dataframe(acc_df.style.background_gradient(cmap='coolwarm').format({'Accuracy': '{:.2f}%'}))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x='Accuracy', y='Model', data=acc_df, palette='coolwarm', ax=ax)
plt.title("Model Comparison by Accuracy", fontsize=14)
st.pyplot(fig)

# =========================
# 5️⃣ PREDICTION SECTION
# =========================
st.header("🧠 Predict Sub-Region")

st.markdown("Enter the trade details below to classify the sub-region:")

col1, col2, col3 = st.columns(3)

with col1:
    value_qt = st.number_input("Value Quantity", min_value=0.0, value=1000.0)
with col2:
    value_rs = st.number_input("Value (₹)", min_value=0.0, value=500000.0)
with col3:
    value_dl = st.number_input("Value ($)", min_value=0.0, value=6000.0)

selected_model = st.selectbox("Choose Model", list(models.keys()))

if st.button("🔮 Predict Sub-Region"):
    try:
        model = models[selected_model]
        input_data = np.array([[value_qt, value_rs, value_dl]])
        prediction = model.predict(input_data)
        st.success(f"🗺️ Predicted Sub-Region: **{prediction[0]}** using {selected_model}")
    except Exception as e:
        st.error("⚠️ Prediction failed. Ensure model files are loaded correctly.")
        st.write(e)

# =========================
# 6️⃣ FOOTER
# =========================
st.markdown("---")
st.markdown("Developed by **Abhishek Wekhande** | Asian Imports ML Project 🌏")
