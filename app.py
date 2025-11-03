import streamlit as st
import pandas as pd
import joblib
import io
import requests
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# Streamlit App Config
# ------------------------------------------------------
st.set_page_config(page_title="Asian Imports ML Dashboard", layout="wide")
st.title("🌏 Asian Imports Classification Dashboard")
st.markdown("### Predict Sub-Region Based on Trade Data using 10 Machine Learning Models")

# ------------------------------------------------------
# 1️⃣ LOAD DATA FROM GOOGLE SHEETS
# ------------------------------------------------------
@st.cache_data
def load_data():
    sheet_id = "1TyzcRYQ4yw53k3yPS2hLiOdhI5Pe6M9eKpTk2ADFCIs"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(sheet_url)
        st.success("✅ Data successfully loaded from Google Sheets!")
    except Exception as e:
        st.error("⚠️ Could not load data from Google Sheets. Please check link or permissions.")
        st.write(e)
        df = pd.DataFrame()
    return df

df = load_data()

if not df.empty:
    with st.expander("📊 View Dataset Sample"):
        st.dataframe(df.head())

# ------------------------------------------------------
# 2️⃣ LOAD MODELS FROM GOOGLE DRIVE
# ------------------------------------------------------
def load_model_from_drive(drive_url):
    try:
        file_id = drive_url.split("/d/")[1].split("/")[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        response = requests.get(download_url)
        response.raise_for_status()
        model = joblib.load(io.BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model from {drive_url}")
        st.write(e)
        return None

@st.cache_resource
def load_all_models():
    model_links = {
        "Logistic Regression": "https://drive.google.com/file/d/1DhHDaRr3LCk_lDugaTcyycx5rvWDz80R/view?usp=drive_link",
        "Decision Tree": "https://drive.google.com/file/d/11HH3s8UVGsmVQnG-AWU564OkErXVidr3/view?usp=drive_link",
        "Random Forest": "https://drive.google.com/file/d/1JQ-BpP7Df63CPtLfGtkltl3Q08J7AxIp/view?usp=drive_link",
        "Support Vector Machine": "https://drive.google.com/file/d/1BsaMIlcCALgenEJJSSFAwMoAh0EUrQVj/view?usp=drive_link",
        "K-Nearest Neighbors": "https://drive.google.com/file/d/1YGxTONVB-xKuViuOctL_Kdpyo0qH5OT0/view?usp=drive_link",
        "Naive Bayes": "https://drive.google.com/file/d/14QQhqDJWgDhexdWC2g4RAN5FYK_K9-Mc/view?usp=drive_link",
        "Gradient Boosting": "https://drive.google.com/file/d/1ZR62i8Qda5CYH63UW81j4L2p9ln50mtZ/view?usp=drive_link",
        "AdaBoost": "https://drive.google.com/file/d/14p_ZU5sVu1quZphqRdWtUtQb5Hah7pnY/view?usp=drive_link",
        "CatBoost": "https://drive.google.com/file/d/1SoRvZli6Hy71iiB_w6mt7sWK_ljWfR0M/view?usp=drive_link"
    }

    models = {}
    for name, link in model_links.items():
        with st.spinner(f"Loading {name} model..."):
            model = load_model_from_drive(link)
            if model:
                models[name] = model
                st.success(f"✅ {name} loaded successfully!")
    return models

models = load_all_models()

# ------------------------------------------------------
# 3️⃣ USER INPUT SECTION (MATCHES TRAINED FEATURES)
# ------------------------------------------------------
st.sidebar.header("🔧 Input Trade Features")

st.sidebar.markdown("Enter all the feature values below:")

val_qt = st.sidebar.number_input("Value Quantity (value_qt)", min_value=0.0, step=0.1)
val_rs = st.sidebar.number_input("Value (₹) (value_rs)", min_value=0.0, step=0.1)
val_dl = st.sidebar.number_input("Value ($) (value_dl)", min_value=0.0, step=0.01)
region_code = st.sidebar.number_input("Region Code", min_value=0, step=1)
sub_region_code = st.sidebar.number_input("Sub-Region Code", min_value=0, step=1)
hs_code = st.sidebar.number_input("HS Code", min_value=0, step=1)
unit = st.sidebar.number_input("Unit (Encoded)", min_value=0, step=1)
commodity = st.sidebar.number_input("Commodity (Encoded)", min_value=0, step=1)

# Create input DataFrame with all 8 features
input_data = pd.DataFrame({
    "value_qt": [val_qt],
    "value_rs": [val_rs],
    "value_dl": [val_dl],
    "region_code": [region_code],
    "sub_region_code": [sub_region_code],
    "hs_code": [hs_code],
    "unit": [unit],
    "commodity": [commodity]
})

st.write("### 🧾 Input Data Preview")
st.dataframe(input_data)

# ------------------------------------------------------
# 4️⃣ MODEL SELECTION AND PREDICTION
# ------------------------------------------------------
if models:
    model_choice = st.selectbox("Select Model for Prediction", list(models.keys()))

    if st.button("🚀 Predict Sub-Region"):
        model = models.get(model_choice)

        if model:
            st.info(f"Running prediction using **{model_choice}**...")
            try:
                scaler = StandardScaler()
                scaled_input = scaler.fit_transform(input_data)

                pred = model.predict(scaled_input)
                st.success(f"✅ Predicted Sub-Region: **{pred[0]}**")

            except Exception as e:
                st.error("⚠️ Error during prediction.")
                st.write(e)
        else:
            st.warning("Please wait for model to finish loading.")
else:
    st.warning("⚠️ No models were loaded. Check Drive links or internet connection.")

# ------------------------------------------------------
# 5️⃣ FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown("📘 **Developed by Abhishek Wekhande** | Data sourced from Open Government Data Platform India")
