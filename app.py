import streamlit as st
import pandas as pd
import joblib
import io
import requests
from sklearn.preprocessing import StandardScaler
import base64

# ------------------------------------------------------
# 🌄 Add Background Image
# ------------------------------------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("pic1.png")

# ------------------------------------------------------
# 🎨 Streamlit Page Config & Styling
# ------------------------------------------------------
st.set_page_config(page_title="Asian Imports ML Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Title Styling */
    .title-text {
        text-align: center;
        color: white;
        font-size: 36px;
        font-weight: 800;
        text-shadow: 1px 1px 5px black;
    }
    /* Subheading */
    .subheader-text {
        text-align: center;
        color: #d4f1f9;
        font-size: 18px;
        margin-bottom: 30px;
        text-shadow: 1px 1px 3px black;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(0,0,0,0.65);
        color: white;
        border-right: 2px solid #3498db;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        height: 3em;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #217dbb;
        color: #f1f1f1;
    }
    /* DataFrame Background */
    .stDataFrame {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">🌏 Asian Imports ML Classification Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Analyze and Predict Sub-Regions using Machine Learning Models</div>', unsafe_allow_html=True)

# ------------------------------------------------------
# 1️⃣ LOAD DATA FROM GOOGLE SHEETS
# ------------------------------------------------------
@st.cache_data
def load_data():
    sheet_id = "1TyzcRYQ4yw53k3yPS2hLiOdhI5Pe6M9eKpTk2ADFCIs"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error("⚠️ Could not load data from Google Sheets.")
        st.write(e)
        df = pd.DataFrame()
    return df

df = load_data()

with st.expander("📊 View Dataset Sample"):
    st.dataframe(df.head())

# ------------------------------------------------------
# 2️⃣ LOAD MODELS FROM GOOGLE DRIVE (Quiet Loading)
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
        model = load_model_from_drive(link)
        if model:
            models[name] = model
    return models

models = load_all_models()

# ------------------------------------------------------
# 3️⃣ USER INPUT SECTION
# ------------------------------------------------------
st.sidebar.header("🔧 Input Trade Features")

val_qt = st.sidebar.number_input("Value Quantity (value_qt)", min_value=0.0, step=0.1)
val_rs = st.sidebar.number_input("Value (₹) (value_rs)", min_value=0.0, step=0.1)
val_dl = st.sidebar.number_input("Value ($) (value_dl)", min_value=0.0, step=0.01)
region_code = st.sidebar.number_input("Region Code", min_value=0, step=1)
sub_region_code = st.sidebar.number_input("Sub-Region Code", min_value=0, step=1)
hs_code = st.sidebar.number_input("HS Code", min_value=0, step=1)
unit = st.sidebar.number_input("Unit (Encoded)", min_value=0, step=1)
commodity = st.sidebar.number_input("Commodity (Encoded)", min_value=0, step=1)

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

# ------------------------------------------------------
# 4️⃣ PREDICTION SECTION
# ------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    model_choice = st.selectbox("🧠 Select Model", list(models.keys()))
    predict_btn = st.button("🚀 Predict Sub-Region")

with col2:
    st.markdown("### 🧾 Input Data Preview")
    st.dataframe(input_data)

if predict_btn:
    model = models.get(model_choice)
    if model:
        st.info(f"Predicting using **{model_choice}**...")
        try:
            scaler = StandardScaler()
            scaled_input = scaler.fit_transform(input_data)
            pred = model.predict(scaled_input)
            st.success(f"✅ Predicted Sub-Region: **{pred[0]}**")
        except Exception as e:
            st.error("⚠️ Error during prediction.")
            st.write(e)
    else:
        st.warning("Model not loaded or invalid.")

# ------------------------------------------------------
# 5️⃣ FOOTER
# ------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:white;'>📘 Developed by <b>Abhishek Wekhande</b> | Data from OGD Platform India</p>",
    unsafe_allow_html=True
)
