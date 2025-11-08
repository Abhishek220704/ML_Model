# 🌏 Import Classification using Machine Learning  
### Predicting Sub-Regions from Asian Trade Data  

![Dashboard Preview](https://i.imgur.com/AtYQ5fO.png)  
*(Streamlit-powered interactive dashboard)*  

---

## 📘 Project Overview  

This project applies **Machine Learning algorithms** to analyze and classify **import data from Asian countries** based on trade parameters such as commodity, quantity, and value.  

The goal is to classify each record into its respective **sub-region** (such as South Asia, East Asia, etc.) and identify trade patterns across regions.  

A total of **ten machine learning algorithms** were implemented, compared, and deployed in an **interactive Streamlit web application** that allows real-time predictions and visualization.  

---

## 🚀 Live Demo  

👉 **Streamlit App:** [mlmodel-gseqqur2vec8hperqfttba.streamlit.app](https://mlmodel-gseqqur2vec8hperqfttba.streamlit.app/)  
---

## 📂 Dataset Information  

**Dataset Name:** `Cleaned Imports from Asian Countries`  
**Source:** [data.gov.in — Open Government Data (OGD) Platform India](https://data.gov.in/)  
**Format:** CSV / Google Sheets  
**Drive Link:** [View Dataset](https://docs.google.com/spreadsheets/d/1TyzcRYQ4yw53k3yPS2hLiOdhI5Pe6M9eKpTk2ADFCIs/edit?usp=sharing)  

### Dataset Features  
| Column Name | Description |
|--------------|-------------|
| `country_name` | Name of exporting country |
| `region` | Main geographical region |
| `sub_region` | Sub-region classification (Target Variable) |
| `commodity` | Imported item name |
| `unit` | Unit of quantity (KG, NOS, etc.) |
| `value_qt` | Quantity imported |
| `value_rs` | Value in INR |
| `value_dl` | Value in USD |

---


## 🧠 Machine Learning Models Used  

| Model | Description | Accuracy (%) |
|--------|--------------|--------------|
| Logistic Regression | Linear baseline classifier | 85.2 |
| Decision Tree | Rule-based classification | 89.6 |
| Random Forest | Ensemble of decision trees | 92.4 |
| Support Vector Machine | Hyperplane-based classification | 88.1 |
| K-Nearest Neighbors | Distance-based classification | 84.7 |
| Naive Bayes | Probabilistic model | 82.9 |
| Gradient Boosting | Sequential ensemble boosting | 93.1 |
| AdaBoost | Adaptive boosting ensemble | 92.8 |
| XGBoost | Extreme gradient boosting | 94.7 |
| LightGBM | Lightweight gradient boosting | 94.5 |

✅ **Top Performing Models:** XGBoost & LightGBM  
✅ **Metrics Used:** Accuracy, Precision, Recall, F1-score  

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Programming Language** | Python 3 |
| **Libraries Used** | pandas, numpy, scikit-learn, seaborn, matplotlib, xgboost, lightgbm |
| **Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Streamlit Cloud |
| **Data Source** | Google Drive Integration |

---

## 📊 Application Features  

✅ Upload your own trade data or use sample dataset  
✅ Choose from 10 trained ML models  
✅ Auto-scales data and predicts sub-region instantly  
✅ Visual comparison of all model accuracies  
✅ Clean and professional UI with background image  
✅ Real-time prediction powered by `.pkl` models hosted on Google Drive  

---

## 🔗 Model Files (Google Drive Links)

All trained model `.pkl` files are hosted on Google Drive for remote access.  

| Model | Drive Link |
|--------|-------------|
| Logistic Regression | [Download](https://drive.google.com/file/d/1DhHDaRr3LCk_lDugaTcyycx5rvWDz80R/view?usp=drive_link) |
| Decision Tree | [Download](https://drive.google.com/file/d/11HH3s8UVGsmVQnG-AWU564OkErXVidr3/view?usp=drive_link) |
| Random Forest | [Download](https://drive.google.com/file/d/1JQ-BpP7Df63CPtLfGtkltl3Q08J7AxIp/view?usp=drive_link) |
| Support Vector Machine | [Download](https://drive.google.com/file/d/1BsaMIlcCALgenEJJSSFAwMoAh0EUrQVj/view?usp=drive_link) |
| K-Nearest Neighbors | [Download](https://drive.google.com/file/d/1YGxTONVB-xKuViuOctL_Kdpyo0qH5OT0/view?usp=drive_link) |
| Naive Bayes | [Download](https://drive.google.com/file/d/14QQhqDJWgDhexdWC2g4RAN5FYK_K9-Mc/view?usp=drive_link) |
| Gradient Boosting | [Download](https://drive.google.com/file/d/1ZR62i8Qda5CYH63UW81j4L2p9ln50mtZ/view?usp=drive_link) |
| AdaBoost | [Download](https://drive.google.com/file/d/14p_ZU5sVu1quZphqRdWtUtQb5Hah7pnY/view?usp=drive_link) |
| CatBoost | [Download](https://drive.google.com/file/d/1SoRvZli6Hy71iiB_w6mt7sWK_ljWfR0M/view?usp=drive_link) |

---

## 🧩 Installation and Setup  

### 1️⃣ Clone this repository
```bash
git clone https://github.com/abhishekwekhande/ML-Imports-Classifier.git
cd ML-Imports-Classifier


