import streamlit as st
import numpy as np
import pandas as pd
import joblib

# -------------------------
# LOAD MODEL DAN SCALER
# -------------------------

@st.cache_resource
def load_models():
    voting = joblib.load("voting_model.joblib")
    rf = joblib.load("rf_model.joblib")
    svc = joblib.load("svc_model.joblib")
    scaler = joblib.load("scaler.joblib")

    feature_names = pd.read_csv("feature_names.csv", header=None)[0].tolist()

    return voting, rf, svc, scaler, feature_names


voting_model, rf_model, svc_model, scaler, feature_names = load_models()

# -------------------------
# UI APLIKASI
# -------------------------

st.set_page_config(page_title="Cancer Prediction - Voting Ensemble", layout="wide")

st.title("🔬 Breast Cancer Prediction (Ensemble Model)")
st.write("""
Aplikasi cerdas ini menggunakan **dua algoritma Machine Learning**:
- Random Forest  
- Support Vector Machine  
- Digabungkan menggunakan **Voting Classifier (Soft Voting)**

Model dilatih di Google Colab menggunakan dataset **Breast Cancer Wisconsin (Diagnostic)**.

Akurasi model ensemble umumnya **95%–98%**.
""")

# -------------------------
# INPUT MODE (FORM)
# -------------------------

st.header("📝 Input Data")

input_mode = st.radio("Pilih metode input:", ["Form Manual", "Upload CSV"])

# -------------------------
# INPUT MANUAL
# -------------------------

if input_mode == "Form Manual":
    st.write("Masukkan nilai fitur berikut:")

    cols = st.columns(3)
    user_data = []

    for i, feature in enumerate(feature_names):
        val = cols[i % 3].number_input(feature, value=0.0)
        user_data.append(val)

    if st.button("Prediksi"):
        arr = np.array(user_data).reshape(1, -1)
        arr_scaled = scaler.transform(arr)

        pred = voting_model.predict(arr_scaled)[0]
        proba = voting_model.predict_proba(arr_scaled)[0]

        label = "Malignant (Kanker)" if pred == 1 else "Benign (Tidak Kanker)"

        st.subheader("🎯 Hasil Prediksi")
        st.write(f"**{label}**")
        st.write("Probabilitas:", proba)

# -------------------------
# INPUT CSV
# -------------------------

else:
    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Preview data:")
        st.dataframe(df.head())

        try:
            arr_scaled = scaler.transform(df.values)

            pred = voting_model.predict(arr_scaled)
            df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in pred]

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(df)

            st.download_button(
                "Download Hasil",
                df.to_csv(index=False),
                file_name="prediction_result.csv"
            )
        except:
            st.error("Pastikan kolom CSV sesuai dengan fitur model!")


st.markdown("---")
st.caption("Dibuat untuk tugas Machine Learning - Streamlit + Ensemble Model.")
