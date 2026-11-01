import streamlit as st
import pandas as pd
import numpy as np
import requests
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression, make_classification, make_blobs

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Sistem Terpadu", page_icon="⚙️", layout="wide")

# --- KONEKSI MONGODB ---
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["dashboard_terpadu_db"]

collection_dataset = db["dataset_retrain"]
collection_comments = db["komentar_publik"]
collection_cuaca = db["log_cuaca_bmkg"]

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Modul:", [
    "1. Monitor Cuaca BMKG",
    "2. Regresi Linear",
    "3. Klasifikasi",
    "4. Clustering",
    "5. Input Data & Retraining (MongoDB)",
    "6. Komentar Publik (MongoDB)"
])

# --- DUMMY DATA BMKG 
SAMPLE_BMKG_JSON = {
    "lokasi": {
        "adm1": "53", "adm2": "53.02", "adm3": "53.02.01", "adm4": "53.02.01.1001",
        "provinsi": "Nusa Tenggara Timur", "kotkab": "Timor Tengah Selatan",
        "kecamatan": "Kota Soe", "desa": "SoE",
        "lon": 124.2718125428, "lat": -9.8651820198, "timezone": "Asia/Makassar"
    },
    "data": [
        {
            "lokasi": {
                "adm1": "53", "adm2": "53.02", "adm3": "53.02.01", "adm4": "53.02.01.1001",
                "provinsi": "Nusa Tenggara Timur", "kotkab": "Timor Tengah Selatan",
                "kecamatan": "Kota Soe", "desa": "SoE",
                "lon": 124.2718125428, "lat": -9.8651820198, "timezone": "+0800", "type": "adm4"
            },
            "cuaca": [
                [
                    {
                        "datetime": "2026-10-31T15:00:00Z", "t": 23, "tcc": 43, "tp": 0, "weather": 1,
                        "weather_desc": "Cerah", "weather_desc_en": "Sunny", "wd_deg": 55, "wd": "NE",
                        "wd_to": "SW", "ws": 6.8, "hu": 90, "vs": None, "vs_text": None, "time_index": "1-2",
                        "analysis_date": "2026-10-31T12:00:00", "image": "https://api-apps.bmkg.go.id/storage/icon/cuaca/cerah-pm.svg",
                        "utc_datetime": "2026-10-31 15:00:00", "local_datetime": "2026-10-31 23:00:00"
                    }
                ],
                [
                    {
                        "datetime": "2026-10-31T18:00:00Z", "t": 23, "tcc": 5, "tp": 0, "weather": 0,
                        "weather_desc": "Cerah", "weather_desc_en": "Sunny", "wd_deg": 45, "wd": "N",
                        "wd_to": "S", "ws": 8.1, "hu": 88, "vs": None, "vs_text": None, "time_index": "4-5",
                        "analysis_date": "2026-10-31T12:00:00", "image": "https://api-apps.bmkg.go.id/storage/icon/cuaca/cerah-pm.svg",
                        "utc_datetime": "2026-10-31 18:00:00", "local_datetime": "2026-11-01 02:00:00"
                    },
                    {
                        "datetime": "2026-10-31T21:00:00Z", "t": 22, "tcc": 1, "tp": 0, "weather": 0,
                        "weather_desc": "Cerah", "weather_desc_en": "Sunny", "wd_deg": 42, "wd": "N",
                        "wd_to": "S", "ws": 6.2, "hu": 91, "vs": None, "vs_text": None, "time_index": "7-8",
                        "analysis_date": "2026-10-31T12:00:00", "image": "https://api-apps.bmkg.go.id/storage/icon/cuaca/cerah-pm.svg",
                        "utc_datetime": "2026-10-31 21:00:00", "local_datetime": "2026-11-01 05:00:00"}]]}]}

def parse_bmkg_data(raw_json):
    lokasi = raw_json.get("lokasi", {})
    flat_cuaca = []
    data_list = raw_json.get("data", [])
    if data_list:
        cuaca_groups = data_list[0].get("cuaca", [])
        for group in cuaca_groups:
            for item in group:
                flat_cuaca.append(item)
    df = pd.DataFrame(flat_cuaca)
    if not df.empty:
        df['t'] = pd.to_numeric(df['t'])
        df['hu'] = pd.to_numeric(df['hu'])
        df['ws'] = pd.to_numeric(df['ws'])
        df['local_datetime'] = pd.to_datetime(df['local_datetime'])
    return lokasi, df

# ==========================================
# MENU 1: MONITOR CUACA BMKG & SIMPAN MONGODB
# ==========================================
if menu == "1. Monitor Cuaca BMKG":
    st.header("🌤️ Monitor Cuaca Realtime / JSON BMKG")
    
    lokasi, df_cuaca = parse_bmkg_data(SAMPLE_BMKG_JSON)
    st.subheader(f"📍 {lokasi.get('desa', '-')}, {lokasi.get('kecamatan', '-')}, {lokasi.get('kotkab', '-')}")

    if not df_cuaca.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Suhu Saat Ini", f"{df_cuaca.iloc[0]['t']} °C")
        col2.metric("Kelembapan", f"{df_cuaca.iloc[0]['hu']} %")
        col3.metric("Kondisi", df_cuaca.iloc[0]['weather_desc'])

        st.line_chart(df_cuaca.set_index("local_datetime")[["t", "hu"]])

        if st.button("💾 Simpan Data Cuaca Ini ke MongoDB"):
            collection_cuaca.insert_one(SAMPLE_BMKG_JSON)
            st.success("Berhasil menyimpan payload JSON cuaca ke MongoDB!")

# ==========================================
# MENU 2: REGRESI LINEAR
# ==========================================
elif menu == "2. Regresi Linear":
    st.header("📈 Model Regresi Linear")
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
    model = LinearRegression().fit(X, y)
    
    val = st.number_input("Input Fitur X:", value=2.0)
    if st.button("Prediksi"):
        res = model.predict([[val]])[0]
        st.success(f"Hasil Prediksi Y: {res:.2f}")

# ==========================================
# MENU 3: KLASIFIKASI
# ==========================================
elif menu == "3. Klasifikasi":
    st.header("🏷️ Model Klasifikasi (Decision Tree)")
    X, y = make_classification(n_samples=100, n_features=2, random_state=42)
    model = DecisionTreeClassifier().fit(X, y)
    
    f1 = st.number_input("Fitur 1:", value=0.5)
    f2 = st.number_input("Fitur 2:", value=-0.5)
    if st.button("Klasifikasikan"):
        res = model.predict([[f1, f2]])[0]
        st.info(f"Hasil Prediksi Kelas: Kelas {res}")

# ==========================================
# MENU 4: CLUSTERING
# ==========================================
elif menu == "4. Clustering":
    st.header("🔍 Model Clustering (K-Means)")
    X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
    k = st.slider("Jumlah Cluster (K):", 2, 5, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    
    df = pd.DataFrame(X, columns=["Fitur 1", "Fitur 2"])
    df["Cluster"] = labels
    st.dataframe(df.head(10))

# ==========================================
# MENU 5: TAMBAH DATA & RETRAINING (MONGODB)
# ==========================================
elif menu == "5. Input Data & Retraining (MongoDB)":
    st.header("🔄 Tambah Data & Retrain Model")
    
    with st.form("form_data"):
        x_val = st.number_input("Nilai X:", value=1.0)
        y_val = st.number_input("Nilai Y:", value=5.0)
        if st.form_submit_button("Simpan ke MongoDB"):
            collection_dataset.insert_one({"x": x_val, "y": y_val})
            st.success("Data berhasil tersimpan di MongoDB!")

    stored = list(collection_dataset.find({}, {"_id": 0}))
    if stored:
        df_mongo = pd.DataFrame(stored)
        st.dataframe(df_mongo)
        
        if st.button("Latih Ulang Model"):
            if len(df_mongo) >= 2:
                model_new = LinearRegression()
                model_new.fit(df_mongo[["x"]].values, df_mongo["y"].values)
                st.success(f"Model berhasil di-retrain! Slope baru: {model_new.coef_[0]:.4f}")
            else:
                st.warning("Butuh minimal 2 data di MongoDB untuk retrain.")

# ==========================================
# MENU 6: KOMENTAR PUBLIK (MONGODB)
# ==========================================
elif menu == "6. Komentar Publik (MongoDB)":
    st.header("💬 Input & Tampilkan Komentar Publik")
    
    with st.form("form_komen"):
        nama = st.text_input("Nama:")
        komen = st.text_area("Komentar:")
        if st.form_submit_button("Kirim"):
            if nama and komen:
                collection_comments.insert_one({"nama": nama, "komentar": komen})
                st.success("Komentar tersimpan!")
    
    st.subheader("Daftar Komentar:")
    all_comments = list(collection_comments.find({}, {"_id": 0}))
    for c in reversed(all_comments):
        st.write(f"**{c['nama']}**: {c['komentar']}")
        st.divider()