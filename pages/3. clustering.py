import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Retrain Clustering - Cuaca BMKG", layout="wide")

# ---------------------------------------------------------
# 1. SIDEBAR PARAMETER (PENGATURAN MODEL CLUSTERING)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Model Clustering")

# Input parameter K-Means di sidebar
k_clusters = st.sidebar.slider("Pilih Jumlah Cluster (K):", min_value=2, max_value=5, value=3, step=1)
random_state = st.sidebar.number_input("Random State:", value=42)

# TOMBOL UTAMA UNTUK MEMULAI PROSES CLUSTERING
btn_train = st.sidebar.button("🚀 Train Model")

# ---------------------------------------------------------
# 2. PERSIAPAN DATA UTAMA / CADANGAN
# ---------------------------------------------------------
raw_json_backup = {
    "data": [{
        "cuaca": [
            [{"local_datetime": "2026-10-31 23:00:00", "t": 23, "hu": 90, "ws": 6.8, "tcc": 43, "weather_desc": "Cerah"}],
            [
                {"local_datetime": "2026-11-01 02:00:00", "t": 23, "hu": 88, "ws": 8.1, "tcc": 5, "weather_desc": "Cerah"},
                {"local_datetime": "2026-11-01 05:00:00", "t": 22, "hu": 91, "ws": 6.2, "tcc": 1, "weather_desc": "Cerah"},
                {"local_datetime": "2026-11-01 08:00:00", "t": 28, "hu": 57, "ws": 4.4, "tcc": 65, "weather_desc": "Cerah Berawan"},
                {"local_datetime": "2026-11-01 11:00:00", "t": 32, "hu": 38, "ws": 22.2, "tcc": 65, "weather_desc": "Cerah Berawan"},
                {"local_datetime": "2026-11-01 14:00:00", "t": 30, "hu": 32, "ws": 24.1, "tcc": 65, "weather_desc": "Cerah Berawan"},
                {"local_datetime": "2026-11-01 17:00:00", "t": 29, "hu": 44, "ws": 22.2, "tcc": 65, "weather_desc": "Cerah Berawan"},
                {"local_datetime": "2026-11-01 20:00:00", "t": 24, "hu": 75, "ws": 14.8, "tcc": 65, "weather_desc": "Cerah Berawan"}
            ]
        ]
    }]
}

if "df_cuaca" in st.session_state:
    df_cluster = st.session_state["df_cuaca"].copy()
else:
    rows = []
    # PERBAIKAN: Menggunakan ["data"][0]["cuaca"]
    for group in raw_json_backup["data"][0]["cuaca"]:
        for item in group:
            rows.append({
                "Waktu": item["local_datetime"],
                "Suhu (°C)": item["t"], 
                "Kelembapan (%)": item["hu"], 
                "Kecepatan Angin (km/h)": item["ws"], 
                "Kondisi": item["weather_desc"]
            })
    df_cluster = pd.DataFrame(rows)
    df_cluster["Waktu"] = pd.to_datetime(df_cluster["Waktu"])

# ---------------------------------------------------------
# 3. KONDISI PEMICU TRAINING MODEL K-MEANS
# ---------------------------------------------------------
st.title("📌 CLUSTERING MACHINE LEARNING")
st.write("Aplikasi demo ML Clustering K-Means + Retrain dengan scikit-learn")

if btn_train:
    features = ["Suhu (°C)", "Kelembapan (%)", "Kecepatan Angin (km/h)"]
    X_cluster = df_cluster[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)
    
    st.session_state["df_cluster_terlatih"] = df_cluster
    st.session_state["k_aktif"] = k_clusters
    st.success(f"🎉 Model K-Means berhasil mengelompokkan data menjadi {k_clusters} Cluster!")

if "df_cluster_terlatih" not in st.session_state:
    st.warning("⚠️ Silakan klik 'Train Model' dulu di sidebar untuk memproses klasterisasi data dan menampilkan grafik.")
else:
    df_aktif = st.session_state["df_cluster_terlatih"]
    k_aktif = st.session_state["k_aktif"]
    
    st.markdown("---")
    st.subheader("📊 Training & Visualisasi Hasil Cluster")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        x="Suhu (°C)", y="Kelembapan (%)", hue="Cluster", style="Cluster", 
        palette="Set1", data=df_aktif, s=120, ax=ax
    )
    ax.set_title(f"Visualisasi Pembagian Sebaran Klaster K-Means (K={k_aktif})")
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📄 Tabel Anggota Hasil Pengelompokan Data")
    st.dataframe(df_aktif[["Waktu", "Suhu (°C)", "Kelembapan (%)", "Kondisi", "Cluster"]], use_container_width=True)
