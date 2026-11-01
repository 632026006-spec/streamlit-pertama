import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Retrain Klasifikasi - Cuaca BMKG", layout="wide")

# ---------------------------------------------------------
# 1. SIDEBAR PARAMETER (PENGATURAN MODEL)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Model")

# Slider untuk Test Size seperti di gambar Anda
test_size = st.sidebar.slider("Test Size (Ukuran Data Uji):", 0.10, 0.50, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random State:", value=42)

# TOMBOL UTAMA UNTUK MEMULAI TRAINING SPERTI DI GAMBAR
btn_train = st.sidebar.button("🚀 Train Model")

# ---------------------------------------------------------
# 2. PERSIAPAN DATA UTAMA / CADANGAN
# ---------------------------------------------------------
raw_json_backup = {
    "data": [{"cuaca": [
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
    ]}]
}

if "df_cuaca" in st.session_state:
    df_cuaca = st.session_state["df_cuaca"]
else:
    rows = []
    for group in raw_json_backup["data"][0]["cuaca"]:
        for item in group:
            rows.append({
                "Suhu (°C)": item["t"], "Kelembapan (%)": item["hu"], 
                "Kecepatan Angin (km/h)": item["ws"], "Tutupan Awan (%)": item["tcc"], "Kondisi": item["weather_desc"]
            })
    df_cuaca = pd.DataFrame(rows)

# Separasi Fitur dan Target
X_cls = df_cuaca[["Suhu (°C)", "Kelembapan (%)", "Kecepatan Angin (km/h)", "Tutupan Awan (%)"]]
y_cls = df_cuaca["Kondisi"]

# ---------------------------------------------------------
# 3. KONDISI PEMICU TRAINING MODEL
# ---------------------------------------------------------
st.title("🏷️ REGRESI / KLASIFIKASI MACHINE LEARNING")
st.write("Aplikasi demo ML + MongoDB CRUD + Retrain dengan scikit-learn")

# Cek apakah user sudah mengklik tombol Train Model
if btn_train:
    # Lakukan split data acak berdasarkan input slider sidebar
    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=test_size, random_state=random_state)
    
    # Inisialisasi model dan fit
    model_cls = DecisionTreeClassifier(random_state=random_state)
    model_cls.fit(X_train, y_train)
    
    # Simpan model yang sudah dilatih ke memori session_state agar tidak hilang saat tombol prediksi ditekan
    st.session_state["model_terlatih"] = model_cls
    
    # Hitung akurasi data uji
    y_pred = model_cls.predict(X_test)
    score_akurasi = accuracy_score(y_test, y_pred) * 100
    
    # Tampilkan notifikasi hijau tanda sukses latih model
    st.success("🎉 Model berhasil dilatih ulang!")
    st.metric("Akurasi Evaluasi Model", f"{score_akurasi:.1f} %")

# Tampilkan peringatan kuning jika tombol di sidebar belum ditekan sama sekali
if "model_terlatih" not in st.session_state:
    st.warning("⚠️ Silakan klik 'Train Model' dulu di sidebar untuk memulai training.")
else:
    # Jika model sudah dilatih, buka form simulator input di bawahnya
    st.markdown("---")
    st.subheader("🎛️ Simulator Prediksi Cuaca Real-Time (Model Aktif)")
    
    col1, col2 = st.columns(2)
    with col1:
        in_t = st.number_input("Suhu (°C):", value=25.0)
        in_hu = st.number_input("Kelembapan (%):", value=70.0)
    with col2:
        in_ws = st.number_input("Kecepatan Angin (km/h):", value=10.0)
        in_tcc = st.number_input("Tutupan Awan (%):", value=50.0)
        
    if st.button("Prediksi Kondisi Cuaca"):
        input_data = pd.DataFrame([{
            "Suhu (°C)": in_t, "Kelembapan (%)": in_hu, 
            "Kecepatan Angin (km/h)": in_ws, "Tutupan Awan (%)": in_tcc
        }])
        
        # Panggil kembali model dari session state
        model_aktif = st.session_state["model_terlatih"]
        pred_kondisi = model_aktif.predict(input_data)
        st.info(f"Hasil Klasifikasi Cuaca: **{pred_kondisi[0]}**")
