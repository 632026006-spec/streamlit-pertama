import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Retrain Regresi - Cuaca BMKG", layout="wide")

# ---------------------------------------------------------
# 1. SIDEBAR PARAMETER (PENGATURAN MODEL)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Model Regresi")

# Input parameter model di sidebar
test_size = st.sidebar.slider("Test Size (Ukuran Data Uji):", 0.10, 0.50, 0.20, step=0.05)
random_state = st.sidebar.number_input("Random State:", value=42)

# TOMBOL UTAMA UNTUK MEMULAI TRAINING REGRESI
btn_train = st.sidebar.button("🚀 Train Model")

# ---------------------------------------------------------
# 2. PERSIAPAN DATA UTAMA / CADANGAN
# ---------------------------------------------------------
raw_json_backup = {
    "data": [{"cuaca": [
        [{"local_datetime": "2026-10-31 23:00:00", "t": 23, "hu": 90}],
        [
            {"local_datetime": "2026-11-01 02:00:00", "t": 23, "hu": 88},
            {"local_datetime": "2026-11-01 05:00:00", "t": 22, "hu": 91},
            {"local_datetime": "2026-11-01 08:00:00", "t": 28, "hu": 57},
            {"local_datetime": "2026-11-01 11:00:00", "t": 32, "hu": 38},
            {"local_datetime": "2026-11-01 14:00:00", "t": 30, "hu": 32},
            {"local_datetime": "2026-11-01 17:00:00", "t": 29, "hu": 44},
            {"local_datetime": "2026-11-01 20:00:00", "t": 24, "hu": 75}
        ]
    ]}]
}

if "df_cuaca" in st.session_state:
    df_cuaca = st.session_state["df_cuaca"]
else:
    rows = []
    for group in raw_json_backup["data"][0]["cuaca"]:
        for item in group:
            rows.append({"Suhu (°C)": item["t"], "Kelembapan (%)": item["hu"]})
    df_cuaca = pd.DataFrame(rows)

X = df_cuaca[["Kelembapan (%)"]]
y = df_cuaca["Suhu (°C)"]

# ---------------------------------------------------------
# 3. KONDISI PEMICU TRAINING MODEL & VISUALISASI
# ---------------------------------------------------------
st.title("📈 REGRESI MACHINE LEARNING")
st.write("Aplikasi demo ML Regresi Linear + Retrain dengan scikit-learn")

# Cek apakah user mengklik tombol Train Model
if btn_train:
    # Lakukan split data acak berdasarkan input slider sidebar
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Inisialisasi model dan latih model
    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)
    
    # Simpan model yang berhasil dilatih ke session_state
    st.session_state["model_reg_terlatih"] = model_reg
    
    # Hitung metrik evaluasi
    y_pred = model_reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    st.success("🎉 Model Regresi Linear berhasil dilatih ulang!")
    
    # Tampilkan Evaluasi Metrik
    st.subheader("📊 Evaluasi Model")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("R-Squared Score (R²)", f"{r2:.4f}")
    col_m2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")

# Cek apakah model regresi sudah pernah dilatih
if "model_reg_terlatih" not in st.session_state:
    st.warning("⚠️ Silakan klik 'Train Model' dulu di sidebar untuk memulai training dan memunculkan grafik.")
else:
    # Ambil model aktif dari session state
    model_aktif = st.session_state["model_reg_terlatih"]
    
    st.markdown("---")
    st.subheader("📈 Training & Visualisasi Regresi")
    
    # Buat dan render Grafik (Hanya muncul jika sudah di-train)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x="Kelembapan (%)", y="Suhu (°C)", data=df_cuaca, color="blue", label="Data Aktual BMKG", s=80, ax=ax)
    
    X_line = np.linspace(df_cuaca["Kelembapan (%)"].min(), df_cuaca["Kelembapan (%)"].max(), 100).reshape(-1, 1)
    X_line_df = pd.DataFrame(X_line, columns=["Kelembapan (%)"])
    y_line = model_aktif.predict(X_line_df)
    ax.plot(X_line, y_line, color="red", linewidth=2, label="Garis Tren Regresi")
    
    ax.set_xlabel("Kelembapan (%)")
    ax.set_ylabel("Suhu (°C)")
    ax.legend()
    st.pyplot(fig)
    
    # Form Input Prediksi Interaktif di bagian bawah
    st.markdown("---")
    st.subheader("💡 Uji Coba Input Prediksi")
    in_x = st.number_input("Masukkan Nilai Kelembapan Udara (%):", min_value=0.0, max_value=100.0, value=50.0)
    
    if st.button("Hitung Estimasi Suhu"):
        input_df = pd.DataFrame([[in_x]], columns=["Kelembapan (%)"])
        pred_y = model_aktif.predict(input_df)
        st.success(f"Hasil Prediksi Y: Perkiraan suhunya adalah **{pred_y[0]:.2f} °C**")
