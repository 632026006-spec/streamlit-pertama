# ============================================================
# 🤖 ML STUDIO — DASHBOARD MACHINE LEARNING
# Dibuat menggunakan Streamlit dan Scikit-learn
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Model klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Model berbasis pohon
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Metrik evaluasi
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,)


# ============================================================
# ⚙️ KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="ML Studio 🤖",
    page_icon="🤖",
    layout="wide",)


# ============================================================
# 🎨 DESAIN DASHBOARD
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;}

        .hero {
            padding: 25px;
            border-radius: 18px;
            background: linear-gradient(120deg, #173b57, #287d8e);
            color: white;
            margin-bottom: 20px;}

        .hero h1 {
            color: white;
            margin-bottom: 8px;}

        .hero p {
            color: #e6f6fb;
            font-size: 16px;}

        div[data-testid="stMetric"] {
            background: rgba(120, 150, 170, 0.10);
            padding: 15px;
            border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,)


# ============================================================
# 📦 MEMUAT DATASET TOY DARI SCIKIT-LEARN
# ============================================================

@st.cache_data
def get_dataset(dataset_name):
    """Memuat dataset dan menyimpan hasilnya dalam cache."""

    if dataset_name == "Iris":
        data = datasets.load_iris()

    elif dataset_name == "Wine":
        data = datasets.load_wine()

    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()

    elif dataset_name == "Digits":
        data = datasets.load_digits()

    elif dataset_name == "Diabetes":
        data = datasets.load_diabetes()

    else:
        raise ValueError("Dataset tidak dikenal.")

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names)

    y = pd.Series(data.target, name="target")

    class_names = getattr(data, "target_names", None)

    if class_names is not None:
        class_names = list(class_names)

    return X, y, class_names


# ============================================================
# 💾 SESSION STATE
# Menyimpan hasil training agar tidak hilang saat rerun
# ============================================================

default_state = {
    "trained": False,
    "model": None,
    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "problem_type": None,
    "dataset_name": None,
    "model_name": None,
    "class_names": None,}

for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 🧭 SIDEBAR — PENGATURAN MACHINE LEARNING
# ============================================================

st.sidebar.title("🛠️ Pengaturan ML")
st.sidebar.caption("Atur eksperimen Machine Learning kamu.")

problem_type = st.sidebar.selectbox(
    "Pilih Jenis Masalah",
    ["Klasifikasi", "Regresi"])


# Pilihan dataset dan model berdasarkan jenis masalah
if problem_type == "Klasifikasi":

    dataset_name = st.sidebar.selectbox(
        "Pilih Dataset",
        [
            "Iris",
            "Wine",
            "Breast Cancer",
            "Digits",])

    model_name = st.sidebar.selectbox(
        "Pilih Model Klasifikasi",
        [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "SVM",])

else:

    dataset_name = st.sidebar.selectbox(
        "Pilih Dataset",
        ["Diabetes"])

    model_name = st.sidebar.selectbox(
        "Pilih Model Regresi",
        [
            "Linear Regression",
            "Ridge",
            "Lasso",
            "Decision Tree",
            "Random Forest",])


# Mengatur proporsi data tes
test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05)


# Tombol untuk memulai training
train_button = st.sidebar.button(
    "🚀 Train Model",
    type="primary",
    use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Dibangun dengan Streamlit, Pandas, NumPy, "
    "Matplotlib, Seaborn, dan Scikit-learn.")


# ============================================================
# 🧠 PROSES TRAINING MODEL
# ============================================================

if train_button:

    try:
        with st.spinner("Sedang melatih model..."):

            # Memuat dataset
            X, y, class_names = get_dataset(dataset_name)

            # Membagi data training dan testing
            stratify_data = (
                y if problem_type == "Klasifikasi"
                else None)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify_data)

            # ------------------------------------------------
            # 🎯 MEMILIH MODEL KLASIFIKASI
            # ------------------------------------------------

            if problem_type == "Klasifikasi":

                if model_name == "Logistic Regression":

                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(max_iter=2000))

                elif model_name == "KNN":

                    model = make_pipeline(
                        StandardScaler(),
                        KNeighborsClassifier())

                elif model_name == "Decision Tree":

                    model = DecisionTreeClassifier(
                        random_state=42)

                elif model_name == "Random Forest":

                    model = RandomForestClassifier(
                        random_state=42)

                else:

                    model = make_pipeline(
                        StandardScaler(),
                        SVC())

            # ------------------------------------------------
            # 📈 MEMILIH MODEL REGRESI
            # ------------------------------------------------

            else:

                if model_name == "Linear Regression":

                    model = LinearRegression()

                elif model_name == "Ridge":

                    model = make_pipeline(
                        StandardScaler(),
                        Ridge())

                elif model_name == "Lasso":

                    model = make_pipeline(
                        StandardScaler(),
                        Lasso(max_iter=5000))

                elif model_name == "Decision Tree":

                    model = DecisionTreeRegressor(
                        random_state=42)

                else:

                    model = RandomForestRegressor(
                        random_state=42)

            # Melatih model
            model.fit(X_train, y_train)

            # Melakukan prediksi pada data tes
            y_pred = model.predict(X_test)

            # Menyimpan hasil ke session state
            st.session_state.trained = True
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.problem_type = problem_type
            st.session_state.dataset_name = dataset_name
            st.session_state.model_name = model_name
            st.session_state.class_names = class_names

        st.success("Model berhasil dilatih! 🎉")

    except Exception as e:

        st.error(f"Terjadi kesalahan saat training: {e}")


# ============================================================
# 🖥️ TAMPILAN UTAMA DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 ML Studio</h1>
        <p>
            Dashboard Machine Learning interaktif untuk
            melatih, memvisualisasikan, dan mengevaluasi model.
        </p>
    </div>
    """,
    unsafe_allow_html=True)


# Informasi pengaturan saat ini
col1, col2, col3 = st.columns(3)

col1.metric("🧩 Jenis Masalah", problem_type)
col2.metric("🗂️ Dataset", dataset_name)
col3.metric("⚙️ Model", model_name)


# Tampilan awal sebelum training
if not st.session_state.trained:

    st.info(
        "👈 Pilih pengaturan pada sidebar, "
        "kemudian klik **🚀 Train Model**.")

    st.markdown(
        """
        ### ✨ Fitur ML Studio

        - 📈 Regresi dan klasifikasi.
        - 🧪 Dataset bawaan Scikit-learn.
        - 📊 Visualisasi hasil prediksi.
        - 📋 Metrik evaluasi model.
        - 💾 Penyimpanan hasil menggunakan session state.
        """)


# ============================================================
# 📊 TAB VISUALISASI DAN EVALUASI
# ============================================================

else:

    st.caption(
        f"Hasil terakhir: {st.session_state.dataset_name} · "
        f"{st.session_state.model_name} · "
        f"{st.session_state.problem_type}"
    )

    tab1, tab2 = st.tabs(
        ["📊 Visualisasi", "📋 Evaluasi"]
    )

    # ========================================================
    # TAB 1 — VISUALISASI
    # ========================================================

    with tab1:

        st.subheader("📈 Visualisasi Hasil Model")

        # Grafik regresi: aktual dibandingkan prediksi
        if st.session_state.problem_type == "Regresi":

            y_true = np.asarray(st.session_state.y_test)
            y_pred = np.asarray(st.session_state.y_pred)

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.scatter(
                y_true,
                y_pred,
                alpha=0.7,
                edgecolor="white")

            min_value = min(y_true.min(), y_pred.min())
            max_value = max(y_true.max(), y_pred.max())

            ax.plot(
                [min_value, max_value],
                [min_value, max_value],
                "r--",
                label="Prediksi ideal")

            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            ax.grid(alpha=0.2)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # Confusion Matrix untuk klasifikasi
        else:

            y_true = np.asarray(st.session_state.y_test)
            y_pred = np.asarray(st.session_state.y_pred)

            labels = np.unique(
                np.concatenate([y_true, y_pred]))

            class_names = st.session_state.class_names

            if class_names is not None:
                tick_labels = [
                    str(class_names[int(label)])
                    for label in labels]
            else:
                tick_labels = [str(label) for label in labels]

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=labels)

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                linewidths=0.5,
                ax=ax)

            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            ax.set_title("Confusion Matrix")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ========================================================
    # TAB 2 — EVALUASI MODEL
    # ========================================================

    with tab2:

        st.subheader("📋 Metrik Evaluasi")

        y_true = st.session_state.y_test
        y_pred = st.session_state.y_pred

        # ----------------------------------------------------
        # METRIK REGRESI
        # ----------------------------------------------------

        if st.session_state.problem_type == "Regresi":

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("MAE", f"{mae:.4f}")
            c2.metric("MSE", f"{mse:.4f}")
            c3.metric("RMSE", f"{rmse:.4f}")
            c4.metric("R² Score", f"{r2:.4f}")

            st.caption(
                "MAE, MSE, dan RMSE mengukur kesalahan prediksi. "
                "Nilai yang lebih rendah umumnya lebih baik. "
                "R² yang mendekati 1 umumnya menunjukkan "
                "model menjelaskan variasi target dengan baik."
            )

        # ----------------------------------------------------
        # METRIK KLASIFIKASI
        # ----------------------------------------------------

        else:

            accuracy = accuracy_score(y_true, y_pred)

            precision = precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0)

            recall = recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0)

            f1 = f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0 )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy:.4f}")
            c2.metric("Precision", f"{precision:.4f}")
            c3.metric("Recall", f"{recall:.4f}")
            c4.metric("F1 Score", f"{f1:.4f}")

            # Classification report
            st.markdown("### 🧾 Classification Report")

            class_names = st.session_state.class_names

            report = classification_report(
                y_true,
                y_pred,
                target_names=(
                    [str(name) for name in class_names]
                    if class_names is not None
                    else None
                ),
                zero_division=0)

            st.code(report, language="text")


# ============================================================
# 📝 FOOTER
# ============================================================

st.divider()

st.caption(
    "🤖 ML Studio | Dibangun dengan Streamlit dan Scikit-learn"
)