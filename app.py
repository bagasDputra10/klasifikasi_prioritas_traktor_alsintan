import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# CUSTOM TRANSFORMER
# ======================================================

class MinMaxComposite(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        X = np.asarray(X, dtype=float)

        self.min_ = X.min(axis=0)

        rng = X.max(axis=0) - self.min_

        self.range_ = np.where(
            rng == 0,
            1.0,
            rng
        )

        return self

    def transform(self, X):

        X = np.asarray(X, dtype=float)

        Z = np.clip(
            (X - self.min_) / self.range_,
            0,
            1
        )

        return (
            (
                Z[:, 0]
                + Z[:, 1]
                + (1 - Z[:, 2])
            ) / 3.0
        ).reshape(-1, 1)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Klasifikasi Prioritas Bantuan Traktor",
    page_icon="🚜",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    with open(
        "./models/composite_tree_best/model.pkl",
        "rb"
    ) as f:
        model = pickle.load(f)

    with open(
        "./models/composite_tree_best/label_encoder.pkl",
        "rb"
    ) as f:
        label_encoder = pickle.load(f)

    with open(
        "./models/composite_tree_best/metadata.pkl",
        "rb"
    ) as f:
        metadata = pickle.load(f)

    return model, label_encoder, metadata


model, label_encoder, metadata = load_model()

KABUPATEN_KOTA = sorted([
    "Bangkalan",
    "Banyuwangi",
    "Blitar",
    "Bojonegoro",
    "Bondowoso",
    "Gresik",
    "Jember",
    "Jombang",
    "Kediri",
    "Lamongan",
    "Lumajang",
    "Madiun",
    "Magetan",
    "Malang",
    "Mojokerto",
    "Nganjuk",
    "Ngawi",
    "Pacitan",
    "Pamekasan",
    "Pasuruan",
    "Ponorogo",
    "Probolinggo",
    "Sampang",
    "Sidoarjo",
    "Situbondo",
    "Sumenep",
    "Trenggalek",
    "Tuban",
    "Tulungagung",
    "Kota Batu",
    "Kota Blitar",
    "Kota Kediri",
    "Kota Madiun",
    "Kota Malang",
    "Kota Mojokerto",
    "Kota Pasuruan",
    "Kota Probolinggo",
    "Kota Surabaya"
])

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🚜 Menu")

menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Information Project",
        "Simulasi Klasifikasi"
    ]
)

# ======================================================
# INFORMATION PROJECT
# ======================================================

if menu == "Information Project":

    st.title("🚜 Sistem Klasifikasi Prioritas Bantuan Traktor")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Tentang Sistem",
            "Dataset",
            "Metodologi",
            "Model Terbaik"
        ]
    )

    with tab1:

        st.header("Tentang Sistem")

        st.markdown("""
        Aplikasi ini menggunakan model Machine Learning terbaik
        yang telah dilatih untuk memprediksi dan memetakan wilayah prioritas distribusi traktor
        roda dua menjadi klasifikasi prioritas

        - 🔴 Tinggi
        - 🟡 Sedang
        - 🟢 Rendah

        berdasarkan profil teknis wilayah.
        """)

    with tab2:

        st.header("Dataset")

        st.markdown("""
        Dataset berasal dari data Kabupaten/Kota
        di Jawa Timur periode 2020–2025.
        """)

    with tab3:

        st.header("Metodologi")

        st.markdown("""
        ### Fitur Model

        1. Luas Baku Sawah
        2. Intensitas Tanam
        3. Rasio Kecukupan Traktor

        ### Intensitas Tanam

        Luas Panen ÷ Luas Baku Sawah

        ### Rasio Kecukupan

        Ketersediaan Traktor ÷ Kebutuhan Traktor

        ### Composite Score

        mean(
            minmax(luas_baku_sawah),
            minmax(intensitas_tanam),
            1-minmax(rasio_kecukupan)
        )

        ### Model

        Decision Tree (Depth = 2)
        """)

    with tab4:

        st.header("Model Terbaik")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Macro F1 CV",
            f"{metadata['cv_macro_f1_score']:.3f}"
        )

        col2.metric(
            "Held-Out Macro F1",
            f"{metadata['heldout_macro_f1_mean']:.3f}"
        )

        col3.metric(
            "Accuracy CV",
            f"{metadata['cv_accuracy']:.3f}"
        )

        st.divider()

        st.subheader("Fitur")

        st.write(metadata["features"])

        st.subheader("Threshold Composite Score")

        st.write(metadata["composite_thresholds"])

        st.subheader("Aturan Decision Tree")

        st.code(
            metadata["tree_rules"],
            language="text"
        )

# ======================================================
# SIMULASI
# ======================================================

elif menu == "Simulasi Klasifikasi":

    AUTO = "<span style='color:red'>*</span>"
    st.title("🔍 Simulasi Klasifikasi Prioritas Bantuan")

    kabupaten_kota = st.selectbox(
    "Kabupaten/Kota",
    KABUPATEN_KOTA
    )

    st.divider()

    st.markdown("""
    Data Pertanian
    """)

    st.markdown("### INPUT DATA PERTANIAN")

    # Baris 1
    a1, a2 = st.columns(2)

    with a1:
        luas_baku = st.number_input(
            "Luas Baku Sawah (Ha)",
            min_value=1.0,
            value=50000.0,
        )

    with a2:
        produksi_padi = st.number_input(
            "Produksi Padi (Ton)",
            min_value=1.0,
            value=300.0
        )

    # Baris 2
    b1, b2 = st.columns(2)

    with b1:
        luas_panen = st.number_input(
            "Luas Panen (Ha)",
            min_value=1.0,
            value=60000.0
        )

    with b2:
        produktivitas_padi = (produksi_padi / luas_baku) / 10

        st.text_input(
            "Produktivitas Padi (Ku/Ha)",
            value=f"{produktivitas_padi:.3f}",
            disabled=True
        )

    # Baris 3
    c1, c2 = st.columns(2)

    with c1:
        indeks_pertanaman = luas_panen / luas_baku

        st.text_input(
            "Indeks Pertanaman(IP)",
            value=f"{indeks_pertanaman:.3f}",
            disabled=True
        )

    with c2:
        st.empty()  # kolom kosong

    st.info(
        """
        - Indeks Pertanaman diperoleh dari Luas Panen (Ha) / Luas Baku Sawah (Ha)\n
        - Produktivitas Padi diperoleh dari (Produksi Padi/Luas Baku Sawah) / 10)
        """
    )

    st.divider()

    st.markdown("### INPUT DATA ALSINTAN")

    # ==========================================
    # PERHITUNGAN OTOMATIS
    # ==========================================

    intensitas_tanam = luas_panen / luas_baku

    kebutuhan = (luas_baku / 25) / intensitas_tanam

    kekurangan = max(
        kebutuhan - ketersediaan if 'ketersediaan' in locals() else kebutuhan,
        0
    )

    # ==========================================
    # BARIS 1
    # ==========================================

    d1, d2 = st.columns(2)

    with d1:

        st.text_input(
            "Kebutuhan Traktor Roda Dua (Unit)",
            value=f"{kebutuhan:.2f}",
            disabled=True
        )

    with d2:

        ketersediaan = st.number_input(
            "Ketersediaan Traktor Roda Dua (Unit)",
            min_value=0.0,
            value=300.0
        )

  

    # ==========================================
    # HITUNG ULANG
    # ==========================================

    kekurangan = max(
        kebutuhan - ketersediaan,
        0
    )

    kejenuhan = (
        (ketersediaan / kebutuhan) * 100
        if kebutuhan > 0
        else 0
    )

    rasio_kecukupan = (
        ketersediaan / kebutuhan
        if kebutuhan > 0
        else 0
    )

    # ==========================================
    # BARIS 2
    # ==========================================

    e1, e2 = st.columns(2)

    with e1:

        st.text_input(
            "Kekurangan TR2 (Unit)",
            value=f"{kekurangan:.2f}",
            disabled=True
        )

    with e2:

        st.text_input(
            "Kejenuhan (%)",
            value=f"{kejenuhan:.2f}",
            disabled=True
        )

    st.info(
        """
        - Kebutuhan Traktor Roda Dua (Unit) Diperoleh dari ( Luas Baku Sawah / 25 ) x Indeks Pertanaman (IP)\n
        - Kekurangan Traktor Roda Dua (Unit) Diperoleh dari (Kebutuhan Traktor Roda Dua (Unit) - Ketersediaan Traktor Roda Dua (Unit) )\n
        - Kejenuhan Traktor Roda Dua (%) Diperoleh dari (Ketersediaan Traktor Roda Dua (Unit) / Kebutuhan Traktor Roda Dua (Unit) x 100 )
        """
    )


    # ==========================================
    # PREDIKSI
    # ==========================================

    if st.button(
        "🚀 Prediksi Prioritas",
        use_container_width=True
    ):

        data_prediksi = pd.DataFrame({
            "luas_baku_sawah_ha": [luas_baku],
            "intensitas_tanam": [intensitas_tanam],
            "rasio_kecukupan": [rasio_kecukupan]
        })

        pred_num = model.predict(
            data_prediksi
        )[0]

        pred_label = label_encoder[
            int(pred_num)
        ]

        prob = model.predict_proba(
            data_prediksi
        )[0]

        confidence = (
            np.max(prob) * 100
        )

        st.divider()

       # ======================================================
        # HASIL REKOMENDASI
        # ======================================================

        st.subheader("📋 Hasil Rekomendasi")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Nama Kabupaten / Kota")

            st.info(
                f"🏙️ {kabupaten_kota}"
            )

        with col2:
            st.markdown("### Rekomendasi Prioritas Distribusi")

            if pred_label == "Tinggi":
                st.error("🔴 R1 = Tinggi")

            elif pred_label == "Sedang":
                st.warning("🟡 R1 = Sedang")

            else:
                st.success("🟢 R1 = Rendah")

        st.markdown("### Kesimpulan")

        if pred_label == "Tinggi":

            st.error(
                f"""
                Kabupaten/Kota **{kabupaten_kota}**
                termasuk dalam wilayah **Prioritas Tinggi**
                untuk distribusi bantuan Alsintan (Traktor Roda Dua).
                Dibutuhkan Penambahan Traktpr roda dua sebanyak f"{kekurangan:.2f}" untuk memenuhi
                kebutuhan alsintan di wilayah tersebut
                """
            )

        elif pred_label == "Sedang":

            st.warning(
                f"""
                Kabupaten/Kota **{kabupaten_kota}**
                termasuk dalam wilayah **Prioritas Sedang**
                untuk distribusi bantuan Alsintan (Traktor Roda Dua).
                """
            )

        else:

            st.success(
                f"""
                Kabupaten/Kota **{kabupaten_kota}**
                termasuk dalam wilayah **Prioritas Rendah**
                untuk distribusi bantuan Alsintan (Traktor Roda Dua).
                """
            )

        st.divider()

        # st.metric(
        #     "Confidence",
        #     f"{confidence:.2f}%"
        # )

        # st.subheader("Fitur yang Digunakan Model")

        # st.dataframe(
        #     data_prediksi,
        #     use_container_width=True
        # )
        st.subheader("Kelayakan Distribusi Probabilitas (%)")

            # Mengalikan dengan 100 dan memformat menjadi string dengan 2 angka di belakang koma
        prob_df = pd.DataFrame({
                "Kelas": [
                    label_encoder[0],
                    label_encoder[1],
                    label_encoder[2]
                ],
                "Probabilitas (%)": (prob * 100).round(2).astype(str) + " %"
            })

        st.dataframe(
                prob_df,
                use_container_width=True
            )