import streamlit as st
import pandas as pd
import numpy as np
import pickle
import folium
import os

from datetime import datetime
from streamlit_folium import st_folium
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
    page_icon="",
    layout="wide"
)

# ======================================================
# LOAD MODEL
# ======================================================

@st.cache_resource
def load_model():

    with open(
        "./models/composite_rf_best/model.pkl",
        "rb"
    ) as f:
        model = pickle.load(f)

    with open(
        "./models/composite_rf_best/label_encoder.pkl",
        "rb"
    ) as f:
        label_encoder = pickle.load(f)

    with open(
        "./models/composite_rf_best/metadata.pkl",
        "rb"
    ) as f:
        metadata = pickle.load(f)

    return model, label_encoder, metadata


model, label_encoder, metadata = load_model()

# ======================================================
# FILE HISTORI
# ======================================================

HISTORY_FILE = "dataset/history_prediksi.csv"

# Kolom standar untuk file histori (dipakai saat file kosong/baru dibuat)
HISTORY_COLUMNS = ["Kabupaten/Kota", "Prioritas", "Confidence", "Tanggal"]

# ======================================================
# LOAD HISTORI
# ======================================================

if "history_prediksi" not in st.session_state:

    if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:

        try:
            st.session_state.history_prediksi = pd.read_csv(
                HISTORY_FILE
            ).to_dict("records")

        except pd.errors.EmptyDataError:
            st.session_state.history_prediksi = []

    else:

        st.session_state.history_prediksi = []

# ======================================================
# DATA KABUPATEN
# ======================================================

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
# KOORDINAT
# ======================================================

KABUPATEN_COORDINATES = {

    'Bangkalan': [-7.030088018827346, 112.74871848591697],
    'Banyuwangi': [-8.218705941606514, 114.3703631600217],
    'Blitar': [-8.09552062796155, 112.16094939013382],
    'Bojonegoro': [-7.1524261058631895, 111.88712120588407],
    'Lamongan': [-7.118436279387343, 112.41358780406564],
    'Jember': [-8.17532824873553, 113.70295260257559],
    'Jombang': [-7.554442154702628, 112.23279413152756],
    'Kediri': [-7.846827595017761, 112.0172183015086],
    'Lumajang': [-8.132009472819155, 113.22196561107916],
    'Madiun': [-7.63073832103087, 111.53079823343255],
    'Magetan': [-7.655384016297651, 111.327904709175],
    'Mojokerto': [-7.469615740386046, 112.4399175190746],
    'Nganjuk': [-7.603960187356083, 111.8973957462841],
    'Ngawi': [-7.4072587883334124, 111.43442041435638],
    'Pacitan': [-8.180546548352382, 111.10484011151017],
    'Pasuruan': [-7.647507986314086, 112.90406331905739],
    'Ponorogo': [-7.867099437414933, 111.46655812859308],
    'Probolinggo': [-7.774859336264693, 113.20130420250976],
    'Sampang': [-7.189288480903806, 113.25256013437516],
    'Sidoarjo': [-7.45174571167862, 112.7020971347661],
    'Situbondo': [-7.706116266550805, 114.01574831236893],
    'Sumenep': [-7.001553307070617, 113.8593323311934],
    'Surabaya': [-7.257204807462584, 112.74667674351251],
    'Trenggalek': [-8.076639876186373, 111.70546393488439],
    'Tuban': [-6.89549369643621, 112.04321041012248],
    'Tulungagung': [-8.09286008625977, 111.96361281038588],
    'Malang': [-7.9663889369111445, 112.62903337261643],
    'Bondowoso': [-7.9141445257667336, 113.81932056843488],
    'Gresik': [-7.165598949093284, 112.65431091970251],
    'Kota Batu': [-7.883502778147921, 112.5338922540014],
    'Kota Blitar': [-8.095501242366215, 112.16302412401751],
    'Kota Kediri': [-7.822177386697373, 112.00948526637482],
    'Kota Madiun': [-7.631418755495591, 111.53217175993035],
    'Kota Malang': [-7.982083743258017, 112.63273700177898],
    'Kota Mojokerto': [-7.470381678502761, 112.43966005520048],
    'Kota Pasuruan': [-7.6391, 112.9094],
    'Kota Probolinggo': [-7.7554, 113.2075],
    'Kota Surabaya': [-7.2575, 112.7521],
    'Pamekasan': [-7.1648, 113.4835]
}


# ==============================
# SIDEBAR UI
# ==============================
st.sidebar.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <h1 style="color:#4CAF50;">JARVIS-Agro</h1>
        <hr style="border:1px solid #eee;">
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("assets/1.png", width=300)

menu = st.sidebar.radio(
    "",
    [
        "Information Project",
        "Simulasi Klasifikasi"
    ]
)

# ==============================
# PETUNJUK
# ==============================
st.sidebar.markdown(
    """
    <div style="
        background-color:#E3F2FD;
        padding:12px;
        border-radius:10px;
        margin-bottom:10px;
        color:#0D47A1;
        font-weight:500;
    ">Pilih menu di atas untuk melanjutkan. </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

# ==============================
# PROJECT INFO
# ==============================
st.sidebar.markdown(
    """
    <div style="
        background-color:#E8F5E9;
        padding:6px;
        border-radius:10px;
        margin-bottom:4px;
        color:#1B5E20;
        font-weight:600;
    ">
        <b style="
        display:block; 
        text-align:center;
        border-bottom:1px solid #1B5E20;
        padding-bottom:3px;
        margin-bottom:4px;
    ">
        FINAL PROJECT
    </b>
        Indonesia AI B10 ML - AI Intensive Bootcamp
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# TEAM INFO
# ==============================
st.sidebar.markdown(
    """
    <div style="
        background-color:#FFF3E0;
        padding:6px;
        border-radius:10px;
        margin-top:12px;
        color:#E65100;
        line-height:1.4;
        text-align:left;
    ">
        <b style="
            display:block;
            text-align:center;
            border-bottom:1px solid #E65100;
            padding-bottom:3px;
            margin-bottom:4px;
        ">
            ML-A IRON MAN
        </b>
        • Noviar Endru <br>
        • Singgih Mahardika <br>
        • Bagas Dwi Putra
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# INFORMATION PROJECT
# ======================================================

if menu == "Information Project":

    st.title("Sistem Penentuan Wilayah Kritis Alokasi Mesin Pertanian Berbasis Machine Learning")

    tab1, tab2, tab3, = st.tabs([
        "Tentang Sistem",
        "Dataset",
        "Metodologi",
        # "Model Terbaik"
    ])

    with tab1:

        st.header("Tentang Sistem")

        st.markdown("""
        Aplikasi ini menggunakan model Machine Learning terbaik
        yang telah dilatih untuk memprediksi prioritas bantuan
        traktor roda dua menjadi:

        🔴 Tinggi
        🟡 Sedang
        🟢 Rendah
        """)

    with tab2:

        st.header("Data Collection")

        st.markdown("""
        Kelompok data utama untuk membangun model prioritas

        ### 4 Kelompok Variabel

        **1. Administratif Wilayah**
        * Jumlah kecamatan
        * Jumlah desa

        **2. Luas & Produksi Padi**
        * Luas baku sawah
        * Luas panen
        * Produktivitas
        * Produksi padi/beras
        * IP

        **3. Indikator Alsintan**
        * Bantuan TR2
        * Ketersediaan
        * Kebutuhan
        * Kejenuhan

        **4. Label Prioritas**
        * Rendah
        * Sedang
        * Tinggi
        """)

    with tab3:

        st.header("Metodologi")

        st.markdown("""
        ### Data Processing Pipeline
        Alur persiapan data sebelum pemodelan

        **1. Load Data**
        * Dataset panel kabupaten-tahun

        **2. Feature Engineering**
        * Intensitas tanam, rasio kecukupan, defisit

        **3. Cek Missing Value**
        * Pastikan data lengkap

        **4. Drop Fitur Konstan**
        * Hapus variabel tanpa variasi

        **5. Encoding Label**
        * Rendah, Sedang, Tinggi

        **6. Group-based Validation**
        * Split berbasis kabupaten

        ---
        """)

    # with tab4:

    #     st.header("Model Terbaik")

    #     col1, col2 = st.columns(2)

    #     col1.metric(
    #         "Macro F1 CV",
    #         f"{metadata['cv_macro_f1_score']:.3f}"
    #     )

    #     col2.metric(
    #         "Held-Out Macro F1",
    #         f"{metadata['heldout_macro_f1_mean']:.3f}"
    #     )

# ======================================================
# MENU SIMULASI
# ======================================================

elif menu == "Simulasi Klasifikasi":

    st.title("Simulasi Klasifikasi Prioritas Bantuan")

    tab1, tab2, tab3 = st.tabs([
        "Simulasi",
        "Peta Sebaran",
        "Histori Simulasi"
    ])

    # ======================================================
    # TAB 1
    # ======================================================

    with tab1:

        kabupaten_kota = st.selectbox(
            "Kabupaten/Kota",
            KABUPATEN_KOTA
        )

        st.divider()

        st.subheader("Input Data Pertanian")

        # col1, col2 = st.columns(2)

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

        # ======================================================
        # PREDIKSI
        # ======================================================

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

            histori_data = {

                "Kabupaten/Kota": kabupaten_kota,
                "Prioritas": pred_label,
                "Confidence": f"{confidence:.2f}%",
                "Tanggal": datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
            }

            st.session_state.history_prediksi.append(
                histori_data
            )

            # ======================================================
            # SIMPAN KE CSV
            # ======================================================

            pd.DataFrame(
                st.session_state.history_prediksi
            ).to_csv(
                HISTORY_FILE,
                index=False
            )

            # ======================================================
            # SIMPAN HASIL TERAKHIR KE SESSION STATE
            # PERBAIKAN: supaya hasil tidak hilang saat rerun
            # (misal akibat interaksi st_folium di tab lain).
            # ======================================================

            prob_df = pd.DataFrame({

                "Kelas": [
                    label_encoder[0],
                    label_encoder[1],
                    label_encoder[2]
                ],

                "Probabilitas (%)":
                (prob * 100).round(2)
            })

            kekurangan_traktor = max(
                0,
                int(np.ceil(kebutuhan - ketersediaan))
            )

            st.session_state.last_pred_label = pred_label
            st.session_state.last_prob_df = prob_df
            st.session_state.last_kabupaten = kabupaten_kota
            st.session_state.last_kebutuhan = kebutuhan
            st.session_state.last_ketersediaan = ketersediaan
            st.session_state.last_kekurangan = kekurangan_traktor

        # ======================================================
        # TAMPILKAN HASIL PREDIKSI TERAKHIR (DI LUAR IF BUTTON)


        if "last_pred_label" in st.session_state:

            st.subheader("📋 Hasil Prediksi")

            pred_label = st.session_state.last_pred_label
            nama_wilayah = st.session_state.last_kabupaten
            kekurangan = st.session_state.last_kekurangan

            warna_map = {
                "Tinggi": "#e74c3c",
                "Sedang": "#f39c12",
                "Rendah": "#27ae60"
            }

            emoji_map = {
                "Tinggi": "🔴",
                "Sedang": "🟡",
                "Rendah": "🟢"
            }

            warna = warna_map.get(pred_label, "#3498db")
            emoji = emoji_map.get(pred_label, "")

            # ======================================================
            # CARD: NAMA WILAYAH & REKOMENDASI PRIORITAS
            # ======================================================

            card_col1, card_col2 = st.columns(2)

            with card_col1:

                st.markdown(f"""
                <div style="
                    background-color:{warna}15;
                    border:1px solid {warna};
                    border-radius:12px;
                    padding:18px;
                    text-align:center;
                    height:100%;
                ">
                    <p style="margin:0; color:#777; font-size:14px;">
                        Nama Kabupaten/Kota
                    </p>
                    <h3 style="margin:6px 0 0 0; color:#777;">
                        {nama_wilayah}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

            with card_col2:

                st.markdown(f"""
                <div style="
                    background-color:{warna}15;
                    border:1px solid {warna};
                    border-radius:12px;
                    padding:18px;
                    text-align:center;
                    height:100%;
                ">
                    <p style="margin:0; color:#777; font-size:14px;">
                        Rekomendasi Prioritas Distribusi
                    </p>
                    <h3 style="margin:6px 0 0 0; color:{warna};">
                        {emoji} {pred_label}
                    </h3>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # ======================================================
            # CARD: KETERANGAN
            # ======================================================

            st.markdown(f"""
            <div style="
                background-color:#f8f9fa;
                border-left:5px solid {warna};
                border-radius:8px;
                padding:18px 20px;
            ">
                <p style="margin:0 0 8px 0; font-weight:600; color:#333;">
                    Keterangan:
                </p>
                <p style="margin:0 0 10px 0; color:#444;">
                    Kabupaten/Kota <b>{nama_wilayah}</b> termasuk dalam
                    wilayah prioritas <b style="color:{warna};">{pred_label}</b>
                    untuk distribusi Alsintan.
                </p>
                <p style="margin:0; color:#444;">
                    {"Dibutuhkan penambahan traktor roda dua sebanyak "
                    f"<b>{kekurangan}</b> unit untuk memenuhi kebutuhan "
                    "Alsintan di wilayah tersebut."
                    if kekurangan > 0 else
                    "Ketersediaan traktor roda dua di wilayah ini sudah "
                    "mencukupi kebutuhan Alsintan."}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            # ======================================================
            # TABEL PROBABILITAS
            # ======================================================

            st.markdown("**Kelayakan Distribusi Probabilitas**")

            st.dataframe(
                st.session_state.last_prob_df,
                use_container_width=True
            )

    # ======================================================
    # TAB 2
    # ======================================================

    with tab2:

        st.subheader(
            "Peta Sebaran Prioritas"
        )

        df_map = pd.read_csv(
            "dataset/r2dataset_compatible.csv"
        )

        tahun_list = sorted(
            df_map["tahun_pengadaan"].unique()
        )

        tahun_selected = st.selectbox(
            "Pilih Tahun",
            tahun_list
        )

        df_filtered = df_map[
            df_map["tahun_pengadaan"]
            == tahun_selected
        ]

        warna_prioritas = {

            "Rendah": "green",
            "Sedang": "orange",
            "Tinggi": "red"
        }

        m = folium.Map(
            location=[-7.7, 112.7],
            zoom_start=8
        )

        for _, row in df_filtered.iterrows():

            kabupaten = row[
                "kabupaten_kota"
            ]

            prioritas = row[
                "rekomendasi_prioritas_bantuan"
            ]

            if kabupaten in KABUPATEN_COORDINATES:

                lat, lon = (
                    KABUPATEN_COORDINATES[
                        kabupaten
                    ]
                )

                warna = warna_prioritas.get(
                    prioritas,
                    "blue"
                )

                folium.CircleMarker(

                    location=[lat, lon],
                    radius=10,
                    color=warna,
                    fill=True,
                    fill_color=warna,
                    fill_opacity=0.8,

                    popup=f"""
                    <b>{kabupaten}</b><br>
                    Prioritas: {prioritas}<br>
                    Tahun: {tahun_selected}
                    """

                ).add_to(m)

        st_folium(
            m,
            width=1200,
            height=600
        )

        st.info("""
        🔴 Tinggi
        🟡 Sedang
        🟢 Rendah
        """)

    # ======================================================
    # TAB 3
    # ======================================================


    # ======================================================
    # TAB 3 — HISTORIS SIMULATION
    # ======================================================

    with tab3:

        st.subheader(
            "📁 Historis Simulasi Prediksi"
        )

        # ======================================================
        # CEK HISTORI
        # ======================================================

        if len(
            st.session_state.history_prediksi
        ) == 0:

            st.info(
                "Belum ada histori prediksi."
            )

        else:

            # ======================================================
            # DATAFRAME HISTORI
            # ======================================================

            df_history = pd.DataFrame(
                st.session_state.history_prediksi
            )

            st.dataframe(
                df_history,
                use_container_width=True
            )

            # ======================================================
            # DOWNLOAD CSV
            # ======================================================

            csv = df_history.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                label="⬇ Download Historis CSV",

                data=csv,

                file_name=
                "historis_prediksi.csv",

                mime="text/csv",

                use_container_width=True
            )

            # ======================================================
            # PETA HISTORI
            # ======================================================

            st.divider()

            st.subheader(
                "🗺️ Peta Historis Simulasi"
            )

            # ======================================================
            # LOAD DATASET ASLI
            # ======================================================

            df_dataset = pd.read_csv(
                "dataset/r2dataset_compatible.csv"
            )

            # ======================================================
            # FILTER TAHUN
            # ======================================================

            tahun_histori = st.selectbox(
                "Pilih Tahun Historis",
                sorted(
                    df_dataset[
                        "tahun_pengadaan"
                    ].unique()
                ),
                key="tahun_histori"
            )

            # ======================================================
            # FILTER DATASET
            # ======================================================

            df_dataset = df_dataset[
                df_dataset["tahun_pengadaan"]
                == tahun_histori
            ].copy()

            # ======================================================
            # WARNA PRIORITAS DATASET
            # ======================================================

            warna_prioritas = {

                "Rendah": "green",
                "Sedang": "orange",
                "Tinggi": "red"
            }

            # ======================================================
            # BASE MAP
            # ======================================================

            m_history = folium.Map(
                location=[-7.7, 112.7],
                zoom_start=8
            )

            # ======================================================
            # PLOT DATASET ASLI
            # ======================================================

            for _, row in df_dataset.iterrows():

                kabupaten = row[
                    "kabupaten_kota"
                ]

                prioritas = row[
                    "rekomendasi_prioritas_bantuan"
                ]

                if kabupaten in KABUPATEN_COORDINATES:

                    lat, lon = (
                        KABUPATEN_COORDINATES[
                            kabupaten
                        ]
                    )

                    warna = warna_prioritas.get(
                        prioritas,
                        "gray"
                    )

                    folium.CircleMarker(

                        location=[lat, lon],

                        radius=9,

                        color=warna,

                        fill=True,

                        fill_color=warna,

                        fill_opacity=0.7,

                        popup=f"""
                        <b>{kabupaten}</b><br>
                        Sumber: Dataset Asli<br>
                        Prioritas: {prioritas}<br>
                        Tahun: {tahun_histori}
                        """

                    ).add_to(m_history)

            # ======================================================
            # PLOT HISTORI USER
            # ======================================================

            for item in st.session_state.history_prediksi:

                kabupaten = item[
                    "Kabupaten/Kota"
                ]

                prioritas = item[
                    "Prioritas"
                ]

                confidence = item[
                    "Confidence"
                ]

                tanggal = item[
                    "Tanggal"
                ]

                if kabupaten in KABUPATEN_COORDINATES:

                    lat, lon = (
                        KABUPATEN_COORDINATES[
                            kabupaten
                        ]
                    )

                    # ======================================================
                    # HISTORI = BIRU
                    # ======================================================

                    folium.CircleMarker(

                        location=[lat, lon],

                        radius=13,

                        color="blue",

                        fill=True,

                        fill_color="blue",

                        fill_opacity=0.9,

                        popup=f"""
                        <b>{kabupaten}</b><br>
                        Sumber: Historis Simulasi<br>
                        Prioritas: {prioritas}<br>
                        Confidence: {confidence}<br>
                        Tanggal: {tanggal}
                        """

                    ).add_to(m_history)

            # ======================================================
            # TAMPILKAN MAP
            # ======================================================

            st_folium(
                m_history,
                width=1200,
                height=650
            )

            # ======================================================
            # LEGENDA
            # ======================================================

            st.info("""
            🔴 Historis Tinggi  
            🟡 Historis Sedang  
            🟢 Historis Rendah  
            🔵 Historis Simulasi User
            """)

            # ======================================================
            # HAPUS HISTORI
            # ======================================================
            # PERBAIKAN:
            # Sebelumnya pd.DataFrame().to_csv(...) menulis file CSV
            # yang benar-benar kosong (tanpa header/kolom), sehingga
            # pd.read_csv() di rerun selanjutnya gagal dengan
            # "EmptyDataError: No columns to parse from file".
            # Sekarang ditulis dengan kolom yang sesuai schema histori,
            # supaya file tetap valid meskipun datanya 0 baris.

            if st.button(
                "🗑️ Hapus Historis",
                use_container_width=True
            ):

                st.session_state.history_prediksi = []

                pd.DataFrame(
                    columns=HISTORY_COLUMNS
                ).to_csv(
                    HISTORY_FILE,
                    index=False
                )

                st.rerun()