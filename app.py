import streamlit as st
import pandas as pd
import joblib

# ======================================
# KONFIGURASI HALAMAN
# ======================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ======================================
# LOAD MODEL
# ======================================

try:
    model = joblib.load("model_churn.pkl")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================
# HEADER
# ======================================

st.title("📊 Customer Churn Prediction")

st.write(
    """
    Masukkan data pelanggan di bawah ini untuk mengetahui apakah pelanggan
    diprediksi **Churn (1)** atau **Tidak Churn (0)**.
    """
)

st.markdown("---")

# ======================================
# FORM INPUT
# ======================================

st.subheader("Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:

    total_spent = st.number_input(
        "💰 Total Spent",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.0f",
        help="Total pengeluaran pelanggan."
    )

    support_tickets = st.number_input(
        "🎫 Support Tickets",
        min_value=0,
        value=0,
        step=1,
        help="Jumlah tiket keluhan pelanggan."
    )

    pages_per_session = st.number_input(
        "🌐 Pages Per Session",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.1f",
        help="Rata-rata halaman yang dikunjungi."
    )

with col2:

    satisfaction_score = st.slider(
        "⭐ Satisfaction Score",
        min_value=1,
        max_value=10,
        value=5,
        help="Nilai kepuasan pelanggan."
    )

    marketing_spend_per_user = st.number_input(
        "📢 Marketing Spend Per User",
        min_value=0.0,
        value=0.0,
        step=50.0,
        format="%.0f",
        help="Biaya pemasaran untuk pelanggan."
    )

st.markdown("---")

# ======================================
# TOMBOL PREDIKSI
# ======================================

if st.button("🔍 Prediksi", use_container_width=True):

    input_data = pd.DataFrame({

        "total_spent":[total_spent],

        "satisfaction_score":[satisfaction_score],

        "support_tickets":[support_tickets],

        "marketing_spend_per_user":[marketing_spend_per_user],

        "pages_per_session":[pages_per_session]

    })

    prediction = int(
        model.predict(input_data)[0]
    )

    confidence = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)

        confidence = probability.max() * 100

    st.markdown("---")

    st.subheader("📈 Hasil Prediksi")

    colA, colB = st.columns(2)

    with colA:

        st.metric(
            "Prediction",
            prediction
        )

    with colB:

        if confidence is not None:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

    st.markdown("### Interpretasi")

    if prediction == 0:

        st.success(
            "✅ Pelanggan diprediksi **Tidak Churn (0)**."
        )

        st.info(
            "Pelanggan diperkirakan tetap menggunakan layanan sehingga risiko churn relatif rendah."
        )

    else:

        st.error(
            "⚠️ Pelanggan diprediksi **Churn (1)**."
        )

        st.warning(
            "Pelanggan diperkirakan akan berhenti menggunakan layanan sehingga perusahaan perlu melakukan strategi retensi."
        )

    st.markdown("---")

    with st.expander("📖 Keterangan Label"):

        st.write("""
        **0 = Tidak Churn**

        Pelanggan diprediksi tetap menggunakan layanan.

        **1 = Churn**

        Pelanggan diprediksi berhenti menggunakan layanan.
        """)

# ======================================
# FOOTER
# ======================================

st.markdown("---")

st.caption("""
Model Machine Learning : Random Forest Classifier

Jumlah Feature : 5

Target : Customer Churn

0 = Tidak Churn

1 = Churn
""")