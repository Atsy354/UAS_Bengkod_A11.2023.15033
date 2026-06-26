import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load(
    'model_churn.pkl'
)

# Judul
st.title(
    'Aplikasi Prediksi Customer Churn'
)

st.write(
    'Upload file CSV untuk melakukan prediksi customer churn'
)

# Upload file
uploaded = st.file_uploader(
    "Upload CSV",
    type=['csv']
)

if uploaded:

    # Baca data
    data = pd.read_csv(
        uploaded
    )

    st.subheader(
        "Preview Data"
    )

    st.dataframe(
        data.head()
    )

    if st.button(
        "Prediksi"
    ):

        try:

            # Copy data agar data asli tidak berubah
            predict_data = data.copy()

            # Hapus target jika ada
            if 'churn' in predict_data.columns:
                predict_data = predict_data.drop(
                    columns=['churn']
                )

            # Feature engineering tanggal
            if 'signup_date' in predict_data.columns:

                predict_data['signup_date'] = (
                    pd.to_datetime(
                        predict_data['signup_date'],
                        errors='coerce'
                    )
                )

                predict_data['signup_year'] = (
                    predict_data['signup_date']
                    .dt.year
                )

            if 'last_purchase_date' in predict_data.columns:

                predict_data['last_purchase_date'] = (
                    pd.to_datetime(
                        predict_data['last_purchase_date'],
                        errors='coerce'
                    )
                )

                predict_data['last_purchase_year'] = (
                    predict_data[
                        'last_purchase_date'
                    ].dt.year
                )

            # Hapus kolom tanggal lama
            predict_data = predict_data.drop(
                columns=[
                    'signup_date',
                    'last_purchase_date'
                ],
                errors='ignore'
            )

            # Prediksi
            pred = model.predict(
                predict_data
            )

            # Simpan hasil
            data['Prediction'] = pred

            # Label lebih jelas
            data['Prediction'] = (
                data[
                    'Prediction'
                ]
                .map({
                    0: 'Tidak Churn',
                    1: 'Churn'
                })
            )

            st.success(
                "Prediksi berhasil"
            )

            st.subheader(
                "Hasil Prediksi"
            )

            st.dataframe(
                data
            )

            # Download
            csv = (
                data
                .to_csv(
                    index=False
                )
                .encode(
                    'utf-8'
                )
            )

            st.download_button(

                "Download Hasil",

                csv,

                "hasil_prediksi.csv",

                "text/csv"

            )

        except Exception as e:

            st.error(
                f"Terjadi error: {e}"
            )