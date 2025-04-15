import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.title("Prediksi Kelulusan Siswa v2")

# Data dummy untuk melatih model
data = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70],
    'nilai_tugas': [85, 75, 60, 95, 68],
    'kehadiran': [90, 80, 70, 95, 75],
})

# Label kelulusan berdasarkan kriteria
def cek_status(u, t, k):
    return 'Lulus' if u >= 75 and t >= 75 and k >= 80 else 'Tidak Lulus'

data['status'] = data.apply(lambda row: cek_status(row['nilai_ujian'], row['nilai_tugas'], row['kehadiran']), axis=1)

# Encode target
le = LabelEncoder()
data['status_encoded'] = le.fit_transform(data['status'])

# Melatih model
X = data[['nilai_ujian', 'nilai_tugas', 'kehadiran']]
y = data['status_encoded']
model = LogisticRegression()
model.fit(X, y)

# Upload file dari user
uploaded_file = st.file_uploader("Upload file CSV kamu di sini", type=["csv"])

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)

    # Validasi kolom
    required_cols = ['nilai_ujian', 'nilai_tugas', 'kehadiran']
    if all(col in user_data.columns for col in required_cols):
        st.subheader("Data yang Diupload")
        st.dataframe(user_data)

        # Prediksi
        pred = model.predict(user_data[required_cols])
        hasil = le.inverse_transform(pred)
        user_data['Hasil Prediksi'] = hasil

        st.subheader("Hasil Prediksi Kelulusan")
        st.dataframe(user_data)

        # Visualisasi
        st.subheader("Visualisasi Nilai Ujian vs Nilai Tugas")
        fig, ax = plt.subplots()
        warna = user_data['Hasil Prediksi'].map({'Lulus': 'green', 'Tidak Lulus': 'red'})
        ax.scatter(user_data['nilai_ujian'], user_data['nilai_tugas'], c=warna)
        ax.set_xlabel("Nilai Ujian")
        ax.set_ylabel("Nilai Tugas")
        ax.set_title("Visualisasi Prediksi")
        st.pyplot(fig)
    else:
        st.error("Kolom dalam file harus: nilai_ujian, nilai_tugas, kehadiran")
else:
    st.info("Silakan upload file CSV untuk mulai prediksi.")


if __name__ == "__main__":
    main()
