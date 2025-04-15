import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Dummy data buat latih model
data = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70],
    'nilai_tugas': [85, 75, 60, 95, 68],
    'kehadiran': [90, 80, 70, 95, 75],
})

def cek_status(u, t, k):
    return 'Lulus' if u >= 75 and t >= 75 and k >= 80 else 'Tidak Lulus'

data['status'] = data.apply(lambda row: cek_status(row['nilai_ujian'], row['nilai_tugas'], row['kehadiran']), axis=1)

# Encode target
le = LabelEncoder()
data['status_encoded'] = le.fit_transform(data['status'])

# Latih model
X = data[['nilai_ujian', 'nilai_tugas', 'kehadiran']]
y = data['status_encoded']
model = LogisticRegression()
model.fit(X, y)

# UI Web
st.title("Prediksi Kelulusan Siswa")

nilai_ujian = st.slider("Nilai Ujian", 0, 100, 75)
nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
kehadiran = st.slider("Persentase Kehadiran (%)", 0, 100, 80)

if st.button("Prediksi"):
    input_data = pd.DataFrame([[nilai_ujian, nilai_tugas, kehadiran]], columns=["nilai_ujian", "nilai_tugas", "kehadiran"])
    pred = model.predict(input_data)
    status = le.inverse_transform(pred)
    st.success(f"Hasil Prediksi: {status[0]}")
