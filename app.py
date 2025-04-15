import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Fungsi untuk menentukan status kelulusan
def cek_status(u, t, k):
    return 'Lulus' if u >= 75 and t >= 75 and k >= 80 else 'Tidak Lulus'

# Data asli dan data dummy
data = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70],
    'nilai_tugas': [85, 75, 60, 95, 68],
    'kehadiran': [90, 80, 70, 95, 75],
})

# Data dummy tambahan
data_dummy = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70, 88, 91, 78, 84, 60, 79, 85, 66, 95, 68, 76, 82, 73, 77, 92],
    'nilai_tugas': [85, 75, 60, 95, 68, 90, 94, 80, 88, 63, 78, 85, 69, 96, 72, 78, 84, 76, 80, 93],
    'kehadiran': [90, 80, 70, 95, 75, 88, 92, 82, 90, 72, 81, 87, 74, 94, 78, 80, 85, 78, 83, 91],
})

# Gabungkan data asli dan data dummy
data = pd.concat([data, data_dummy], ignore_index=True)

# Update status kelulusan berdasarkan data
data['status'] = data.apply(lambda row: cek_status(row['nilai_ujian'], row['nilai_tugas'], row['kehadiran']), axis=1)

# Encode status kelulusan menjadi angka
le = LabelEncoder()
data['status_encoded'] = le.fit_transform(data['status'])

# Latih model Logistic Regression
X = data[['nilai_ujian', 'nilai_tugas', 'kehadiran']]
y = data['status_encoded']
model = LogisticRegression()
model.fit(X, y)

# Evaluasi model dengan Cross Validation (CV)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold Cross Validation
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Visualisasi cross-validation result
st.sidebar.subheader("📈 Cross-Validation")
st.sidebar.write(f"CV Akurasi Rata-rata: **{cv_mean:.2f}**")
st.sidebar.write(f"Standar Deviasi: ±{cv_std:.2f}")

# Visualisasi data kelulusan
fig, ax = plt.subplots()
sns.countplot(x='status', data=data, ax=ax)
ax.set_title("Distribusi Status Kelulusan")
st.pyplot(fig)

# UI Web - Formulir Input untuk prediksi
st.title("Prediksi Kelulusan Siswa")

# Input nilai dari user
nilai_ujian = st.slider("Nilai Ujian", 0, 100, 75)
nilai_tugas = st.slider("Nilai Tugas", 0, 100, 75)
kehadiran = st.slider("Persentase Kehadiran (%)", 0, 100, 80)

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    input_data = pd.DataFrame([[nilai_ujian, nilai_tugas, kehadiran]], columns=["nilai_ujian", "nilai_tugas", "kehadiran"])
    pred = model.predict(input_data)
    status = le.inverse_transform(pred)
    st.success(f"Hasil Prediksi: {status[0]}")

