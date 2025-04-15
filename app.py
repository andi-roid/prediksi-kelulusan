import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score

st.title("Prediksi Kelulusan Siswa v2")


# Data asli sebelum ditambah data dummy
data = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70],
    'nilai_tugas': [85, 75, 60, 95, 68],
    'kehadiran': [90, 80, 70, 95, 75],
})

# Tambah data dummy ke dataset
data_dummy = pd.DataFrame({
    'nilai_ujian': [80, 72, 65, 90, 70, 88, 91, 78, 84, 60, 79, 85, 66, 95, 68, 76, 82, 73, 77, 92],
    'nilai_tugas': [85, 75, 60, 95, 68, 90, 94, 80, 88, 63, 78, 85, 69, 96, 72, 78, 84, 76, 80, 93],
    'kehadiran': [90, 80, 70, 95, 75, 88, 92, 82, 90, 72, 81, 87, 74, 94, 78, 80, 85, 78, 83, 91],
})

# Gabungkan data asli dan data dummy
data = pd.concat([data, data_dummy], ignore_index=True)

# Update status kelulusan berdasarkan data yang baru
data['status'] = data.apply(lambda row: cek_status(row['nilai_ujian'], row['nilai_tugas'], row['kehadiran']), axis=1)

# Encode target
le = LabelEncoder()
data['status_encoded'] = le.fit_transform(data['status'])

# Latih model dengan data yang sudah ditambah
X = data[['nilai_ujian', 'nilai_tugas', 'kehadiran']]
y = data['status_encoded']
model = LogisticRegression()
model.fit(X, y)

# Evaluasi model dengan Cross Validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold CV
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

# Tampilkan hasil evaluasi
st.sidebar.subheader("📈 Cross-Validation")
st.sidebar.write(f"CV Akurasi Rata-rata: **{cv_mean:.2f}**")
st.sidebar.write(f"Standar Deviasi: ±{cv_std:.2f}")

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

from sklearn.model_selection import cross_val_score

# Lakukan Cross Validation
cv_scores = cross_val_score(model, X, y, cv=3)  # 5-fold CV
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()


# Evaluasi model dengan data training
y_pred_train = model.predict(X)
akurasi = accuracy_score(y, y_pred_train)
conf_matrix = confusion_matrix(y, y_pred_train)

st.sidebar.subheader("📊 Evaluasi Model")
st.sidebar.write(f"Akurasi model pada data latih: **{akurasi:.2f}**")

# Visualisasi confusion matrix
st.sidebar.write("Confusion Matrix:")
fig_cm = plt.figure(figsize=(3, 2))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.sidebar.pyplot(fig_cm)
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Cross-Validation")
st.sidebar.write(f"CV Akurasi Rata-rata: **{cv_mean:.2f}**")
st.sidebar.write(f"Standar Deviasi: ±{cv_std:.2f}")


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


