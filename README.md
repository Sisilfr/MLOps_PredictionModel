# Deteksi Penipuan Pembukaan Rekening Bank (Kasus A)

Implementasi model *machine learning* untuk mendeteksi penipuan (*fraud*) pada pengajuan pembukaan rekening bank digital menggunakan dataset **Bank Account Fraud (BAF) varian Base (Bulan 0–3)**.

---

## 1. Metode yang Digunakan

* **Algoritma Model:** LightGBM (*Light Gradient Boosting Machine*) dengan 300 estimators, learning rate 0.05, dan 31 daun (*num_leaves*).
* **Pra-pemrosesan Data:** *Ordinal Encoding* pada 5 variabel kategorikal (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) dengan fitting yang hanya dilakukan pada data latih.
* **Kalibrasi Ambang Batas (*Decision Threshold*):** Penetapan batas probabilitas operasional melalui kurva ROC untuk mengunci *False Positive Rate* (FPR) pada batas toleransi operasional $\le 5\%$.

---

## 2. Alasan Memilih Metode

1. **Efektivitas pada Data Tabular Imbalanced:** Arsitektur *Gradient Boosted Decision Trees* terbukti unggul dan menjadi standar industri dalam menangani data tabular transaksi finansial dengan ketidakseimbangan kelas ekstrem (~1% fraud).
2. **Efisiensi Komputasi Tinggi:** Algoritma berbasis *histogram binning* pada LightGBM memungkinkan pelatihan ~400.000 baris data selesai dalam hitungan detik tanpa membutuhkan akselerasi GPU.
3. **Kalibrasi Probabilitas yang Halus:** Skor probabilitas keluaran LightGBM terdistribusi secara kontinu, memudahkan penentuan *threshold* keputusan yang presisi untuk mematuhi batas kesalahan nasabah wajar.
4. **Artefak Ringan & Portabel:** Ukuran model tersimpan hanya **~1.00 MB**, sehingga dapat disimpan langsung di repositori Git tanpa dependensi Git LFS.

---

## 3. Pembagian Data Latih dan Validasi (*Data Splitting*)

Dataset dipartisi secara **Temporal (Out-of-Time / OOT)** berdasarkan kolom waktu `month`:
* **Data Latih (Train):** Bulan 0, 1, dan 2 (**397.039 baris** / 72,5% data)
* **Data Validasi (Validation):** Bulan 3 (**150.936 baris** / 27,5% data)

### Rasional Pemilihan Strategi
Pola dan teknik penipuan perbankan terus mengalami pergeseran konsep (*concept drift*) seiring waktu. Penggunaan *random split* berisiko menimbulkan kebocoran informasi temporal (*lookahead bias*), di mana model memanfaatkan pola masa depan untuk memprediksi masa lalu. Pendekatan *temporal split* merefleksikan kondisi operasional nyata di mana model dilatih menggunakan data historis (bulan 0–2) untuk memprediksi pengajuan nasabah baru pada periode berikutnya (bulan 3).

---

## 4. Skor di Data Validasi (Bulan 3)

Evaluasi dilakukan pada data validasi bulan 3 dengan membatasi nasabah wajar yang keliru ditandai maksimal 5% (*FPR $\le$ 5%*).

| Metrik Evaluasi | Nilai Capaian | Keterangan / Ketentuan Tugas |
|---|:---:|---|
| **ROC-AUC Score** | **0.9100** | Menunjukkan kemampuan diskriminasi kelas yang sangat baik |
| **Batas Ambang (*Threshold*)** | **0.0523** | Pengajuan dengan skor $\ge$ 0.0523 ditandai sebagai indikasi fraud |
| **Nasabah Wajar Tertandai (*FPR*)** | **4.98%** | **Memenuhi syarat operasional (Batas Maksimal $\le 5.0\%$)** |
| **Penipu Berhasil Ditangkap (*TPR*)** | **56.54%** | **Metrik utama: 56.54% dari total penipu berhasil terdeteksi** |

### Distribusi Matriks Kebingungan (*Confusion Matrix*) pada Threshold 0.0523

| | Prediksi Wajar (0) | Prediksi Fraud (1) | Total Aktual |
|---|:---:|:---:|:---:|
| **Aktual Wajar (0)** | **142.093** (TN) | **7.451** (FP) | 149.544 |
| **Aktual Fraud (1)** | **605** (FN) | **787** (TP) | 1.392 |
| **Total Prediksi** | 142.698 | 8.238 | 150.936 |

* **True Positive (TP = 787):** Sebanyak 56,54% fraud tertangkap oleh sistem.
* **False Positive (FP = 7.451):** Sebesar 4,98% nasabah wajar yang memerlukan verifikasi manual (memenuhi batas $\le 5\%$).
* **True Negative (TN = 142.093):** Nasabah wajar yang langsung diproses secara otomatis.
* **False Negative (FN = 605):** Kasus fraud yang belum terdeteksi pada batas toleransi FPR 5%.

---

## 5. Panduan Menjalankan & Memuat Model

1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Eksekusi Notebook:**
   Jalankan `Bank_Fraud_Model_Prediction.ipynb` dari atas ke bawah. Berkas data akan dimuat dari direktori lokal atau diunduh otomatis jika belum tersedia.
3. **Memuat Ulang Model Tersimpan:**
   ```python
   import joblib

   # Pemuatan artefak model dan threshold terkalibrasi
   bundle = joblib.load('model/fraud_detector_lgbm.joblib')
   model = bundle['model']
   threshold = bundle['threshold']
   print(f"Model berhasil dimuat dengan threshold operasional: {threshold:.4f}")
   ```

---

## 6. Struktur Repositori

```text
.
├── model/
│   └── fraud_detector_lgbm.joblib       # Artefak model tersimpan (~1 MB)
├── Bank_Fraud_Model_Prediction.ipynb    # Notebook terstruktur (eksekusi tanpa error)
├── requirements.txt                     # Daftar pustaka dan versi yang dibutuhkan
├── README.md                            # Laporan metode dan hasil validasi (1 halaman)
└── .gitignore                           # Konfigurasi pengabaian berkas data besar & temporary
```
