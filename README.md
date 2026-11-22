# 🏦 Deteksi Penipuan Pembukaan Rekening Bank (Kasus A)

Repository ini berisi implementasi machine learning untuk mendeteksi penipuan (*fraud*) pada pengajuan pembukaan rekening bank digital menggunakan dataset **Bank Account Fraud (BAF) varian Base (Bulan 0–3)**.

---

## 1. Metode yang Digunakan
* **Algoritma Model:** **LightGBM (Light Gradient Boosting Machine)**
* **Preprocessing:** *Ordinal Encoding* pada 5 variabel kategorikal (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`).
* **Optimasi Threshold:** Pencarian batas ambang probabilitas operasional (*decision threshold*) pada kurva ROC agar tingkat kesalahan penuduhan nasabah jujur (*False Positive Rate / FPR*) terkunci **$\le 5\%$**.

---

## 2. Alasan Memilih Metode
1. **Standar Industri FinTech:** Algoritma *Gradient Boosted Decision Trees* (seperti LightGBM) secara konsisten menjadi standar terbaik untuk data tabel transaksi finansial yang memiliki ketimpangan kelas ekstrem (*imbalanced data* ~1% fraud).
2. **Cepat & Efisien:** Model dilatih pada hampir 400.000 data dalam waktu belasan detik tanpa membutuhkan GPU.
3. **Ukuran File Ringan:** Bobot model tersimpan hanya **~1.00 MB** sehingga mudah disimpan di GitHub dan dimuat ulang (*reload*) tanpa dependensi Git LFS.
4. **Probabilitas Terkalibrasi Baik:** Memudahkan penentuan *decision threshold* yang presisi untuk mematuhi batas eror 5%.

---

## 3. Pembagian Data Latih dan Validasi (*Data Splitting*)
Data dibagi secara **Temporal (Out-of-Time / OOT)** berdasarkan kolom waktu `month`:
* **Data Latih (Train):** Bulan 0, 1, dan 2 (**397.039 baris** / ~72.5% data)
* **Data Validasi (Validation):** Bulan 3 (**150.936 baris** / ~27.5% data)

### 💡 Alasan Pemilihan:
Di industri perbankan nyata, pola penipuan selalu berkembang seiring waktu (*concept drift*). Pengacakan data secara acak (*random split*) akan menyebabkan kebocoran data masa depan ke masa lalu (*data leakage/lookahead bias*). Dengan *temporal split*, performa model diuji secara realistis: **dilatih menggunakan riwayat masa lalu (bulan 0–2) untuk memprediksi calon nasabah baru di masa depan (bulan 3)**.

---

## 4. Skor di Data Validasi (Bulan 3)
Sesuai instruksi, nasabah jujur yang keliru tertandai dibatasi paling banyak 5% (*FPR $\le$ 5%*).

| Metrik Evaluasi | Nilai Capaian | Keterangan / Syarat Tugas |
|---|:---:|---|
| **ROC-AUC Score** | **0.9100** | Menunjukkan kemampuan pemisahan kelas yang sangat kuat |
| **Batas Ambang (*Threshold*)** | **0.0523** | Pengajuan dengan skor $\ge$ 0.0523 ditandai sebagai indikasi fraud |
| **Nasabah Jujur Tertandai (*FPR*)** | **4.98%** | **MEMENUHI SYARAT (Batas Maksimal $\le 5.0\%$)** |
| **Penipu Tertangkap (*TPR*)** | **56.54%** | **56.54% dari total penipu berhasil dijaring sistem** |

> **Penjelasan untuk Orang Awam:**  
> Dari setiap 100 nasabah jujur yang mengajukan rekening, sistem meloloskan lebih dari 95 orang secara instan (hanya 4.98 orang yang perlu verifikasi ulang). Pada saat yang sama, sistem berhasil menangkap **56.54% penipu** yang mencoba membobol bank.

---

## 5. Cara Menjalankan Notebook & Memuat Model

1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan Notebook:**
   Buka dan jalankan `notebook.ipynb` dari atas ke bawah. Dataset akan diunduh otomatis jika belum ada.
3. **Memuat Model Tersimpan:**
   ```python
   import joblib

   # Muat paket model lengkap
   bundle = joblib.load('model/fraud_detector_lgbm.joblib')
   model = bundle['model']
   threshold = bundle['threshold']
   print(f"Model berhasil dimuat dengan threshold operasional: {threshold:.4f}")
   ```

---

## 6. Struktur Direktori Repository
```text
.
├── model/
│   └── fraud_detector_lgbm.joblib   # File model tersimpan (~1 MB, siap pakai)
├── notebook.ipynb                   # Notebook lengkap & tereksekusi tanpa error
├── requirements.txt                 # Daftar dependensi library
├── README.md                        # Laporan metode & hasil (maksimal 1 halaman)
└── .gitignore                       # Mengabaikan file data besar & temporary files
```
