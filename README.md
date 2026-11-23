# Deteksi Penipuan Pembukaan Rekening Bank (Kasus A)
## Studi Komparatif: LightGBM vs. XGBoost

Repository ini berisi implementasi dan analisis komparatif model *machine learning* untuk mendeteksi penipuan (*fraud*) pada pembukaan rekening bank digital menggunakan dataset **Bank Account Fraud (BAF) varian Base (Bulan 0–3)**.

---

## 1. Metode yang Digunakan

* **Algoritma yang Diuji:** Dua model *Gradient Boosted Decision Trees* (GBDT) *state-of-the-art*: **LightGBM** (*leaf-wise splitting* dengan GOSS) dan **XGBoost** (*depth-wise splitting* dengan regularisasi terintegrasi).
* **Konfigurasi Setara (*Fair Benchmark*):** 300 estimators, learning rate 0,05, kapasitas kedalaman sepadan (`num_leaves=31` vs `max_depth=6`).
* **Pra-pemrosesan Data:** *Ordinal Encoding* pada 5 variabel kategorikal (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) dengan fitting yang hanya dilakukan pada data latih.
* **Kalibrasi Ambang Batas (*Decision Threshold*):** Optimasi batas probabilitas melalui kurva ROC untuk mengunci *False Positive Rate* (FPR) pada batas operasional $\le 5\%$.

---

## 2. Pembagian Data Latih dan Validasi (*Data Splitting*)

Dataset dipartisi menggunakan pendekatan **Temporal (Out-of-Time / OOT)** berdasarkan kolom waktu `month`:
* **Data Latih (Train):** Bulan 0, 1, dan 2 (**397.039 baris** / 72,5%)
* **Data Validasi (Validation):** Bulan 3 (**150.936 baris** / 27,5%)

**Rasional Pemilihan Strategi:**
Pola dan teknik penipuan perbankan terus berevolusi seiring waktu (*concept drift*). Penggunaan *random split* berisiko menimbulkan kebocoran informasi temporal (*lookahead bias*), di mana model memanfaatkan pola masa depan untuk memprediksi masa lalu. Pendekatan *temporal split* merefleksikan kondisi operasional riil perbankan: model dilatih pada riwayat masa lalu (bulan 0–2) untuk mengevaluasi calon nasabah baru pada periode berikutnya (bulan 3).

---

## 3. Skor di Data Validasi (Bulan 3) & Hasil Komparatif

Evaluasi dilakukan pada data validasi bulan 3 ($N = 150.936$) dengan syarat nasabah wajar yang keliru ditandai dibatasi maksimal 5% (*FPR $\le$ 5%*).

| Metrik / Indikator Evaluasi | LightGBM (Terpilih) | XGBoost | Keterangan / Analisis |
|---|:---:|:---:|---|
| **ROC-AUC Score** | **0.9100** | **0.9109** | Performa pemisahan kelas setara ($\Delta = 0,0009$) |
| **Batas Ambang (*Threshold*)** | **0.0523** | **0.0572** | Ambang batas keputusan pada syarat FPR $\le$ 5% |
| **Nasabah Wajar Tertandai (*FPR*)** | **4.98%** | **4.96%** | **Keduanya memenuhi syarat operasional ($\le 5.0\%$)** |
| **Penipu Tertangkap (*TPR*)** | **56.54% (787 fraud)** | **56.90% (792 fraud)** | Selisih marjinal hanya 5 fraud dari total 1.392 |
| **Durasi Pelatihan (*Latency*)** | **~9.84 detik** | **~13.73 detik** | **LightGBM ~40% lebih cepat** |
| **Ukuran Berkas Model** | **1.00 MB** | **1.20 MB** | LightGBM lebih ringkas dan portabel |

### Rincian Matriks Kebingungan (*Confusion Matrix*)

| Kuadran | LightGBM | XGBoost | Interpretasi Operasional |
|---|:---:|:---:|---|
| **True Negative (TN)** | 142.093 | 142.121 | Nasabah wajar yang langsung lolos otomatis |
| **False Positive (FP)** | 7.451 | 7.423 | Nasabah wajar yang keliru ditandai verifikasi manual |
| **False Negative (FN)** | 605 | 600 | Kasus fraud yang lolos deteksi pada ambang batas operasional |
| **True Positive (TP)** | 787 | 792 | **Kasus fraud yang berhasil dicegat oleh sistem** |

---

## 4. Analisis Keunggulan, Kekurangan, dan Alasan Pemilihan Model

| Aspek Evaluasi | LightGBM | XGBoost |
|---|---|---|
| **Keunggulan (*Pros*)** | Komputasi sangat cepat (*leaf-wise splitting* + GOSS), memori rendah (*histogram binning*), ukuran file ringan (~1 MB), sangat ramah untuk *retraining* berkala. | Regulasi L1/L2 terintegrasi mendalam pada *loss function*, struktur pohon seimbang (*depth-wise*), skor deteksi marjinal sedikit lebih tinggi (+0,36% TPR). |
| **Kekurangan (*Cons*)** | Perlu batasan kedalaman daun agar tidak overfit pada data kecil (aman pada data besar ratusan ribu baris). | Waktu pelatihan lebih lambat pada volume data besar, konsumsi memori matriks gradien lebih tinggi, ukuran model lebih besar. |

### Justifikasi Pemilihan Model (*Champion Model*): **LightGBM**
Meskipun XGBoost menangkap 5 fraud lebih banyak (+0,36% TPR), **LightGBM dipilih sebagai model produksi utama** dengan pertimbangan operasional MLOps perbankan:
1. **Paritas Kinerja Deteksi:** Selisih performa TPR sangat kecil (56,54% vs 56,90%) dan skor AUC identik (0,9100 vs 0,9109). Keduanya sama-sama berhasil menekan FPR di bawah 5%.
2. **Efisiensi Siklus Hidup MLOps (*Retraining Efficiency*):** Pada sistem deteksi fraud perbankan, model wajib dilatih ulang secara berkala (mingguan/bulanan) akibat *concept drift*. Efisiensi LightGBM yang **~40% lebih cepat** menghemat biaya komputasi *cloud* dan mempercepat *continuous training pipeline*.
3. **Ukuran Artefak & Portabilitas:** Ukuran 1.00 MB memudahkan distribusi ke dalam layanan mikro (*microservices*) tanpa dependensi Git LFS.

---

## 5. Panduan Menjalankan & Memuat Model

1. **Instalasi Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Eksekusi Notebook:**
   Jalankan `Bank_Fraud_Model_Prediction.ipynb` dari atas ke bawah. Berkas data akan dimuat dari direktori lokal atau diunduh otomatis.
3. **Memuat Model Tersimpan:**
   ```python
   import joblib

   bundle = joblib.load('model/fraud_detector_lgbm.joblib')
   model = bundle['model']
   threshold = bundle['threshold']
   print(f"Model terpilih ({bundle['selected_model_name']}) dimuat dengan threshold: {threshold:.4f}")
   ```

---

## 6. Struktur Repositori

```text
.
├── model/
│   └── fraud_detector_lgbm.joblib       # Artefak model terpilih tersimpan (~1 MB)
├── Bank_Fraud_Model_Prediction.ipynb    # Notebook komparatif lengkap (eksekusi tanpa error)
├── requirements.txt                     # Daftar dependensi (termasuk lightgbm dan xgboost)
├── README.md                            # Laporan metode, perbandingan, dan analisis (1 halaman)
└── .gitignore                           # Konfigurasi pengabaian berkas data besar & temporary
```
