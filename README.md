# MLOps Hands-on 2: End-to-End Model Experimentation & Tracking

[![MLflow](https://img.shields.io/badge/MLflow-3.17.0-blue.svg)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repositori ini memuat implementasi komprehensif **Tugas Praktikum MLOps Minggu 6 — Hands-on 2: Studi Kasus End-to-End: MLflow + Scikit-Learn (Dari Hipotesis hingga Model di Produksi)**, Program Studi Sains Data, Institut Teknologi Sumatera (ITERA).

---

## 📌 1. Latar Belakang & Masalah Bisnis

- **Masalah Klinis**: Membantu dokter memprioritaskan pemeriksaan lanjutan untuk tumor payudara yang kemungkinan ganas (*malignant*).
- **Karakteristik Biaya**: *False Negative* (pasien dengan tumor ganas tetapi terlewat dan divonis jinak) memiliki risiko fatal yang sangat tinggi. Oleh karena itu, metrik pencarian utama difokuskan pada sensitivitas tinggi.
- **Metrik Desain Eksperimen**:
  - **Optimizing Metric**: **Recall** $\rightarrow$ Meminimalkan pasien ganas yang terlewat.
  - **Satisficing Metric**: **Precision $\ge 0.90$** $\rightarrow$ Menjaga ambang batas minimum agar tidak membanjiri dokter dengan alarm palsu (*alert fatigue*).
- **Protokol Evaluasi**:
  - Dataset: *Wisconsin Breast Cancer* (569 sampel, 30 fitur numerik).
  - Target: Dibalik menjadi $1 = \text{Malignant}$, $0 = \text{Benign}$.
  - Split: 80% Training (dievaluasi dengan 5-Fold Stratified Cross-Validation) dan 20% Test (dikunci hingga tahap final).

---

## 🧪 2. Hipotesis Eksperimen

1. **H1**: Model linear sederhana (Logistic Regression) sudah jauh lebih baik daripada baseline dummy (*most frequent*).
2. **H2**: Standardisasi fitur (`StandardScaler`) secara signifikan meningkatkan recall pada model berbasis jarak/margin (Logistic Regression, KNN, SVC).
3. **H3**: Hyperparameter tuning pada Support Vector Classifier (SVC) meningkatkan recall tanpa melanggar satisficing metric $\text{Precision} \ge 0.90$.

---

## 🏗️ 3. Arsitektur Alur Eksperimen (L1 – L9)

```text
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │   L1. Data   │───▶ │ L2. Baseline │───▶ │   L3. Model  │───▶ │  L4. Tuning  │───▶ │ L5. Analisis │
 │ split + hash │     │  Dummy, LR   │     │   selection  │     │ GridSearchCV │     │ search_runs  │
 │  log_input   │     │  (±scaling)  │     │ 5 algoritma  │     │  nested runs │     │ + UI compare │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                                            │
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
 │L9. Reproduce │◀─── │  L8. Serving │◀─── │ L7. Registry │◀─── │ L6. Evaluasi │◀───────────┘
 │ dari run_id  │     │   REST API   │     │ quality gate │     │   test set   │
 │              │     │  @champion   │     │ @challenger  │     │   (sekali)   │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **L1 — Data Management & Versioning**: Ekstraksi dataset, kalkulasi hash MD5 (`data_md5`), stratified split 80:20, serta pencatatan metadata dataset ke MLflow.
2. **L2 — Baseline Models (H1 & H2)**: Uji batas bawah (Dummy Classifier) vs Logistic Regression tanpa normalisasi vs Logistic Regression dengan standardisasi.
3. **L3 — Model Selection (Controlled Experiments)**: Evaluasi 5 model standar (KNN, Decision Tree, Random Forest, Gradient Boosting, SVC RBF) menggunakan 5-Fold Stratified CV identik.
4. **L4 — Hyperparameter Tuning (Nested Runs) (H3)**: Grid search 12 kombinasi parameter $C$ dan $\gamma$ untuk SVC. Setiap kombinasi dicatat sebagai child run dan model terbaik disimpan di parent run.
5. **L5 — Programmatic Selection**: Kueri otomatis via `mlflow.search_runs` dengan filter `metrics.cv_precision_mean >= 0.90` dan pengurutan `metrics.cv_recall_mean DESC`.
6. **L6 — Final Test Set Evaluation**: Model terbaik dievaluasi satu kali pada data uji untuk memastikan tidak terjadi kebocoran informasi (*data snooping*).
7. **L7 — Model Registry & Quality Gate**: Registrasi model `bc-malignant-classifier`, penetapan alias `@challenger`, evaluasi aturan Quality Gate ($\text{Test Precision} \ge 0.90 \land \text{Test Recall} \ge \text{Champion Recall} + 0.005$), dan promosi ke `@champion`.
8. **L8 — Serving dari Registry**: Pemuatan dinamis menggunakan pointer alias `models:/bc-malignant-classifier@champion` serta validasi proteksi *Model Signature* terhadap data rusak.
9. **L9 — Reproducibility & Lineage Audit**: Penelusuran silsilah model produksi kembali ke Run ID, commit Git, hash MD5 data, parameter estimator, dan dependensi lingkungan (*bit-exact audit*).

---

## 📁 4. Struktur Repositori

```text
bc-mlops/
├── data/
│   ├── train.csv                     # Dataset training 80% (di-hash)
│   └── test.csv                      # Dataset testing 20% (terkunci)
├── src/
│   ├── __init__.py
│   ├── data.py                       # L1: Data preparation, hashing & split
│   ├── tracking.py                   # Helper logging MLflow & CV 5-fold
│   ├── train.py                      # L2-L4: Baseline, Selection, Tuning (Nested Runs)
│   ├── evaluate.py                   # L5-L6: Programmatic Search & Test Evaluation
│   ├── register.py                   # L7: Model Registry & Quality Gate
│   ├── serve_test.py                 # L8: Model Serving via @champion & Signature Validation
│   └── reproduce.py                  # L9: Audit Lineage & Reproducibility
├── plots/                            # Artefak gambar OOF dan Test Confusion Matrix
├── reports/                          # Artefak teks Classification Reports
├── generate_pdf_report.py            # Generator Dokumen Laporan Resmi (PDF)
├── main.py                           # Orkestrator eksekusi pipeline L1 - L9
├── requirements.txt                  # Daftar dependensi Python
├── .gitignore                        # Konfigurasi pengecualian git
└── README.md                         # Dokumentasi teknis proyek
```

---

## 🚀 5. Cara Menjalankan

### A. Persiapan Lingkungan

```bash
# Clone repositori
git clone https://github.com/tantywidiyastuti/bc-mlops.git
cd bc-mlops

# Buat virtual environment & install dependensi
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### B. Menjalankan Server MLflow (Opsional jika ingin UI)

```bash
# Terminal 1:
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --port 5000
```
Buka peramban di `http://127.0.0.1:5000` untuk melihat dashboard MLflow UI, perbandingan runs, grafik parallel coordinates, dan Model Registry.

### C. Menjalankan Pipeline End-to-End

```bash
# Terminal 2:
python main.py
```

### D. Generate Dokumen Laporan (PDF)

```bash
python generate_pdf_report.py
```
File laporan akademik resmi akan terbuat di root direktori: `Laporan_HandsOn2_MLOps_Deteksi_Tumor_Ganas.pdf`.

---

## 📊 6. Ringkasan Hasil Eksperimen

| Run Name | Model Family | Stage | CV Recall (Mean ± Std) | CV Precision (Mean ± Std) | CV F1-Score | Status Quality Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `00-dummy-most-frequent` | Dummy | Baseline | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 | ❌ Gagal Precision |
| `01-logreg-noscale` | LogReg | Baseline | 0.9176 ± 0.0461 | 0.9575 ± 0.0381 | 0.9366 | Lolos Filter |
| `02-logreg-scaled` | LogReg | Baseline | 0.9529 ± 0.0388 | 0.9765 ± 0.0321 | 0.9643 | Lolos Filter |
| `03-knn-default` | KNN | Model Selection | 0.9176 ± 0.0423 | 0.9882 ± 0.0235 | 0.9510 | Lolos Filter |
| `04-tree-default` | Tree | Model Selection | 0.8529 ± 0.0632 | 0.8938 ± 0.0489 | 0.8715 | ❌ Gagal Precision |
| `05-rf-default` | RF | Model Selection | 0.9353 ± 0.0357 | 0.9641 ± 0.0354 | 0.9490 | Lolos Filter |
| `06-gb-default` | GB | Model Selection | 0.9471 ± 0.0388 | 0.9647 ± 0.0336 | 0.9553 | Lolos Filter |
| `07-svc-default` | SVC | Model Selection | 0.9471 ± 0.0388 | 0.9765 ± 0.0321 | 0.9612 | Lolos Filter |
| **`10-svc-gridsearch`** | **SVC** | **Tuning (H3)** | **0.9647 ± 0.0287** | **0.9714 ± 0.0360** | **0.9678** | **🏆 Terpilih (@champion)** |

### Hasil Evaluasi Final pada Test Set (L6):
- **Test Recall**: `0.9286` (39 dari 42 sampel tumor ganas berhasil diidentifikasi dengan benar)
- **Test Precision**: `1.0000` (100% presisi sempurna tanpa satupun alarm palsu!)
- **Test F1-Score**: `0.9630`
- **Test ROC-AUC**: `0.9934`

---

## 👩‍💻 Author & Repository

- **Nama**: Tanty Widiyastuti
- **Email**: tantywidiyastuti35551@gmail.com
- **Institusi**: Institut Teknologi Sumatera (ITERA) - Program Studi Sains Data
- **GitHub Repository**: [https://github.com/tantywidiyastuti/bc-mlops](https://github.com/tantywidiyastuti/bc-mlops)
