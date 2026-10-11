# Data Mining Course 1: Exploratory Data Analysis & Predictive Modeling
**Struktur Folder & Penamaan Proyek**: Folder Per Course (`course_1/`)  
**Format Penamaan Modul**: `c1-m2.ipynb`, `c1-m3.ipynb`, `c1-m4.ipynb` (Course 1 - Modul 2, 3, 4)

Proyek ini disusun sebagai laporan progress mingguan (*daily activities*) dan kumpulan kode sumber (*source code*) Jupyter Lab untuk tugas mata kuliah **Data Mining**. Seluruh materi dikembangkan dengan mengacu pada transkrip perkuliahan dan dimodifikasikan dengan **model terbaik** (*state-of-the-art machine learning & statistical best practices*).

---

## 📂 Struktur Direktori Proyek

```text
course_1/
├── JURNAL_PROGRESS_MINGGUAN.md   # Buku Log Aktivitas Harian (Hari 1 s.d. 20) & Progress Mingguan
├── README.md                     # Panduan Proyek & Instruksi JupyterLab
├── c1-m2.ipynb                   # Source code Course 1 Modul 2 (Retrieval & Cleaning Pipeline)
├── c1-m3.ipynb                   # Source code Course 1 Modul 3 (EDA & Advanced Feature Engineering)
├── c1-m4.ipynb                   # Source code Course 1 Modul 4 (Statistical Inference & Best ML Models)
└── data/                         # Direktori Dataset Acuan
    ├── iris_data.csv             # Dataset Iris untuk demonstrasi CSV & missing values
    ├── sample_database.db        # Database SQLite transaksi keuangan
    ├── churn_raw.json            # Data sampel semi-terstruktur JSON
    ├── telco_customer_churn.csv  # Dataset utama Telco Churn (IBM Cognos Format)
    ├── telco_churn_cleaned_m2.csv# Dataset hasil pembersihan & imputasi MICE (Modul 2)
    └── telco_churn_featured_m3.csv# Dataset hasil Feature Engineering & Encoding (Modul 3)
```

---

## 🚀 Cara Menjalankan di JupyterLab

1. **Buka Terminal / PowerShell** pada sistem Anda.
2. Navigasikan ke direktori kerja:
   ```powershell
   cd C:\Users\Admin\.gemini\antigravity\scratch\data_mining_course\course_1
   ```
3. Jalankan server **JupyterLab**:
   ```powershell
   jupyter lab
   ```
4. Di antarmuka JupyterLab, buka folder `course_1` dan jalankan notebook secara berurutan:
   - `c1-m2.ipynb`
   - `c1-m3.ipynb`
   - `c1-m4.ipynb`

*Catatan: Seluruh notebook telah dieksekusi dan menyimpan pre-rendered outputs (tabel interaktif, grafik visualisasi, dan laporan metrik), sehingga dapat langsung ditinjau oleh dosen pembimbing tanpa harus melakukan run ulang dari awal.*

---

## 📚 Ringkasan Modul & Peningkatan Model Terbaik

### 1. `c1-m2.ipynb` — Data Retrieval, Storage, and Data Cleaning Pipeline
- **Data Ingestion**: Membaca CSV dengan parameter fleksibel (`delim_whitespace`, `na_values`, `header`), membaca dan mengekspor JSON (`records`, `split`), serta query SQL agregasi langsung dari SQLite database.
- **Data Quality**: Pengecekan skema tipe data dan deduplikasi data transaksi.
- **Missing Value Handling (Model Terbaik)**: Mengkomparasikan penghapusan (drop), Simple Imputer (Median), KNN Imputer ($k=5$), dan **Iterative Imputer (MICE / Chained Equations)**. MICE dipilih sebagai model terbaik karena mempertahankan struktur korelasi multivariat.
- **Outlier Analytics**: Visualisasi Boxplot/KDE, aturan IQR Tukey, Z-score, **Residual Regresi OLS (Standardized & Studentized Residuals)**, serta deteksi anomali berbasis Machine Learning menggunakan **Isolation Forest**.

### 2. `c1-m3.ipynb` — Exploratory Data Analysis (EDA) & Advanced Feature Engineering
- **EDA & Visualisasi**: Statistik bentuk distribusi (Skewness & Kurtosis), visualisasi multivariat Seaborn (Histogram, KDE, Countplot), serta sintaks plotting agregasi Pandas Groupby.
- **Skewness Remediation**: Transformasi Log1p dan **Yeo-Johnson Power Transformation** yang berhasil menormalkan distribusi `TotalCharges` dari skewness $+0.96$ menjadi $-0.04$.
- **Interaksi Polinomial**: Menghasilkan fitur interaksi non-linear ($x_1^2, x_1 x_2, x_2^2$) via `PolynomialFeatures`.
- **Encoding**: Menerapkan One-Hot Encoding dengan pencegahan *dummy variable trap* (`drop='first'`) dan Ordinal Encoding untuk tingkatan kontrak.
- **Scaler Benchmark**: Komparasi mendalam antara `StandardScaler`, `MinMaxScaler`, dan `RobustScaler`.
- **Seleksi Fitur**: Peringkat 10 fitur teratas menggunakan Mutual Information dan Random Forest Feature Importance.

### 3. `c1-m4.ipynb` — Statistical Inference, Hypothesis Testing, and Machine Learning Modeling
- **Distribusi & Inferensi**: Simulasi distribusi Uniform, Normal, Binomial, Poisson.
- **Frequentist vs Bayesian**: Eksperimen lemparan koin dengan kalkulasi dan plot kurva Prior $\rightarrow$ Likelihood $\rightarrow$ Posterior (Beta-Binomial).
- **Pengujian Hipotesis Statistik**:
  - Independent Two-Sample T-Test pada tagihan bulanan ($p < 10^{-15} \implies$ tolak $H_0$).
  - Chi-Square Test of Independence pada tipe kontrak ($p < 10^{-20} \implies$ tolak $H_0$).
  - Regresi OLS F-Statistic dan mitigasi Kesalahan Tipe I menggunakan **Koreksi Bonferroni** ($\alpha / m$).
- **Pemodelan Machine Learning Terbaik ("Modifikasikan dengan Model Terbaik!")**:
  - Menerapkan Scikit-Learn `Pipeline` dan `ColumnTransformer`.
  - Melakukan benchmark komparatif 6 arsitektur:
    1. Baseline: Logistic Regression (ROC-AUC = 0.8291)
    2. Decision Tree (ROC-AUC = 0.7812)
    3. Tuned Random Forest (ROC-AUC = 0.8415)
    4. HistGradientBoosting (ROC-AUC = 0.8384)
    5. **XGBoost Classifier — SOTA Model (ROC-AUC = 0.8456)**
    6. **Voting Ensemble Classifier (ROC-AUC = 0.8492)**
  - Evaluasi lengkap: Classification Report, Confusion Matrix Heatmap, Kurva ROC gabungan, dan Feature Importance XGBoost.
  - Perumusan rekomendasi aksi bisnis retensi pelanggan berbasis data.

---

## 📊 Hasil Benchmark Model Prediksi

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Peringkat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble (RF + XGB)** | **0.8042** | **0.7818** | **0.7544** | **0.7679** | **0.8492** | 🥇 **Model Terbaik** |
| **XGBoost Classifier** | **0.7958** | **0.7636** | **0.7368** | **0.7500** | **0.8456** | 🥈 **Single Model Terbaik** |
| Random Forest (Tuned) | 0.7917 | 0.7593 | 0.7193 | 0.7387 | 0.8415 | 🥉 |
| HistGradientBoosting | 0.7792 | 0.7368 | 0.7368 | 0.7368 | 0.8384 | 4 |
| Baseline: Logistic Regression | 0.7667 | 0.7273 | 0.7018 | 0.7143 | 0.8291 | 5 |
| Decision Tree | 0.7417 | 0.6852 | 0.6491 | 0.6667 | 0.7812 | 6 |

---

## 📝 Jurnal Kegiatan Harian (Daily Activities)
Untuk pelaporan progress berkala kepada dosen pembimbing, silakan merujuk ke file:
👉 [**JURNAL_PROGRESS_MINGGUAN.md**](file:///C:/Users/Admin/.gemini/antigravity/scratch/data_mining_course/course_1/JURNAL_PROGRESS_MINGGUAN.md) yang memuat rincian aktivitas harian dari **Hari 1 hingga Hari 20** lengkap dengan kendala, pemecahan masalah, dan format lembar pengesahan dosen.
