# BUKU LOG AKTIVITAS HARIAN & LAPORAN PROGRESS MINGGUAN (DAILY ACTIVITIES)
## MATA KULIAH: DATA MINING — PROGRAM STUDI TEKNIK INFORMATIKA

---

### IDENTITAS MAHASISWA & KEGIATAN
- **Nama Mahasiswa**     : **Nayyara Rauf Ghani Al Khayri**
- **NIM / ID**           : **A11.2024.16055**
- **Program Studi**      : **Teknik Informatika**
- **Mata Kuliah**        : **Data Mining**
- **Dosen Pengampu**     : **Abu Salam, M.Kom**
- **Target Capaian**     : Penguasaan End-to-End Pipeline Data Mining Mandiri & Modular (Data Ingestion Multi-Sumber, Relational SQLite Storage, MICE Imputation, Multivariate Isolation Forest, Automated Yeo-Johnson Transformation, Comparative Encoding, VIF Multicollinearity Diagnostics, Mutual Information, Uji Hipotesis Statistik Formal Chi-Square/ANOVA/T-Test dengan Bonferroni, Stratified 5-Fold CV Machine Learning Benchmark, Kurva ROC-AUC & PR-AUC, serta Optimasi Finansial Cost-Sensitive Matrix).
- **Periode Pelaksanaan**: 4 Minggu (20 Hari Kerja Efektif)
- **Struktur Lingkungan**: Local Environment (VS Code + Python 3.13 Virtual Environment `.venv` + SQLite Database `telco_churn.db`)
- **Format Penamaan Modul Profesional**:
  1. `01_data_ingestion_and_cleaning.ipynb`
  2. `02_eda_and_feature_engineering.ipynb`
  3. `03_hypothesis_testing_and_modeling.ipynb`

---

## RINGKASAN PROGRESS MINGGUAN (EXECUTIVE SUMMARY)

| Minggu | Modul Terkait | Fokus Utama & Topik Kegiatan | Output / Deliverable | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Minggu 1** | `01_data_ingestion_and_cleaning.ipynb` (Bagian 1) | Data Ingestion Multi-Format (CSV, JSON, SQLite), Skema Relasional, Inisialisasi Database SQLite Lokal `telco_churn.db` (`raw_customers`), dan Audit Kualitas Data. | Script `src/ingestion.py`, `src/database.py`, Tabel SQLite `raw_customers`, Laporan Audit Skema. | **SELESAI (100%)** |
| **Minggu 2** | `01_data_ingestion_and_cleaning.ipynb` (Bagian 2) | Data Cleaning Tingkat Lanjut: Komparasi Imputasi MICE (`IterativeImputer`) vs Simple Median vs KNN, serta Deteksi Outlier Multivariat (Tukey IQR vs Isolation Forest vs OLS Residuals). | Script `src/cleaning.py`, Notebook `01_data_ingestion_and_cleaning.ipynb`, Tabel SQLite `cleaned_customers`. | **SELESAI (100%)** |
| **Minggu 3** | `02_eda_and_feature_engineering.ipynb` | Exploratory Data Analysis (EDA) Bentuk Distribusi (Skewness & Kurtosis), Remidiasi Skewness Otomatis (Yeo-Johnson), Comparative Encoding (Ordinal & One-Hot drop-first), VIF, dan Mutual Information. | Script `src/eda.py`, `src/feature_engineering.py`, Notebook `02_eda_and_feature_engineering.ipynb`, Tabel SQLite `featured_customers`. | **SELESAI (100%)** |
| **Minggu 4** | `03_hypothesis_testing_and_modeling.ipynb` | Inferensi Statistik Formal (Chi-Square, One-Way ANOVA, Welch's T-Test, Bonferroni), Stratified 5-Fold CV, GridSearch Tuning (Logistic Regression, Random Forest, XGBoost, Voting Ensemble), Kurva ROC/PR-AUC, dan Cost-Sensitive Threshold Optimization ($\tau^*$). | Script `src/hypothesis_testing.py`, `src/modeling.py`, Notebook `03_hypothesis_testing_and_modeling.ipynb`, Tabel SQLite `model_metrics`, Rekomendasi Bisnis. | **SELESAI (100%)** |

---

## LOG AKTIVITAS HARIAN (DAILY ACTIVITIES BREAKDOWN)

### MINGGU 1: DATA INGESTION MULTI-SUMBER, RELATIONAL STORAGE & AUDIT KUALITAS
**Modul Acuan**: `01_data_ingestion_and_cleaning.ipynb`

#### Hari 1 (Senin): Pengenalan Silabus, Arsitektur Modular, dan Konfigurasi Local Virtual Environment
- **Aktivitas**:
  - Menyusun rancangan arsitektur penambangan data end-to-end terstruktur (*Data Ingestion $\rightarrow$ Advanced Cleaning $\rightarrow$ EDA $\rightarrow$ Feature Engineering $\rightarrow$ Statistical Hypothesis Testing $\rightarrow$ Predictive Modeling $\rightarrow$ Cost-Sensitive Optimization*).
  - Mengonfigurasi lingkungan kerja lokal menggunakan VS Code dan virtual environment Python 3.13 (`.venv`).
  - Menginstal pustaka inti: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `xgboost`, `matplotlib`, `seaborn`, `nbformat`, dan `nbclient`.
  - Merancang struktur direktori profesional dengan folder modular `src/`, `data/`, `logs/`, dan `scripts/`.
- **Hasil/Output**:
  - Virtual environment `.venv` terisolasi dan terverifikasi.
  - Modul dasar `src/config.py` dan `src/logger.py` terpasang dengan handler logging konsol dan file `logs/pipeline.log`.

#### Hari 2 (Selasa): Ekstraksi Data Flat Files (CSV & Delimitasi Fleksibel)
- **Aktivitas**:
  - Mengembangkan fungsi `load_csv_data()` pada `src/ingestion.py` dengan penanganan otomatis terhadap missing representation (`"", " ", "NA", "null"`).
  - Menguji pembacaan file `data/iris_data.csv` untuk validasi parsing parameter.
  - Membaca dataset studi kasus utama: `data/telco_customer_churn.csv` (1.200 baris, 17 kolom).
- **Hasil/Output**:
  - Parser CSV yang tangguh terhadap inkonsistensi spasi dan format numerik.
  - Verifikasi pembacaan awal 1.200 pelanggan telekomunikasi.

#### Hari 3 (Rabu): Ingestion Semi-Structured JSON & Relational Legacy SQLite
- **Aktivitas**:
  - Mempelajari representasi data semi-terstruktur dari API menggunakan format JSON.
  - Mengembangkan fungsi `load_json_data()` untuk membaca `data/churn_raw.json`.
  - Mengembangkan fungsi `query_legacy_sqlite()` untuk mengekstraksi data transaksi keuangan dari `data/sample_database.db` menggunakan query SQL teragregasi.
- **Hasil/Output**:
  - Fungsi ingestion JSON dan SQLite terintegrasi pada `src/ingestion.py`.
  - Kemampuan integrasi data heterogen dari berbagai format penyimpanan.

#### Hari 4 (Kamis): Inisialisasi Database Relasional SQLite Lokal (`telco_churn.db`)
- **Aktivitas**:
  - Mengembangkan modul manajemen database `src/database.py` berbasis pustaka `sqlite3`.
  - Merancang skema tabel untuk database operasional `data/telco_churn.db`:
    - Tabel `raw_customers`: Menyimpan data mentah hasil ingestion.
    - Tabel `cleaned_customers`: Menyimpan data terbersihkan.
    - Tabel `featured_customers`: Menyimpan fitur hasil rekayasa.
    - Tabel `model_metrics`: Menyimpan riwayat evaluasi performa dan hyperparameter model.
  - Menyimpan seluruh data mentah Telco Churn ke dalam tabel `raw_customers`.
- **Hasil/Output**:
  - Database SQLite lokal `telco_churn.db` berhasil diinisialisasi dan diisi 1.200 baris data mentah.
  - Integritas data terjamin melalui format transaksi relasional.

#### Hari 5 (Jumat): Audit Kualitas Data Komprehensif (*Data Quality Audit*)
- **Aktivitas**:
  - Mengembangkan fungsi `validate_data_quality()` pada `src/ingestion.py`.
  - Melakukan audit menyeluruh terhadap:
    - Keunikan *Primary Key*: Mengecek duplikasi pada `customerID` (ditemukan 0 duplikat).
    - Nilai Hilang (*Missing Values*): Mendeteksi 4 nilai null pada atribut tagihan `TotalCharges`.
    - Konsumsi memori dan kelengkapan skema data.
- **Hasil/Output**:
  - Laporan audit kualitas data formal tercatat di log pipeline.
  - Identifikasi prioritas pembersihan untuk Minggu ke-2.

---

### MINGGU 2: ADVANCED DATA CLEANING, IMPUTASI MICE & MULTIVARIATE OUTLIER
**Modul Acuan**: `01_data_ingestion_and_cleaning.ipynb`

#### Hari 6 (Senin): Analisis Mekanisme Missing Values & Keterbatasan Median Imputation
- **Aktivitas**:
  - Menganalisis mekanisme nilai hilang pada `TotalCharges` (Missing at Random / MAR).
  - Menguji kelemahan imputasi median univariat (*Simple Imputation*).
  - Menemukan bahwa imputasi median menyuntikkan nilai konstan yang merusak struktur korelasi alamiah antara `TotalCharges`, `tenure`, dan `MonthlyCharges`.
- **Hasil/Output**:
  - Pemahaman teoritis bahwa imputasi univariat menyebabkan penyempitan varians (*variance shrinkage*) dan bias kovariansi.

#### Hari 7 (Selasa): Eksperimen Komparasi: Simple Median vs KNN Imputer ($k=5$)
- **Aktivitas**:
  - Mengimplementasikan `KNNImputer(n_neighbors=5)` pada modul `src/cleaning.py`.
  - Menghitung jarak Euclidean antar pelanggan untuk mengestimasi nilai `TotalCharges` yang hilang berdasarkan tetangga terdekat.
  - Mengamati pergeseran nilai statistik (Mean, Standar Deviasi, Median) pasca-imputasi KNN.
- **Hasil/Output**:
  - KNN Imputer terbukti lebih baik daripada Simple Median dalam menangkap kedekatan lokal, namun sensitif terhadap skala fitur.

#### Hari 8 (Rabu): Penerapan Model Imputasi Terbaik: MICE (IterativeImputer)
- **Aktivitas**:
  - Mengkaji teori *Multivariate Imputation by Chained Equations* (MICE).
  - Mengimplementasikan `IterativeImputer` berbasis regresi Bayesian Ridge (`max_iter=15`) pada `src/cleaning.py`.
  - Melakukan benchmark komparatif tiga arah (Median vs KNN vs MICE) menggunakan fungsi `compare_imputation_strategies()`.
- **Hasil/Output**:
  - MICE terbukti sebagai metode terbaik karena berhasil mempertahankan koefisien korelasi terhadap `tenure` ($r = 0.8256$) dan `MonthlyCharges` ($r = 0.6512$), identik dengan data asli tanpa missing values.
  - Nilai null pada `TotalCharges` berhasil diimputasi secara elegan tanpa distorsi kovariansi.

#### Hari 9 (Kamis): Deteksi Outlier Univariat (Tukey IQR) & Residual Regresi OLS
- **Aktivitas**:
  - Mengimplementasikan deteksi pencilan klasik Interquartile Range (IQR): $[Q_1 - 1.5 \times IQR, \, Q_3 + 1.5 \times IQR]$.
  - Menganalisis kelemahan IQR univariat yang tidak mendeteksi outlier pada `TotalCharges` karena rentang data yang lebar.
  - Mengembangkan deteksi anomali regresi menggunakan *Studentized Residuals* OLS ($|r_i| > 2.5$) pada formula $\text{TotalCharges} \sim \text{tenure} + \text{MonthlyCharges}$.
- **Hasil/Output**:
  - Fungsi `detect_outliers_iqr()` dan `detect_outliers_ols_residuals()` pada `src/cleaning.py`.
  - Pemetaan observasi yang menyimpang dari hubungan linear tagihan.

#### Hari 10 (Jumat): Deteksi Outlier Multivariat via Isolation Forest & Persistensi Clean Store
- **Aktivitas**:
  - Mengimplementasikan algoritma Machine Learning unsupervised **Isolation Forest** (`contamination=0.03`, `n_estimators=150`) untuk deteksi anomali multidimensi.
  - Algoritma berhasil mengidentifikasi 36 pelanggan anomali (3.0%) tanpa memotong data (*no truncation bias*), melainkan menandainya melalui fitur `anomaly_score`.
  - Menjalankan `run_complete_cleaning_pipeline()` dan menyimpan dataset bersih ke tabel SQLite `cleaned_customers` serta file `data/processed/telco_cleaned.csv`.
- **Hasil/Output**:
  - Visualisasi dual-panel anomali multidimensi dan distribusi skor anomali.
  - Tabel SQLite `cleaned_customers` terverifikasi bersih (1.200 baris, 0 missing value).
  - Notebook `01_data_ingestion_and_cleaning.ipynb` tereksekusi dan ter-render sempurna.

---

### MINGGU 3: EXPLORATORY DATA ANALYSIS (EDA) & REKAYASA FITUR (FEATURE ENGINEERING)
**Modul Acuan**: `02_eda_and_feature_engineering.ipynb`

#### Hari 11 (Senin): Audit Bentuk Distribusi (Skewness & Kurtosis) & Uji Normalitas
- **Aktivitas**:
  - Membaca dataset bersih dari SQLite `cleaned_customers`.
  - Mengembangkan fungsi `calculate_distribution_metrics()` pada `src/eda.py` untuk menghitung Mean, Std, Median, IQR, Skewness, Kurtosis, dan uji normalitas D'Agostino-Pearson.
  - Mengidentifikasi bahwa `TotalCharges` mengalami kemiringan positif (*moderately positively skewed*) sebesar **+0.546**, sedangkan `tenure` dan `MonthlyCharges` relatif simetris.
- **Hasil/Output**:
  - Tabel audit parameter bentuk distribusi fitur kontinu.
  - Visualisasi multi-histogram dengan kurva densitas KDE dan badge nilai kemiringan.

#### Hari 12 (Selasa): Segmentasi Bivariate Churn & Analisis Korelasi Pearson
- **Aktivitas**:
  - Mengembangkan fungsi visualisasi `plot_categorical_churn_breakdown()` pada `src/eda.py`.
  - Menganalisis sebaran Churn pada fitur strategis: `Contract`, `InternetService`, `TechSupport`, dan `PaymentMethod`.
  - Menampilkan heatmap matriks korelasi Pearson dengan penutupan segitiga atas (*upper triangle mask*).
- **Hasil/Output**:
  - Temuan kunci: Pelanggan dengan kontrak bulanan (*Month-to-month*) dan internet Fiber Optic tanpa fasilitas TechSupport mendominasi angka pembatalan langganan.
  - Grafik korelasi Pearson mengonfirmasi kolinearitas antara `tenure` dan `TotalCharges` ($r = 0.826$).

#### Hari 13 (Rabu): Remidiasi Skewness Otomatis via Yeo-Johnson Power Transformation
- **Aktivitas**:
  - Mengembangkan fungsi `auto_skewness_transform()` pada `src/feature_engineering.py`.
  - Menerapkan transformasi daya **Yeo-Johnson** secara otomatis pada fitur dengan $|\text{skewness}| > 0.5$.
  - Membandingkan kurva distribusi sebelum dan sesudah transformasi.
- **Hasil/Output**:
  - Koefisien kemiringan `TotalCharges` berhasil direduksi dari **+0.546** menjadi **-0.139** (turun 0.407 poin mendekati distribusi normal).
  - Varians kesalahan stabil (*homoskedasticity*), sangat menguntungkan bagi algoritma berbasis gradien.

#### Hari 14 (Kamis): Skema Encoding Komparatif (Ordinal & One-Hot dengan Proteksi Dummy Trap)
- **Aktivitas**:
  - Mengembangkan fungsi `encode_categorical_features()` pada `src/feature_engineering.py`.
  - Menerapkan **Ordinal Encoding** pada variabel `Contract` (`Month-to-month: 0, One year: 1, Two year: 2`) guna merefleksikan hierarki komitmen pelanggan.
  - Menerapkan **One-Hot Encoding** pada 10 fitur nominal dengan parameter `drop='first'` guna mengeliminasi jebakan multikolinearitas (*dummy variable trap*).
  - Mengonversi target `Churn` ke format biner integer (`{0, 1}`).
- **Hasil/Output**:
  - Matriks fitur bertambah terstruktur menjadi 22 kolom prediktor siap latih.
  - Matriks desain terverifikasi *full-rank* (bebas singularitas).

#### Hari 15 (Jumat): Audit Multikolinearitas (VIF), Seleksi Fitur, dan Persistensi Featured Store
- **Aktivitas**:
  - Mengembangkan fungsi `calculate_vif()` untuk menghitung *Variance Inflation Factor* pada fitur prediktor kontinu.
  - Mengembangkan fungsi `compute_feature_importance_ranking()` yang menggabungkan dua paradigma: *Mutual Information* ($I(X; Y)$) dan *Random Forest Gini Impurity*.
  - Menjalankan `run_complete_feature_engineering_pipeline()` dan menyimpan dataset ke tabel SQLite `featured_customers` serta `data/processed/telco_featured.csv`.
- **Hasil/Output**:
  - Evaluasi VIF mendasari keputusan penggunaan penalti $L_2$ Ridge pada Logistic Regression.
  - Peringkat 10 fitur teratas membuktikan `Contract`, `tenure`, dan `MonthlyCharges` sebagai fitur paling diskriminatif.
  - Tabel SQLite `featured_customers` tersimpan dan notebook `02_eda_and_feature_engineering.ipynb` selesai dieksekusi 100%.

---

### MINGGU 4: STATISTICAL HYPOTHESIS TESTING, ML BENCHMARK & COST-SENSITIVE OPTIMIZATION
**Modul Acuan**: `03_hypothesis_testing_and_modeling.ipynb`

#### Hari 16 (Senin): Perumusan & Eksekusi Uji Hipotesis Formal (Chi-Square & ANOVA)
- **Aktivitas**:
  - Mengembangkan modul inferensi statistik `src/hypothesis_testing.py`.
  - **Uji 1: Pearson's Chi-Square Test of Independence**:
    - Menguji $H_0$: Tipe Kontrak independen dari Churn.
    - Menghitung derajat kebebasan, statistik $\chi^2$, p-value, dan ukuran efek *Cramer's V*.
  - **Uji 2: One-Way ANOVA**:
    - Menguji $H_0$: Rata-rata biaya bulanan identik di semua kelompok kontrak.
    - Menghitung F-statistic, p-value, dan ukuran efek *Eta-Squared* ($\eta^2$).
- **Hasil/Output**:
  - Chi-Square Test: $\chi^2 = 143.88$, $p = 5.71 \times 10^{-32} < 0.05$, Cramer's $V = 0.3463$ (*Strong Association*) $\implies$ Tolak $H_0$.
  - Interpretasi bisnis formal tercatat otomatis pada fungsi inferensi.

#### Hari 17 (Selasa): Welch's Two-Sample T-Test & Koreksi Bonferroni untuk Pengujian Jamak
- **Aktivitas**:
  - **Uji 3: Welch's Two-Sample Independent T-Test**:
    - Menguji perbedaan masa langganan (`tenure`) antara pelanggan churn dan pelanggan setia.
    - Menghitung $t$-statistic, p-value, dan ukuran efek *Cohen's d*.
  - **Koreksi Bonferroni**:
    - Menghitung ambang signifikansi terkoreksi: $\alpha_{adj} = \frac{0.05}{3} = 0.0167$ guna mengontrol *Family-Wise Error Rate* (FWER).
- **Hasil/Output**:
  - Welch's T-Test: $t = 12.39$, $p < 10^{-20}$, Cohen's $d = 0.72$ (*Large Effect*) $\implies$ Tolak $H_0$. Pelanggan churn memiliki masa langganan jauh lebih singkat.
  - Seluruh hipotesis inti tetap terbukti signifikan secara statistik pasca-koreksi Bonferroni.

#### Hari 18 (Rabu): Stratified 5-Fold Cross Validation Setup & Hyperparameter Tuning
- **Aktivitas**:
  - Mengembangkan modul pemodelan prediktif `src/modeling.py`.
  - Membagi dataset menggunakan `prepare_modeling_data()` (Stratified 80% Train, 20% Holdout Test).
  - Melakukan tuning hyperparameter menggunakan `StratifiedKFold(n_splits=5)` dan `GridSearchCV` untuk tiga arsitektur:
    1. Baseline: Logistic Regression (tuning parameter penalti $L_2$ $C$).
    2. Random Forest Classifier (tuning `max_depth`, `min_samples_split`, `n_estimators`).
    3. XGBoost Classifier (tuning `scale_pos_weight`, `max_depth`, `learning_rate`, `subsample`).
  - Membangun model Soft-Voting Ensemble (Random Forest + XGBoost).
- **Hasil/Output**:
  - Hyperparameter terbaik ditemukan tanpa *data leakage* dan dengan pembobotan kelas seimbang.

#### Hari 19 (Kamis): Multi-Model Benchmark Komparatif (ROC-AUC & PR-AUC Curves)
- **Aktivitas**:
  - Mengevaluasi seluruh model pada holdout test set (240 observasi tidak terlihat).
  - Menghitung metrik lengkap: Akurasi, Presisi, Recall, F1-Score, ROC-AUC, dan PR-AUC (*Average Precision Score*).
  - Membuat visualisasi kurva ROC dan kurva Precision-Recall (PR) berdampingan, visualisasi matriks konfusi, dan feature importance XGBoost.
- **Hasil/Output**:
  - Tabel perbandingan komparatif model pada holdout test set.
  - Random Forest dan Voting Ensemble mencapai ROC-AUC $\approx 0.705$ dan PR-AUC $\approx 0.639$, melampaui garis dasar acak (*no-skill baseline* = 0.45) secara signifikan.

#### Hari 20 (Jumat): Optimasi Ambang Batas Sensitif Biaya (Cost-Sensitive Matrix) & Finalisasi
- **Aktivitas**:
  - Memformulasikan matriks kerugian finansial riil:
    $$\text{Total Kerugian} = (FN \times \$500) + (FP \times \$50)$$
  - Mengembangkan fungsi `optimize_decision_threshold()` untuk menemukan ambang batas keputusan optimal $\tau^*$ yang meminimalkan total kerugian finansial.
  - Membandingkan biaya pada ambang default ($\tau = 0.50$) vs ambang optimal ($\tau^* \approx 0.18 - 0.24$).
  - Mencatat seluruh metrik evaluasi ke tabel SQLite `model_metrics`.
  - Menyusun 4 pilar rekomendasi strategis retensi bagi manajemen.
- **Hasil/Output**:
  - Optimasi ambang batas keputusan terbukti menghemat kerugian finansial perusahaan lebih dari **\$14.000 hingga \$16.000 (> 65% reduksi biaya)**.
  - Tabel SQLite `model_metrics` terisi riwayat pengujian.
  - Notebook `03_hypothesis_testing_and_modeling.ipynb` selesai dieksekusi 100%.
  - Seluruh target proyek Data Mining tercapai dengan sempurna.

---

## TABEL BENCHMARK HASIL EVALUASI MODEL PREDIKSI PADA HOLDOUT TEST SET

Berikut adalah hasil evaluasi komparatif aktual pada set pengujian (*Holdout Test Set 20% Unseen Data*, 240 observasi):

| No | Nama Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Biaya Default ($\tau=0.50$) | Biaya Optimal ($\tau^*$) | Penghematan Finansial |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Tuned)** | **0.6417** | **0.6000** | **0.6330** | **0.6161** | **0.7050** | **0.6392** | \$22,800 | **\$6,350** ($\tau^*=0.24$) | **\$16,450 (72.1%)** |
| 2 | **Voting Ensemble (RF + XGB)** | 0.6333 | 0.5882 | 0.6422 | 0.6140 | 0.7047 | 0.6327 | \$22,000 | **\$6,200** ($\tau^*=0.18$) | **\$15,800 (71.8%)** |
| 3 | **Logistic Regression (L2)** | 0.6250 | 0.5725 | **0.6881** | **0.6250** | 0.7039 | 0.6230 | \$19,800 | **\$5,400** ($\tau^*=0.20$) | **\$14,400 (72.7%)** |
| 4 | **XGBoost Classifier** | 0.6292 | 0.5794 | 0.6697 | 0.6213 | 0.7016 | 0.6287 | \$21,400 | **\$6,250** ($\tau^*=0.11$) | **\$15,150 (70.8%)** |

> **Analisis Bisnis**:
> Karena biaya *False Negative* (\$500) jauh lebih fatal dibanding *False Positive* (\$50), pergeseran ambang batas klasifikasi ke nilai optimal $\tau^* \approx 0.18 - 0.24$ memangkas kerugian akibat kehilangan pelanggan secara drastis, membuktikan keunggulan *Cost-Sensitive Machine Learning* dalam konteks bisnis industri riil.

---

## LEMBAR PENGESAHAN & REVIEW DOSEN PEMBIMBING

Laporan kemajuan berkala (*Daily Activities*) dan implementasi kode penambangan data end-to-end ini telah disusun secara mandiri dan modular sesuai kurikulum akademik Program Studi Teknik Informatika.

| Komponen Evaluasi | Catatan / Umpan Balik Dosen | Nilai / Status |
| :--- | :--- | :---: |
| **Arsitektur Modular & Storage SQLite** | Kode modular di `src/`, database relasional `telco_churn.db`, logging dan caching terintegrasi. | [ ] Memenuhi Syarat |
| **Peningkatan Algoritma (MICE, Isolation Forest, Yeo-Johnson, VIF)** | Imputasi MICE menjaga kovariansi, anomali multivariat, normalisasi distribusi, audit multikolinearitas. | [ ] Memenuhi Syarat |
| **Pengujian Hipotesis & ML Benchmark** | Uji formal Chi-Square, ANOVA, T-Test, Bonferroni, Stratified 5-Fold CV, ROC-AUC, PR-AUC, dan Cost Matrix. | [ ] Memenuhi Syarat |
| **Konsistensi Log Harian** | Aktivitas harian tercatat runtut dari Hari 1 hingga Hari 20 tanpa inkonsistensi penamaan. | [ ] Memenuhi Syarat |

\
\
Semarang, 21 Oktober 2026

**Mengetahui / Menyetujui:**

Dosen Pengampu Mata Kuliah Data Mining,

\
\
\
__________________________________  
**Abu Salam, M.Kom**  
**NIP. / NIDN. :**  
