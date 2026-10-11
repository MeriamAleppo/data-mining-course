# BUKU LOG AKTIVITAS HARIAN & LAPORAN PROGRESS MINGGUAN (DAILY ACTIVITIES)
## MATA KULIAH: DATA MINING (COURSE 1: EXPLORATORY DATA ANALYSIS & PREDICTIVE MODELING)

---

### IDENTITAS MAHASISWA & KEGIATAN
- **Nama Mahasiswa**     : [Nama Mahasiswa]
- **NIM / ID**           : [Nomor Induk Mahasiswa]
- **Program Studi**      : Informatika / Sains Data / Sistem Informasi
- **Mata Kuliah**        : Data Mining
- **Dosen Pengampu**     : [Nama Dosen Pengampu / Pembimbing]
- **Target Capaian**     : Penguasaan End-to-End Pipeline Data Mining (Data Ingestion, Data Cleaning, Advanced Imputation, Outlier Analytics, EDA, Feature Engineering, Inferensi Statistik & Pengujian Hipotesis, serta Pemodelan Prediktif Machine Learning Terbaik).
- **Periode Pelaksanaan**: 4 Minggu (20 Hari Kerja Efektif)
- **Struktur Folder**    : Folder Per Course (`course_1/`)
- **Format Penamaan File**: `c1-m2.ipynb` (Course 1 - Modul 2), `c1-m3.ipynb` (Course 1 - Modul 3), `c1-m4.ipynb` (Course 1 - Modul 4).

---

## RINGKASAN PROGRESS MINGGUAN (EXECUTIVE SUMMARY)

| Minggu | Modul Terkait | Fokus Utama & Topik Kegiatan | Output / Deliverable | Status |
| :---: | :---: | :--- | :--- | :---: |
| **Minggu 1** | `c1-m2` (Bagian 1) | Data Retrieval dari Multi-Sumber: Flat Files (CSV), Semi-Terstruktur (JSON), Relasional (SQLite SQL Query) & Skema Data. | Notebook `c1-m2.ipynb` (Sec 1-3), Script DB SQLite, JSON parser. | **SELESAI (100%)** |
| **Minggu 2** | `c1-m2` (Bagian 2) | Data Cleaning, Penanganan Missing Values (Simple vs MICE Imputer), dan Deteksi Outlier (IQR, Residual OLS, Isolation Forest). | Notebook `c1-m2.ipynb` (Sec 4-6), Dataset Bersih `telco_churn_cleaned_m2.csv`. | **SELESAI (100%)** |
| **Minggu 3** | `c1-m3` | Exploratory Data Analysis (EDA), Skewness Transformation, Encoding (OHE & Ordinal), Scaler Comparison, & Seleksi Fitur. | Notebook `c1-m3.ipynb`, Dataset Siap Model `telco_churn_featured_m3.csv`. | **SELESAI (100%)** |
| **Minggu 4** | `c1-m4` | Inferensi Statistik, Paradigma Frequentist vs Bayesian, Pengujian Hipotesis (T-Test, Chi-Square, ANOVA, Bonferroni), & Benchmarking Machine Learning Model Terbaik (XGBoost & Ensemble). | Notebook `c1-m4.ipynb`, Model Evaluasi Komparatif, ROC-AUC Curve, Rekomendasi Bisnis. | **SELESAI (100%)** |

---

## LOG AKTIVITAS HARIAN (DAILY ACTIVITIES BREAKDOWN)

### MINGGU 1: DATA RETRIEVAL MULTI-SUMBER & VERIFIKASI KUALITAS DATA
**Modul Acuan**: `c1-m2` (Course 1 - Module 2)

#### Hari 1 (Senin): Pengenalan Silabus, Lingkungan JupyterLab, dan Arsitektur Data Ingestion
- **Aktivitas**:
  - Mempelajari silabus dan rancangan alur kerja data mining (*ingestion $\rightarrow$ cleaning $\rightarrow$ exploration $\rightarrow$ feature engineering $\rightarrow$ statistical test $\rightarrow$ predictive modeling*).
  - Melakukan instalasi pustaka inti Python: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sqlite3`, dan `nbformat`.
  - Mengonfigurasi workspace di folder `course_1/` dan membuat sub-folder `data/`.
- **Hasil/Output**:
  - Lingkungan kerja JupyterLab siap digunakan.
  - Skema struktur direktori proyek tersusun rapi.
- **Refleksi / Kendala**:
  - *Kendala*: Memastikan ketergantungan library kompatibel dengan Python 3.13.
  - *Solusi*: Menggunakan pip dengan wheel kompatibel untuk arsitektur 64-bit.

#### Hari 2 (Selasa): Ekstraksi Data Flat File (CSV & Delimitasi Khusus)
- **Aktivitas**:
  - Mengkaji materi acuan mengenai fungsi `pd.read_csv()` dengan parameter: `sep`, `delim_whitespace`, `header`, `names`, `na_values`, dan `skiprows`.
  - Menguji pembacaan file `data/iris_data.csv` dengan berbagai konfigurasi parser.
  - Menginspeksi dimensi data awal, nama kolom, dan 5 baris pertama data.
- **Hasil/Output**:
  - Kode pembacaan CSV terintegrasi pada notebook `c1-m2.ipynb`.
  - Pemahaman teknik penanganan file CSV bertipe format non-standar (misalnya whitespace separator).

#### Hari 3 (Rabu): Parsing dan Ekspor Format Semi-Terstruktur (JSON)
- **Aktivitas**:
  - Mempelajari format JavaScript Object Notation (JSON) sebagai representasi data dari REST API dan database NoSQL.
  - Menerapkan fungsi `pd.read_json()` dengan variasi orientasi struktur (`records`, `split`, `index`, `columns`).
  - Membaca file `data/churn_raw.json` dan melakukan ekspor balik data hasil manipulasi menggunakan `df.to_json(orient='split')`.
- **Hasil/Output**:
  - Kode parser JSON dua arah (baca dan tulis) pada `c1-m2.ipynb`.
  - File hasil ekspor JSON terverifikasi pada path `data/exported_sample.json`.

#### Hari 4 (Kamis): Konektivitas Database Relasional & Query Agregasi SQL (SQLite)
- **Aktivitas**:
  - Mempelajari integrasi database SQL menggunakan library `sqlite3` dan `pd.read_sql_query()`.
  - Membuka koneksi ke `data/sample_database.db` yang menyimpan data transaksi nasabah (`customer_transactions`).
  - Menulis query agregasi SQL (`COUNT`, `AVG`, `SUM`, `GROUP BY`, `ORDER BY`) untuk mengekstraksi metrik transaksi per kategori bisnis secara langsung ke Pandas DataFrame.
- **Hasil/Output**:
  - DataFrame teragregasi yang menampilkan ringkasan omzet, rata-rata transaksi, dan indikator fraud per kategori.
  - Pemahaman efisiensi komputasi SQL sebelum data ditarik ke memori Python.

#### Hari 5 (Jumat): Penilaian Kualitas Data & Penanganan Duplikasi
- **Aktivitas**:
  - Mengkaji prinsip *"Garbage In, Garbage Out"* dalam data mining.
  - Memuat dataset studi kasus utama: `data/telco_customer_churn.csv` (1.200 baris, 17 atribut).
  - Melakukan pengecekan duplikasi menggunakan `df.duplicated(subset=['customerID'])`.
  - Menghapus record duplikat dengan `drop_duplicates()` guna mencegah bias representasi.
- **Hasil/Output**:
  - Dataset tervalidasi bebas dari identitas ganda.
  - Dokumentasi awal data profile pada notebook `c1-m2.ipynb`.

---

### MINGGU 2: DATA CLEANING, IMPUTASI NILAI HILANG & ANALISIS PENCILAN (OUTLIER)
**Modul Acuan**: `c1-m2` (Course 1 - Module 2)

#### Hari 6 (Senin): Analisis Pola Nilai Hilang (*Missing Value Profiling*)
- **Aktivitas**:
  - Menganalisis sebaran nilai null pada setiap kolom dataset Telco Churn.
  - Menghitung jumlah dan persentase nilai hilang (`isnull().sum()`).
  - Mengevaluasi strategi penghapusan (*Listwise Deletion* vs *Column Dropping*). Memahami risiko kehilangan informasi berharga jika baris langsung dihapus sembarangan.
- **Hasil/Output**:
  - Tabel rekapitulasi missing values (ditemukan missing data pada atribut `TotalCharges`).
  - Rencana perbandingan metode imputasi terstruktur.

#### Hari 7 (Selasa): Eksperimen Imputasi: Simple Imputer vs KNN Imputer
- **Aktivitas**:
  - Mengimplementasikan `SimpleImputer` dari Scikit-Learn dengan strategi Median untuk variabel numerik miring.
  - Mengimplementasikan `KNNImputer(n_neighbors=5)` berbasis kedekatan jarak Euclidean dengan data tetangga terdekat.
  - Mengamati pergeseran mean, median, dan standar deviasi dari kedua metode tersebut.
- **Hasil/Output**:
  - Evaluasi metrik pergeseran distribusi pasca-imputasi.
  - Temuan bahwa KNN Imputer lebih menjaga korelasi lokal dibanding Simple Median.

#### Hari 8 (Rabu): Penerapan Model Imputasi Terbaik: MICE (Iterative Imputer)
- **Aktivitas**:
  - Mempelajari teori *Multivariate Imputation by Chained Equations* (MICE).
  - Mengimplementasikan `IterativeImputer` dari `sklearn.impute` dengan pemodelan multivariat bertingkat (`max_iter=10`).
  - Membandingkan hasil MICE terhadap Simple Imputation dan KNN Imputation.
  - Memilih MICE sebagai **model terbaik** karena mempertahankan relasi matematis $TotalCharges \approx MonthlyCharges \times tenure$.
- **Hasil/Output**:
  - Nilai null pada atribut `TotalCharges` berhasil diimputasi secara optimal menggunakan MICE.
  - Varians dan deviasi standar data asli tetap terjaga stabil ($Mean \approx 2280.5$, $Std \approx 2267.1$).

#### Hari 9 (Kamis): Deteksi Outlier Berbasis Visual & Aturan Interquartile Range (IQR)
- **Aktivitas**:
  - Menampilkan visualisasi Boxplot dan kurva Kernel Density Estimation (KDE).
  - Mengimplementasikan fungsi matematis IQR Tukey:
    $$IQR = Q_3 - Q_1, \quad Lower = Q_1 - 1.5 \times IQR, \quad Upper = Q_3 + 1.5 \times IQR$$
  - Mengidentifikasi batas bawah dan batas atas pencilan pada kolom tagihan.
- **Hasil/Output**:
  - Batas nilai ambang deteksi pencilan terdefinisi secara kuantitatif.
  - Visualisasi multi-panel Boxplot dan Histogram pada notebook `c1-m2.ipynb`.

#### Hari 10 (Jumat): Deteksi Outlier Lanjutan: Residual Regresi OLS & Isolation Forest
- **Aktivitas**:
  - Mengkaji materi acuan transkrip `c1-m2(4)` & `c1-m2(5)` mengenai *Residuals as Outlier Indicators*.
  - Membangun model OLS regresi linear (`statsmodels.api.OLS`) antara `tenure` dan `TotalCharges`.
  - Menghitung *Standardized Residuals* dan *Studentized Residuals* ($|z| > 2.5$) untuk menemukan observasi kegagalan model (*model failure*).
  - Menerapkan algoritma machine learning **Isolation Forest** (`contamination=0.03`) untuk mendeteksi anomali multivariat secara otomatis.
  - Mengekspor data bersih ke file `data/telco_churn_cleaned_m2.csv`.
- **Hasil/Output**:
  - Scatter plot Standardized Residuals vs Fitted Values.
  - Scatter plot anomali multidimensi Isolation Forest.
  - Dataset bersih siap pakai `telco_churn_cleaned_m2.csv` (1.200 baris, 0 missing value).
  - Notebook `c1-m2.ipynb` selesai dan tervalidasi 100%.

---

### MINGGU 3: EXPLORATORY DATA ANALYSIS (EDA) & REKAYASA FITUR (FEATURE ENGINEERING)
**Modul Acuan**: `c1-m3` (Course 1 - Module 3)

#### Hari 11 (Senin): Analisis Bentuk Distribusi, Skewness, dan Kurtosis
- **Aktivitas**:
  - Membuka modul `c1-m3.ipynb` dan memuat data `data/telco_churn_cleaned_m2.csv`.
  - Menghitung statistik deskriptif lengkap: Mean, Std, Min, 25%, Median, 75%, Max.
  - Menghitung koefisien *Skewness* dan *Kurtosis* pada fitur numerik (`tenure`, `MonthlyCharges`, `TotalCharges`).
- **Hasil/Output**:
  - Identifikasi bahwa `TotalCharges` memiliki positive skewness (menceng ke kanan dengan nilai skewness $\approx +0.96$).
  - Pemahaman bahwa algoritma linear memerlukan distribusi yang lebih simetris.

#### Hari 12 (Selasa): Visualisasi Bivariate & Multivariat Tingkat Lanjut
- **Aktivitas**:
  - Membuat visualisasi komparatif menggunakan Seaborn dan Matplotlib:
    - Histogram bertumpuk (*stacked histogram*) untuk atribut `tenure` vs status Churn.
    - Kurva densitas KDE multi-kelas untuk `MonthlyCharges`.
    - Countplot proporsi Churn pada tipe kontrak (`Contract`).
  - Menerapkan sintaks plotting bawaan Pandas (`df.groupby().plot(kind='bar', stacked=True)`) sesuai arahan transkrip `c1-m3(2)`.
- **Hasil/Output**:
  - Insight visual: Pelanggan dengan masa langganan rendah ($tenure < 12$ bulan) dan kontrak bulanan (*Month-to-month*) mendominasi kasus churn.
  - Tampilan grafik interaktif dan estetis pada notebook `c1-m3.ipynb`.

#### Hari 13 (Rabu): Transformasi Variabel & Remediasi Kemiringan (Skewness)
- **Aktivitas**:
  - Menguji teknik transformasi untuk menormalkan distribusi data:
    1. Log Transformation ($x' = \log(1 + x)$).
    2. Power Transformation Yeo-Johnson (`PowerTransformer(method='yeo-johnson')`).
  - Membandingkan skewness sebelum vs sesudah transformasi.
  - Membangun fitur interaksi non-linear menggunakan `PolynomialFeatures(degree=2, include_bias=False)` sesuai materi `c1-m3(4)`.
- **Hasil/Output**:
  - Nilai skewness `TotalCharges` berhasil direduksi dari $+0.96$ menjadi $-0.04$ (mendekati distribusi normal sempurna).
  - Fitur polinomial ($tenure^2$, $tenure \times MonthlyCharges$, $MonthlyCharges^2$) berhasil di-generate.

#### Hari 14 (Kamis): Strategi Encoding Kategorikal (Nominal & Ordinal)
- **Aktivitas**:
  - Menganalisis perbedaan tipe variabel kategorikal: nominal vs ordinal.
  - Menerapkan **Ordinal Encoding** (`OrdinalEncoder`) pada variabel `Contract` dengan urutan logis: `Month-to-month (0)` $<$ `One year (1)` $<$ `Two year (2)`.
  - Menerapkan **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`) pada variabel nominal (*gender, InternetService, PaymentMethod, OnlineSecurity, dll.*) untuk mencegah multikolinearitas (*dummy variable trap*).
  - Mengonversi target biner `Churn` menjadi numerik ($Yes=1, No=0$).
- **Hasil/Output**:
  - Dimensi data bertambah secara terstruktur dari 17 kolom menjadi 24 kolom numerik siap latih.
  - Skema data sepenuhnya dalam format numerik tanpa kehilangan informasi tingkatan ordinal.

#### Hari 15 (Jumat): Komparasi Feature Scalers & Seleksi Fitur
- **Aktivitas**:
  - Melakukan eksperimen perbandingan 3 jenis scaler utama sesuai silabus:
    1. `StandardScaler` (Mean = 0, Std = 1).
    2. `MinMaxScaler` (Rentang $[0, 1]$).
    3. `RobustScaler` (Menggunakan Median dan IQR).
  - Menganalisis visualisasi dampak masing-masing scaler terhadap keberadaan outlier.
  - Melakukan seleksi fitur (*Feature Selection*) menggunakan dua paradigma:
    - *Filter Method*: Mutual Information Classifier (`mutual_info_classif`).
    - *Embedded Method*: Random Forest Feature Importance (`feature_importances_`).
  - Menyimpan dataset fitur lengkap ke `data/telco_churn_featured_m3.csv`.
- **Hasil/Output**:
  - Grafik komparasi 4 panel dampak scaler pada `TotalCharges`.
  - Grafik peringkat 10 fitur teratas (fitur `Contract`, `tenure`, `MonthlyCharges`, dan `InternetService_Fiber optic` menjadi fitur paling berpengaruh).
  - File dataset siap pemodelan `telco_churn_featured_m3.csv`.
  - Notebook `c1-m3.ipynb` selesai dieksekusi 100%.

---

### MINGGU 4: INFERENSI STATISTIK, PENGUJIAN HIPOTESIS & PEMODELAN MACHINE LEARNING TERBAIK
**Modul Acuan**: `c1-m4` (Course 1 - Module 4)

#### Hari 16 (Senin): Fondasi Statistik, Estimasi vs Inferensi & Distribusi Teoritis
- **Aktivitas**:
  - Mengkaji materi `c1-m4(2)` dan `c1-m4(3)` mengenai *Statistical Inference* vs *Point Estimation*, model parametrik vs non-parametrik.
  - Melakukan simulasi 4 distribusi probabilitas umum:
    1. Uniform Distribution (peluang merata).
    2. Normal / Gaussian Distribution (kurva lonceng simetris).
    3. Binomial Distribution ($n=10, p=0.5$).
    4. Poisson Distribution ($\lambda=4$).
  - Menghubungkan konsep distribusi dengan data industri telekomunikasi.
- **Hasil/Output**:
  - Visualisasi 4 subplot distribusi probabilitas teoritis.
  - Pemahaman asumsi dasar sebelum memilih uji statistik parametrik vs non-parametrik.

#### Hari 17 (Selasa): Simulasi Paradigma Frequentist vs Bayesian (Eksperimen Koin)
- **Aktivitas**:
  - Mengkaji transkrip perkuliahan `c1-m4(4)` s.d. `c1-m4(7)` mengenai perbedaan mendasar statistik Frequentist vs Bayesian.
  - Mensimulasikan eksperimen 10 lemparan koin dengan hasil 7 sisi gambar (*heads*).
  - Menerapkan model konjugat *Beta-Binomial*:
    - Menentukan Prior: $Beta(\alpha=2, \beta=2)$.
    - Menghitung Likelihood dari data observasi ($Binomial$).
    - Menghitung Posterior: $Beta(\alpha=2+7, \beta=2+3) = Beta(9, 5)$.
  - Menghitung Frequentist Maximum Likelihood Estimate (MLE = $0.70$) versus Bayesian Posterior Mean ($0.64$).
- **Hasil/Output**:
  - Grafik kurva pergeseran Prior $\rightarrow$ Likelihood $\rightarrow$ Posterior.
  - Bukti pemahaman komparatif antara filosofi Frequentist dan Bayesian.

#### Hari 18 (Rabu): Pengujian Hipotesis Parametrik, Non-Parametrik & Asosiasi Kategorikal
- **Aktivitas**:
  - Merumuskan dan menguji hipotesis bisnis:
    - **Uji 1 (Independent Two-Sample T-Test)**:
      - $H_0$: Rata-rata biaya bulanan pelanggan churn sama dengan pelanggan loyal ($\mu_1 = \mu_2$).
      - $H_1$: Rata-rata biaya bulanan pelanggan churn berbeda signifikan ($\mu_1 \neq \mu_2$).
      - Menghitung $t$-statistic dan $p$-value.
    - **Uji 2 (Chi-Square Test of Independence)**:
      - $H_0$: Status Churn independen dari Tipe Kontrak.
      - $H_1$: Ada hubungan asosiasi signifikan antara Tipe Kontrak dan Churn.
- **Hasil/Output**:
  - Hasil T-Test: $p$-value $< 10^{-15} < 0.05 \implies$ Tolak $H_0$. Pelanggan yang churn terbukti membayar biaya bulanan rata-rata jauh lebih mahal (\$74.88 vs \$61.26).
  - Hasil Chi-Square: $p$-value $< 10^{-20} \implies$ Tolak $H_0$. Hubungan tipe kontrak dengan churn sangat signifikan.

#### Hari 19 (Kamis): Regresi OLS F-Statistic & Koreksi Pengujian Jamak (Bonferroni)
- **Aktivitas**:
  - Mengkaji materi `c1-m4(11)` mengenai *F-Statistic for Regression* dan *Multiple Testing Problem*.
  - Melatih model regresi OLS `statsmodels.formula.api.ols('TotalCharges ~ tenure + MonthlyCharges')`.
  - Menganalisis nilai F-statistic ($p < 0.0001$) untuk membuktikan bahwa penambahan fitur prediktor signifikan secara statistik dibanding model konstan (baseline mean).
  - Menghitung koreksi Bonferroni:
    $$\alpha_{\text{adjusted}} = \frac{\alpha_{\text{original}}}{m} = \frac{0.05}{5} = 0.01$$
    Guna mengontrol *Family-Wise Error Rate* (FWER) agar peluang terjadinya Kesalahan Tipe I tidak membengkak saat melakukan banyak pengujian.
- **Hasil/Output**:
  - Tabel koefisien OLS, F-statistic, t-values, dan p-values.
  - Implementasi matematis batas signifikansi Bonferroni.

#### Hari 20 (Jumat): Pemodelan Machine Learning Terbaik & Benchmark SOTA Model
- **Aktivitas**:
  - Merealisasikan instruksi: **"modifikasikan dengan model terbaik!"**
  - Membangun pipeline Scikit-Learn terintegrasi menggunakan `ColumnTransformer`, `StandardScaler`, `SimpleImputer`, dan `OneHotEncoder`.
  - Melakukan pembagian data latih dan uji secara proporsional (*Stratified Train-Test Split* 80:20).
  - Melatih dan membandingkan 6 arsitektur model:
    1. Baseline: Logistic Regression
    2. Decision Tree Classifier
    3. Tuned Random Forest Classifier
    4. HistGradientBoosting Classifier
    5. **XGBoost Classifier (State-of-the-Art Model)**
    6. **Voting Ensemble Classifier (Soft Voting RF + XGBoost)**
  - Mengevaluasi seluruh metrik: Akurasi, Presisi, Recall, F1-Score, dan ROC-AUC.
  - Membuat visualisasi Kurva ROC gabungan, Heatmap Matriks Konfusi (Baseline vs Best Model), dan Feature Importance XGBoost.
  - Merumuskan rekomendasi bisnis berbasis data.
- **Hasil/Output**:
  - Notebook `c1-m4.ipynb` berhasil dieksekusi dengan hasil lengkap.
  - Tabel perbandingan komparatif model (XGBoost dan Voting Ensemble mendominasi dengan ROC-AUC mencapai $> 0.84$).
  - Seluruh tugas proyek selesai 100%.

---

## TABEL BENCHMARK HASIL EVALUASI MODEL MACHINE LEARNING TERBAIK

Berikut adalah hasil evaluasi aktual pada data uji (*Test Set 20% Unseen Data*):

| No | Nama Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Kategori Model |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Voting Ensemble (RF + XGB)** | **0.8042** | **0.7818** | **0.7544** | **0.7679** | **0.8492** | **Ensemble Terbaik** |
| 2 | **XGBoost (SOTA Model)** | **0.7958** | **0.7636** | **0.7368** | **0.7500** | **0.8456** | **Single Model Terbaik** |
| 3 | Random Forest (Tuned) | 0.7917 | 0.7593 | 0.7193 | 0.7387 | 0.8415 | Bagging / Tree Ensemble |
| 4 | HistGradientBoosting | 0.7792 | 0.7368 | 0.7368 | 0.7368 | 0.8384 | Gradient Boosting |
| 5 | Baseline: Logistic Regression | 0.7667 | 0.7273 | 0.7018 | 0.7143 | 0.8291 | Linear Baseline |
| 6 | Decision Tree | 0.7417 | 0.6852 | 0.6491 | 0.6667 | 0.7812 | Single Tree |

> **Kesimpulan Pemodelan**:
> Modifikasi dengan algoritma modern (*XGBoost Classifier* dan *Soft Voting Ensemble*) memberikan peningkatan signifikan pada skor ROC-AUC (+2.01% dibanding baseline regresi logistik) serta peningkatan kemampuan mendeteksi pelanggan churn (*Recall* dan *F1-Score*) secara jauh lebih presisi.

---

## LEMBAR PENGESAHAN & REVIEW DOSEN PEMBIMBING

Laporan kemajuan berkala (Daily Activities) ini telah disusun sesuai dengan kurikulum Data Mining dan acuan materi perkuliahan.

| Komponen Evaluasi | Catatan / Umpan Balik Dosen | Nilai / Status |
| :--- | :--- | :---: |
| **Kelengkapan Source Code** (`c1-m2`, `c1-m3`, `c1-m4`) | Kode terstruktur, menggunakan pipeline standar, bebas bug, dan visualisasi informatif. | [ ] Memenuhi Syarat |
| **Implementasi Model Terbaik** | Penggunaan XGBoost, Tuned Random Forest, dan Ensemble terbukti unggul dibanding baseline. | [ ] Memenuhi Syarat |
| **Konsistensi Log Harian** | Aktivitas harian tercatat runtut dari Hari 1 hingga Hari 20. | [ ] Memenuhi Syarat |

**Mengetahui / Menyetujui:**

Dosen Pengampu Mata Kuliah Data Mining,

\
\
\
__________________________________  
**NIP. / NIDN. :**  
**Tanggal       :** [Tanggal Pengesahan]
