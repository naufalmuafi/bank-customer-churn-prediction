# **Project Documentation Report**

Di era yang didominasi oleh transformasi digital dan harapan pelanggan yang berkembang pesat, industri perbankan dihadapkan pada tantangan kritis untuk mempertahankan basis pelanggan mereka. Menyadari pentingnya strategi proaktif, proyek ini berusaha mengatasi kekhawatiran mendesak tentang churn pelanggan melalui lensa analitika prediktif.


# Table of Contents

1. [**Project Overview**](#project-overview)<br><t>
    1.1. [Project Title](#project-title)<br><t>
    1.2. [Project Executive Summary](#project-executive-summary)<br><t>
    1.3. [Project Author](#project-author)<br><t>

2. [**Project Domain or Background**](#project-background-or-domain)<br><t>

3. [**Business Understanding**](#business-understanding)<br><t>
    3.1. [Problem Statements](#problem-statements)<br><t>
    3.2. [Goals](#goals)<br><t>
    3.3. [Solution Statements](#solution-statements)<br><t>

4. [**Data Understanding**](#data-understanding)<br><t>
    4.1. [Dataset Information](#dataset-information)<br><t>
        4.1.1. [Variables in the Dataset](#variables-in-the-dataset)<br><t>

5. [**Data Preparation**](#data-preparation)<br><t>
    5.1. [Handle Imbalanced Data with Resample](#handle-imbalanced-data-with-resample)<br><t>
    5.2. [Category Feature Encoding](#category-feature-encoding)<br><t>
    5.3. [Correlation Analysis](#correlation-analysis)<br><t>
    5.4. [Train Test Split](#train-test-split)<br><t>
    5.5. [Feature Scaling](#feature-scaling)<br><t>

6. [**Model Development**](#model-development)<br><t>
    6.1. [K-Nearest Neighbourhood Algorithm](#k-nearest-neighbourhood-algorithm)<br><t>
    6.2. [Logistic Regression Algorithm](#logistic-regression-algorithm)<br><t>
    6.3. [Support Vector Classifier Algorithm](#support-vector-classifier-algorithm)<br><t>
    6.4. [Random Forest Algorithm](#random-forest-algorithm)<br><t>

7. [**Evaluation**](#model-evaluation)<br><t>
    7.1. [Confusion Matrix](#confusion-matrix)<br><t>
    7.2. [Model Comparison](#model-comparison)<br><t>

8. [**Model Prediction**](#model-prediction)<br><t>
    8.1. [Sample Predictions](#sample-predictions)<br><t>

9. [**Conclusion**](#conclusion)<br><t>
    9.1. [Recommendations](#recommendations)<br><t>
    9.2. [Future Work](#future-work)<br><t>

10. [**References**](#references)<br><t>


## Project Overview

### Project Title
Bank Customer Churn Prediction

### Project Executive Summary

Dalam proyek ini, kami bertujuan untuk memprediksi churn pelanggan bank menggunakan analitika prediktif. Tujuannya adalah untuk menentukan apakah seorang pelanggan lebih cenderung tinggal atau keluar dari bank. Kami akan membandingkan berbagai algoritma machine learning untuk mengidentifikasi model terbaik untuk prediksi ini.

### Project Author
Naufal Mu'afi
[naufalmuafi@mail.ugm.ac.id](mailto:naufalmuafi@mail.ugm.ac.id)

---

## Project Background or Domain


![Customer-Churn-Illustration-960x343[^pict1]](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/a2216848-5581-4617-a506-f4c8d44e17f6)

[^pict1]: [Illustration by CLEARTOUCH](https://www.cleartouch.in/what-is-customer-churn-and-how-do-you-prevent-it/)


Di industri perbankan, churn pelanggan adalah kekhawatiran yang kritis karena berdampak langsung pada pendapatan dan profitabilitas bank. Memahami dan memprediksi churn pelanggan dapat membantu bank mengambil langkah-langkah proaktif untuk mempertahankan pelanggan. Churn pelanggan, yang didefinisikan sebagai kemungkinan pelanggan menghentikan asosiasinya dengan sebuah perusahaan dalam jangka waktu tertentu, merupakan tantangan signifikan yang dihadapi banyak perusahaan global [^1]. Dikenal sebagai agitasi pelanggan dalam bisnis, ini terjadi ketika pelanggan menyatakan ketidakpuasan terhadap layanan atau produk yang diberikan oleh perusahaan dan memilih untuk beralih ke pesaing atau berhenti menggunakan layanan tersebut sama sekali.

Penelitian telah menunjukkan bahwa biaya mendapatkan pelanggan baru lebih tinggi daripada mempertahankan pelanggan yang sudah ada. Oleh karena itu, bank memiliki insentif finansial untuk mengidentifikasi faktor-faktor yang menyebabkan churn dan mengambil langkah-langkah untuk menguranginya. Salah satu pendekatan yang dapat digunakan adalah menggunakan machine learning untuk membangun model prediktif yang dapat mengidentifikasi pelanggan yang cenderung churn.

Penelitian pionir oleh F. F. Reichheld dan W. E. Sasser Jr. menunjukkan korelasi kuat antara retensi pelanggan dan keuntungan perusahaan, dengan menunjukkan bahwa peningkatan retensi pelanggan sebesar 5 persen saja menghasilkan peningkatan profitabilitas, mulai dari 20 hingga 85 persen di berbagai bisnis. Selain itu, penelitian secara konsisten menunjukkan bahwa mempertahankan klien yang sudah ada biayanya sekitar lima kali lebih rendah daripada mengakuisisi yang baru [^2].


## Business Understanding

### Problem Statements

Berdasarkan latar belakang, kita dapat mengidentifikasi masalah yang dapat dipecahkan dalam proyek ini.

1. Bank ingin secara proaktif mengidentifikasi pelanggan yang kemungkinan akan meninggalkan, memungkinkan mereka mengimplementasikan strategi retensi. Jadi, bagaimana data dapat didefinisikan untuk digunakan dalam membuat model yang baik?
2. Prediksi yang akurat terkait dengan churn pelanggan dapat berdampak signifikan pada kepuasan pelanggan dan kinerja bisnis secara keseluruhan. Lalu, bagaimana kita dapat membuat model machine learning untuk memprediksi churn pelanggan bank?


### Goals

Selanjutnya, kita dapat merinci tujuan yang diinginkan dari proyek ini.

1. Mengembangkan model machine learning untuk memprediksi churn pelanggan bank menggunakan data yang telah didefinisikan dan dianalisis dengan baik.
2. Bertujuan mencapai akurasi tinggi, melampaui 85%, dalam memprediksi apakah seorang pelanggan akan memilih untuk tinggal atau keluar dari bank.

### Solution Statements

Untuk mencapai tujuan kami, kami akan mengeksplorasi data dengan analisis yang kuat dan beberapa algoritma machine learning, termasuk K-Nearest Neighbors (KNN), Regresi Logistik, Support Vector Classifier (SVC), dan Random Forest. Kami akan menyempurnakan hiperparameter dan memilih model terbaik berdasarkan akurasi.

- Pertama, kami memuat data dari dataset open source, seperti Kaggle. Kemudian, penting untuk memahami definisi setiap fitur dalam dataset.
  
- Selama fase Analisis Data, kami menganalisis data dengan:
   - **Menilai dan Membersihkan Data:**
     Identifikasi dan tangani nilai-nilai yang hilang, outlier, dan kesalahan untuk memastikan kualitas data.
   - **Analisis Univariat:**
     Analisis fitur individu untuk mendapatkan wawasan tentang distribusi dan karakteristiknya.
   - **Analisis Multivariat:**
     Menjelajahi hubungan dan interaksi antara beberapa fitur untuk mengungkap pola.

- Kemudian, dalam fase Persiapan Data, kami menggunakan berbagai metode, termasuk:
   - **Menangani Data yang Tidak Seimbang dengan Resample:**
     Menangani dataset yang tidak seimbang dengan teknik resampling seperti oversampling atau undersampling.
   - **Encoding Fitur Kategorikal:**
     Mengubah fitur kategorikal ke dalam format yang sesuai untuk algoritma machine learning.
   - **Analisis Korelasi:**
     Meneliti korelasi antara fitur untuk mengidentifikasi dan menangani multicollinearity.
   - **Pembagian Data Latih dan Uji:**
     Membagi dataset menjadi set latih dan uji untuk evaluasi model.
   - **Scaling Fitur:**
     Menormalkan atau menyesuaikan skala fitur numerik untuk memastikan mereka berada pada skala yang serupa, bermanfaat bagi beberapa algoritma machine learning.
     
- Dalam pengembangan model machine learning, kami akan menerapkan beberapa algoritma. Oleh karena itu, proyek ini akan mencakup implementasi empat algoritma, yaitu:
   - **Algoritma Tetangga Terdekat (KNN):**
     K-Nearest Neighbors (KNN) adalah algoritma sederhana dan efektif yang digunakan untuk tugas klasifikasi dan regresi. Ini mengklasifikasikan titik data berdasarkan kelas mayoritas dari tetangga terdekatnya di ruang fitur.
   - **Algoritma Regresi Logistik:**
     Regresi Logistik adalah model statistik yang digunakan untuk klasifikasi biner. Ini memperkirakan probabilitas bahwa suatu instansi tertentu termasuk dalam kategori tertentu dan membuat prediksi berdasarkan fungsi logistik.
   - **Algoritma Support Vector Classifier (SVC):**
     Support Vector Classifier, atau Support Vector Machine (SVM), adalah algoritma kuat untuk tugas klasifikasi dan regresi. Ini bekerja dengan menemukan hyperplane yang terbaik memisahkan data ke dalam kelas yang berbeda sambil memaksimalkan margin.
   - **Algoritma Random Forest:**
     Random Forest adalah metode ensemble learning yang membangun banyak pohon keputusan selama pelatihan dan mengeluarkan modus dari kelas (klasifikasi) atau prediksi rata-rata (regresi) dari pohon-pohon individu. Ini dikenal karena kekokohannya dan akurasi tinggi.

- Dalam upaya mencapai versi optimal model, proyek ini akan menggunakan metode **Grid Search Cross Validation** untuk menentukan parameter terbaik untuk model.

  Grid Search Cross Validation adalah teknik penyetelan hiperparameter yang secara sistematis mengevaluasi serangkaian kombinasi hiperparameter yang telah ditentukan sebelumnya untuk model machine learning. Ini bekerja dengan membuat grid dari semua nilai hiperparameter yang mungkin dan melakukan validasi silang untuk setiap kombinasi untuk mengidentifikasi set yang menghasilkan kinerja terbaik.

  **Algoritma:**
  1. Tentukan grid nilai hiperparameter untuk model.
  2. Untuk setiap kombinasi hiperparameter di grid:
     - a. Bagi dataset menjadi set latih dan validasi.
     - b. Latih model pada set latih.
     - c. Evaluasi model pada set validasi menggunakan metrik kinerja yang dipilih.
     - d. Ulangi proses untuk lipatan yang berbeda dalam validasi silang jika berlaku.
  3. Pilih kombinasi hiperparameter yang menghasilkan kinerja terbaik.

  **Ekspresi Matematika:**
  Biarkan $H$ menjadi himpunan kombinasi hiperparameter, $S$ menjadi metrik kinerja, dan $K$ menjadi jumlah lipatan dalam validasi silang. Kombinasi hiperparameter optimal $\theta^*$ diperoleh dengan:

  $\theta^* = \arg\min_{\theta \in H} \left( \frac{1}{K} \sum_{k=1}^K S(\text{Model}_{\theta}, \text{Validasi}_k) \right)$

  di mana $\text{Model}_{\theta}$ mewakili model yang dilatih dengan kombinasi hiperparameter $\theta$ dan $\text{Validasi}_k$ menunjukkan set validasi untuk lipatan ke-$k$.

  Grid Search Cross Validation membantu menemukan nilai hiperparameter yang mengoptimalkan kinerja dan generalisasi model ke data yang tidak terlihat.


## Data Understanding

"Dataset yang digunakan untuk proyek ini adalah [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data) yang diperoleh dari Kaggle. Ini berisi informasi tentang nasabah bank dan kemungkinan mereka untuk beralih.

![dataset_info](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/e706020e-5584-4cc0-bfd4-092d2f17b9f1)

**Informasi Dataset:**

Tipe | Informasi
--- | ---
Sumber | [Dataset Kaggle: Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data)
Lisensi | Lainnya
Kategori | `Ekonomi`, `Bisnis`, `Keuangan`, `Perbankan`
Rating Kegunaan | 9,71
Tipe dan Ukuran Berkas | CSV (268 kb)

### Variabel dalam Dataset:

File dataset `Churn_Modelling.csv` memiliki 10.000 baris dan 14 kolom. Ini berarti dataset berisi informasi untuk 10.000 nasabah bank, termasuk 14 detail seperti Jenis Kelamin, Usia, Geografi, dan lain-lain. Dataset terdiri dari 9 kolom tipe data `int64`, 2 kolom tipe data `float64`, dan 3 kolom tipe data `object`. Tidak mungkin menentukan apakah data mewakili fitur numerik atau kategorikal hanya berdasarkan tipe data. Pada awalnya, kami mengidentifikasi 3 fitur palsu dalam dataset: `RowNumber`, `CustomerId`, dan `Surname`. Setelah pemeriksaan lebih lanjut, kami mengklasifikasikan dataset ini memiliki 4 fitur numerik dan 7 fitur kategorikal.

Penjelasan 14 kolom atau fitur dari dataset `Churn_Modelling.csv`:

1. `RowNumber` - Menunjukkan nomor baris.

2. `CustomerId` - Setiap nasabah memiliki ID unik tersimpan dalam fitur ini.

3. `Surname` - Nama belakang nasabah bank.

4. `CreditScore` - Menjelaskan bagaimana Skor Kredit Nasabah Bank dinilai. Dalam dataset ini, nilai berkisar dari 350 hingga 850.

5. `Geography` - Demografi pelanggan berdasarkan negara.

6. `Gender` - Jenis kelamin Nasabah Bank.

7. `Age` - Usia Nasabah Bank, berkisar dari 18 hingga 92 tahun.

8. `Tenure` - Durasi nasabah telah terkait atau memiliki rekening dengan bank. Dalam dataset ini, distribusi fitur ini berkisar dari 0 hingga 10 tahun.

9. `Balance` - Merujuk pada jumlah rata-rata uang yang disimpan di rekening nasabah. Ini adalah fitur kontinu dengan saldo minimum $0, dan saldo maksimum $250.898,09.

10. `NumOfProducts` - Ragam atau jumlah produk atau layanan keuangan yang dimiliki nasabah dengan Bank. Dalam dataset ini, bank memiliki total 4 produk.

11. `HasCrCard` - Klasifikasi biner apakah Nasabah Bank memiliki kartu kredit atau tidak.

12. `IsActiveMember` - fitur untuk klasifikasi nasabah; menunjukkan apakah nasabah masih menjadi anggota aktif di bank atau telah menjadi anggota pasif.

13. `EstimatedSalary` - Perkiraan atau prediksi pendapatan individu nasabah dalam sebulan. Dalam dataset ini, pendapatan berkisar dari $11,58 hingga $199.992,48.

14. `Exited` - Target atau output yang memberikan berbagai detail untuk menentukan apakah nasabah lebih cenderung tetap di bank atau telah keluar dari bank.

Berdasarkan Geografi, kita dapat memvisualisasikan distribusinya seperti:

![geo](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/7a13ccf8-c376-4ad8-a82f-4e648dc91c6d)

Selain itu, berikut adalah visualisasi distribusi fitur numerik dalam dataset:

![data-dist](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/b95b07f8-db0a-4cae-8a81-a90c3d4eb8b9)

Dari plot, dapat disimpulkan bahwa fitur 'CreditScore,' 'Age,' dan 'Balance' menunjukkan karakteristik yang membuatnya cocok untuk dikategorikan sebagai distribusi normal. Meskipun terdapat lonjakan pada fitur 'Balance' di 0, menunjukkan konsentrasi nilai, distribusi keseluruhan tampak mengikuti pola normal. Di sisi lain, fitur 'EstimatedSalary' tidak menunjukkan korelasi atau hubungan yang jelas dalam data. Pengamatan ini menunjukkan bahwa 'EstimatedSalary' mungkin tidak mengikuti pola distribusi yang jelas atau mungkin tidak memiliki tren yang dapat dikenali.

![outliers](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/9002256c-f367-4a11-844a-9e2778bfe1da)

Dapat dilihat bahwa beberapa fitur CreditScore dan Age menunjukkan adanya outlier. Untuk mengatasi hal ini, kita dapat menghapus outlier menggunakan metode Interquartile Range (IQR). Metode Interquartile Range (IQR) adalah teknik statistik yang digunakan untuk mengidentifikasi outlier dalam dataset. Ini didasarkan pada konsep kuartil, yang membagi dataset menjadi empat bagian sama. IQR adalah rentang antara kuartil pertama dan ketiga, dan mewakili 50% tengah data. Metode IQR mengidentifikasi outlier sebagai titik data yang berada di luar rentang 1,5 kali IQR di bawah kuartil pertama atau di atas kuartil ketiga.

Secara matematis, IQR dapat diungkapkan sebagai berikut:<br>
$IQR = Q3 − Q1$

di mana $Q1$ adalah kuartil pertama dan $Q3$ adalah kuartil ketiga. Batas bawah dan batas atas dapat diungkapkan sebagai berikut:

$Batas bawah= Q1 − 1.5 × IQR$<br>
$Batas atas= Q3 + 1.5 × IQR$

Metode IQR adalah teknik yang berguna untuk mengidentifikasi outlier dalam dataset, terutama ketika data tidak terdistribusi normal. Ini tahan terhadap nilai ekstrim dan dapat digunakan untuk mengidentifikasi outlier rendah dan tinggi. Namun, mungkin tidak efektif untuk dataset dengan ukuran sampel kecil atau ketika data sangat condong. Dalam kasus tersebut, metode deteksi outlier lainnya mungkin lebih sesuai."


## Data Preparation

Seperti yang disebutkan dalam pernyataan solusi, kami menggunakan beberapa teknik, termasuk:

### Mengatasi Data Tidak Seimbang dengan Resample

Untuk memberikan gambaran komprehensif tentang masalah data tidak seimbang, kami telah mulai dengan mengeksplorasi data target.

![Distribusi-Target](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/4becac2a-bd83-4222-8931-15d6f740e8c9)

Gambar di atas menunjukkan bahwa Data Target memiliki Data yang Tidak Seimbang, sehingga kami dapat mengatasinya dengan metode resample. **Over-sampling** acak, yang diklasifikasikan sebagai algoritma non-heuristik, bertujuan untuk mengatasi ketidakseimbangan kelas dengan secara acak menduplikasi instance dari target minoritas, sehingga mempromosikan distribusi yang lebih seimbang [^over-sampling-1]. Namun, pendekatan ini memiliki dua kelemahan. Pertama, itu meningkatkan risiko overfitting dengan menghasilkan reproduksi identik dari instance kelas minoritas [^over-sampling-1]. Kedua, itu memperparah sifat yang memakan waktu dari proses pembelajaran, terutama ketika dataset asli besar tetapi tidak seimbang, mencerminkan karakteristik dataset kami.

Fungsi resample() dari modul imbalanced-learn scikit-learn adalah alat yang nyaman untuk oversampling yang dapat diungkapkan sebagai:

$Data Resampled = resample(kelas minoritas, n-samples = jumlah yang diinginkan)$

Fungsi ini secara acak menduplikasi instance dari kelas minoritas untuk mencocokkan jumlah instance dalam kelas mayoritas, sehingga mengurangi ketidakseimbangan kelas.

### Encoding Fitur Kategori

Dalam bagian ini dari proyek, kami fokus pada pengkodean fitur kategorikal dalam dataset untuk memudahkan integrasi mereka ke dalam model pembelajaran mesin. Dengan mengonversi variabel kategorikal menjadi representasi biner, kami memastikan bahwa data berada dalam format yang sesuai untuk berbagai algoritma. Operasi inti melibatkan penggunaan metode pd.get_dummies() pada DataFrame 'churn'. Kolom kategorikal yang dikenakan one-hot encoding adalah 'Geography' dan 'Gender'. Parameter drop_first=True digunakan untuk mengabaikan tingkat kategori pertama selama encoding, mengurangi potensi masalah multicollinearity dalam analisis selanjutnya, meningkatkan ketangguhan analisis selanjutnya.

### Analisis Korelasi

Setelah mengkodekan data kategorikal, kita dapat melangkah lebih jauh ke langkah berikutnya dari analisis, yaitu menghasilkan matriks korelasi untuk secara komprehensif mengeksplorasi hubungan antara fitur dalam dataset. Matriks korelasi memberikan wawasan tentang bagaimana variabel, baik kategorikal maupun numerik, saling berhubungan. Terutama, selama analisis ini, diperhatikan bahwa korelasi antara fitur dan variabel target 'Exited' relatif rendah. Temuan ini menunjukkan bahwa dataset menunjukkan seperangkat fitur yang beragam dengan tingkat pengaruh yang berbeda pada variabel target.

Analisis korelasi menilai hubungan linear antara pasangan variabel dalam dataset. Matriks korelasi memberikan gambaran komprehensif tentang hubungan ini. Fungsi heatmap() dari pustaka seaborn sering digunakan untuk memvisualisasikan matriks korelasi. Secara matematis, korelasi (koefisien korelasi Pearson, $ρ$) antara variabel $X$ dan $Y$ dapat diungkapkan sebagai:

$ρ\left(X, Y\right)=\frac{Cov\left(X, Y\right)​}{σ_X⋅ \ σ_Y}$

Heatmap memvisualisasikan koefisien korelasi ini, dengan warna yang lebih hangat menunjukkan korelasi positif yang lebih kuat, warna yang lebih dingin menunjukkan korelasi negatif yang lebih kuat, dan warna netral untuk korelasi yang lebih lemah.

### Pembagian Data Latih dan Uji

Setelah analisis korelasi, dataset menjalani langkah penting yang dikenal sebagai pembagian data latih-uji. Proses ini melibatkan pembagian dataset menjadi dua subset: dataset latihan dan dataset uji. Memisahkan dataset menjadi set latihan dan pengujian penting untuk mengevaluasi kinerja model. Pembagian dilakukan dengan rasio 80-20, mengalokasikan 80% data untuk melatih model pembelajaran mesin dan menyisakan 20% sisanya untuk mengevaluasi kinerja model. Pendekatan ini membantu memastikan bahwa model dilatih pada dataset yang cukup besar sambil tetap memiliki set data independen untuk validasi dan pengujian. Lebih lanjut, pembagian ini memastikan bahwa model dilatih pada dataset yang cukup besar sambil tetap memiliki dataset independen untuk evaluasi model yang tidak bias.

### Penskalaan Fitur

Setelah pembagian data latih-uji, fitur numerik dalam dataset dikenakan penskalaan fitur menggunakan StandardScaler. Penskalaan fitur adalah langkah pra-pemrosesan penting yang menstandarisasi rentang fitur numerik, membawa mereka ke skala yang konsisten. Dalam kasus ini, penskalaan dilakukan untuk membatasi fitur numerik dalam rentang 0 hingga 1. Standarisasi fitur membantu mencegah fitur dengan skala lebih besar mendominasi proses pemodelan, memastikan pertimbangan yang adil terhadap semua fitur selama pelatihan dan evaluasi model. Penggunaan StandardScaler adalah praktik umum untuk mencapai normalisasi ini.

Penskalaan fitur memastikan bahwa fitur numerik berada pada skala yang serupa, mencegah beberapa fitur mendominasi proses pelatihan model. StandardScaler() dari scikit-learn menskalakan fitur dengan mentransformasikannya agar memiliki mean $(μ)$ 0 dan standar deviasi $(σ)$ 1. Matematis, untuk suatu fitur $X$:

$X_{scaled}=\frac{X − μ​}{σ}$

Transformasi ini menjaga perbedaan relatif antara nilai fitur sambil menempatkannya pada skala yang dapat dibandingkan, meningkatkan stabilitas dan konvergensi model pembelajaran mesin selama pelatihan.


## Model Development

### **K-Nearest Neighbourhood Algorithm**

Algoritma K-Nearest Neighbors (KNN) adalah jenis pembelajaran berbasis contoh, di mana fungsi hanya diperkirakan secara lokal, dan semua komputasi ditangguhkan hingga evaluasi fungsi. Ini adalah algoritma sederhana dan efektif untuk tugas klasifikasi dan regresi. Algoritma KNN didasarkan pada prinsip bahwa titik data yang mirip berada dekat satu sama lain dalam ruang fitur. Secara matematis, algoritma KNN memprediksi klasifikasi suatu titik data berdasarkan kelas mayoritas dari K tetangga terdekatnya. Jarak antara titik data umumnya dihitung menggunakan metode seperti jarak Euclidean atau jarak Manhattan [^knn1]. Prediksi untuk titik data baru x didasarkan pada suara mayoritas dari k tetangga terdekatnya, dan dapat diungkapkan sebagai:

$\hat{y}^​ = mayoritas suara\left(y_1, y_2, ..., y_k\right)$

di mana $\hat{y}$ adalah kelas yang diprediksi untuk $x$, dan $y_1, y_2, ..., y_k$ adalah kelas dari k tetangga terdekat [^knn2]. Algoritma KNN diterapkan, dan penyetelan hiperparameter dilakukan menggunakan GridSearchCV. Model mencapai akurasi sebesar 81,24%.


|              | precision | recall  | f1-score | support     |
|--------------|-----------|---------|----------|-------------|
| Stay         | 0.858744  | 0.748047| 0.799582 | 1536.000000 |
| Exit         | 0.776688  | 0.876873| 0.823745 | 1535.000000 |
| accuracy     | 0.812439  | 0.812439| 0.812439 | 0.812439    |
| macro avg    | 0.817716  | 0.812460| 0.811664 | 3071.000000 |
| weighted avg | 0.817729  | 0.812439| 0.811660 | 3071.000000 |


### **Logistic Regression Algorithm**

Regresi Logistik adalah model statistik yang menggunakan fungsi logistik untuk memodelkan probabilitas hasil biner. Ini banyak digunakan untuk masalah klasifikasi biner. Fungsi logistik, juga dikenal sebagai fungsi sigmoid, didefinisikan sebagai:

$σ\left(z\right)=\frac{1}{1+e^{-z}}$

di mana z adalah kombinasi linear dari fitur input dan parameter model. Algoritma regresi logistik meminimalkan fungsi kerugian logistik untuk menemukan parameter model terbaik. Ini adalah model linear dan membuat prediksi berdasarkan jumlah tertimbang dari fitur input [^lr1]. Probabilitas hasil berada dalam kelas tertentu diberikan oleh fungsi logistik, dan prediksi dapat diungkapkan sebagai:

$P\left(Y=1∣X\right)=\frac{1}{1+e^{−\left(β_0​+β_1​X_1​+...+β_p​X_p​\right)}}$

di mana $P(Y=1|X)$ adalah probabilitas hasil berada dalam kelas 1 dengan fitur input $X$, $\beta_0, \beta_1, ..., \beta_p$ adalah parameter model, dan $X_1, X_2, ..., X_p$ adalah fitur input [^lr2]. Regresi Logistik diimplementasikan, dan penyetelan hiperparameter dilakukan menggunakan GridSearchCV. Model mencapai akurasi sebesar 72,48%.


|              | precision | recall  | f1-score | support     |
|--------------|-----------|---------|----------|-------------|
| Stay         | 0.719644  | 0.736979| 0.728208 | 1536.000000 |
| Exit         | 0.730307  | 0.712704| 0.721398 | 1535.000000 |
| accuracy     | 0.724845  | 0.724845| 0.724845 | 0.724845    |
| macro avg    | 0.724976  | 0.724841| 0.724803 | 3071.000000 |
| weighted avg | 0.724974  | 0.724845| 0.724804 | 3071.000000 |


### Support Vector Classifier Algorithm

Support Vector Classifier (SVC) adalah algoritma pembelajaran terawasi yang dapat digunakan untuk tugas klasifikasi dan regresi. Dalam konteks klasifikasi, algoritma SVC menemukan hiperrata terbaik yang memisahkan data ke dalam kelas yang berbeda. Ini sangat efektif dalam ruang berdimensi tinggi. Algoritma bekerja dengan menemukan hiperrata margin maksimum, yang merupakan hiperrata yang memaksimalkan jarak ke titik data terdekat dari setiap kelas. Secara matematis, tujuan dari algoritma SVC adalah untuk memecahkan masalah optimasi yang memaksimalkan margin dan meminimalkan kesalahan klasifikasi [^svc1]. Fungsi keputusan untuk SVC diberikan oleh:

$f\left(x\right)=sign\left(\sum_{i=1}^{n}​α_i​y_i​K\left(x,x_i\right)+b\right)$

di mana $f(x)$ adalah fungsi keputusan, $α_i$ adalah multiplikator Lagrange yang dipelajari, $y_i$ adalah label kelas, $K(x, x_i)$ adalah fungsi kernel, dan b adalah istilah bias [^svc1]. SVC diterapkan, dan penyetelan hiperparameter dilakukan menggunakan GridSearchCV. Model mencapai akurasi yang mengesankan sebesar 97,62%.


|              | precision | recall    | f1-score  | support     |
|--------------|-----------|-----------|-----------|-------------|
| Stay         | 0.954630  | 1.000000  | 0.976789  | 1536.000000 |
| Exit         | 1.000000  | 0.952443  | 0.975642  | 1535.000000 |
| accuracy     | 0.976229  | 0.976229  | 0.976229  | 0.976229    |
| macro avg    | 0.977315  | 0.976221  | 0.976215  | 3071.000000 |
| weighted avg | 0.977308  | 0.976229  | 0.976216  | 3071.000000 |


### Random Forest Algorithm

Algoritma Random Forest adalah metode pembelajaran ensemble yang beroperasi dengan membangun banyak pohon keputusan pada saat pelatihan dan mengeluarkan kelas yang merupakan modus dari kelas-kelas (klasifikasi) atau prediksi rata-rata (regresi) dari pohon-pohon individu. Random forest mengoreksi kebiasaan pohon keputusan yang cenderung overfitting pada set pelatihan mereka. Algoritma memasukkan unsur keacakan saat membangun pohon, yang menghasilkan set pohon yang beragam. Selama prediksi, random forest menggabungkan prediksi pohon-pohon individu untuk membuat prediksi final. Secara matematis, algoritma menggabungkan prediksi dari beberapa pohon keputusan untuk meningkatkan generalitas dan kekokohan [^rf1]. Prediksi random forest untuk titik data baru dapat diungkapkan sebagai modus dari prediksi pohon-pohon individu:

$\hat{Y}^​ = model\left(Y_1, Y_2, ..., Y_n\right)$

di mana $\hat{Y}$ adalah kelas yang diprediksi untuk titik data baru, dan $Y_1, Y_2, ..., Y_n$ adalah prediksi pohon-pohon individu [^rf2]. Random Forest dikembangkan, dan penyetelan hiperparameter dilakukan menggunakan GridSearchCV. Model mencapai akurasi sebesar 94,89%.


|              | precision | recall    | f1-score  | support     |
|--------------|-----------|-----------|-----------|-------------|
| Stay         | 0.977163  | 0.919271  | 0.947333  | 1536.000000 |
| Exit         | 0.923739  | 0.978502  | 0.950332  | 1535.000000 |
| accuracy     | 0.948877  | 0.948877  | 0.948877  | 0.948877    |
| macro avg    | 0.950451  | 0.948886  | 0.948833  | 3071.000000 |
| weighted avg | 0.950460  | 0.948877  | 0.948832  | 3071.000000 |


## Model Evaluation

### Evaluation Metrics

Presisi, recall, dan F1-score adalah metrik penting untuk mengevaluasi kinerja model klasifikasi. Metrik-metrik ini sangat berguna dalam menilai kemampuan model untuk mengidentifikasi kasus positif dengan benar dan menghindari kesalahan klasifikasi.

#### Precision
Presisi adalah rasio prediksi positif benar terhadap total prediksi positif yang dibuat oleh model. Ini dihitung menggunakan rumus berikut:

$\text{Presisi} = \frac{TP}{TP + FP}$

Dimana:<br>
$(TP)$ adalah jumlah prediksi positif benar (kasus positif yang diprediksi dengan benar).<br>
$(FP)$ adalah jumlah prediksi positif palsu (kasus negatif yang salah diklasifikasikan sebagai positif).

Presisi adalah ukuran akurasi dari prediksi positif. Nilai presisi tinggi menunjukkan bahwa ketika model memprediksi kasus positif, kemungkinan besar prediksinya benar.

### Recall

Recall, juga dikenal sebagai sensitivitas, adalah rasio prediksi positif benar terhadap total jumlah kasus positif sebenarnya dalam dataset. Ini dihitung menggunakan rumus berikut:

$\text{Recall} = \frac{TP}{TP + FN}$

Dimana:<br>
$(FN)$ adalah jumlah prediksi negatif palsu (kasus positif yang salah diklasifikasikan sebagai negatif).

Recall mengukur kemampuan model untuk mengidentifikasi semua kasus positif sebenarnya. Nilai recall tinggi menunjukkan bahwa model dapat mengidentifikasi proporsi besar kasus positif dalam dataset.

### F1-score

F1-score adalah rata-rata harmonis dari presisi dan recall, dan memberikan keseimbangan antara kedua metrik tersebut. Ini dihitung menggunakan rumus berikut:

$F_{\beta }=\left(1+\beta ^2\right)\frac{precision\times recall}{\beta ^2precision+recall}$

F1-score memperhitungkan baik positif palsu maupun negatif palsu, sehingga menjadi metrik yang berguna untuk dataset yang tidak seimbang di mana jumlah kasus negatif jauh lebih besar daripada jumlah kasus positif. Nilai F1-score tinggi menunjukkan bahwa model memiliki presisi dan recall yang baik. Metrik-metrik ini umumnya digunakan untuk mengevaluasi kinerja model klasifikasi, terutama ketika distribusi kelas tidak seimbang. Mereka memberikan wawasan berharga tentang seberapa baik model berkinerja dalam mengidentifikasi kasus positif dan menghindari kesalahan klasifikasi.



### Confusion Matrix

Matriks kebingungan untuk setiap model dihasilkan, menunjukkan kinerja yang baik dalam memprediksi nilai positif dan nilai negatif sejati.

![matriks-kebingungan](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/b7d1790f-765c-45a9-b437-05b24bd37591)

Matriks kebingungan adalah tabel yang merangkum kinerja model klasifikasi dengan membandingkan label kelas yang diprediksi dan aktual. Ini terdiri dari empat nilai: true positive (TP), false positive (FP), true negative (TN), dan false negative (FN).


### Model Comparison

Dalam hal kelebihan dan kekurangan, algoritma KNN sederhana dan mudah diimplementasikan, tetapi dapat menjadi mahal secara komputasional untuk dataset besar dan memerlukan pemilihan nilai K yang hati-hati. Regresi Logistik adalah model linear yang mudah diinterpretasi dan dapat menangani fitur input biner dan kontinu, tetapi mungkin tidak berperforma baik ketika hubungan antara fitur input dan output bersifat non-linear. Algoritma SVC efektif dalam ruang berdimensi tinggi dan dapat menangani batas keputusan non-linear, tetapi dapat sensitif terhadap pilihan fungsi kernel dan hiperparameter. Algoritma Random Forest tahan terhadap overfitting dan dapat menangani fitur input kategori dan kontinu, tetapi dapat menjadi mahal secara komputasional dan sulit diinterpretasi.


|       | train     | test      |
|-------|-----------|-----------|
| KNN   | 88.773101 | 81.243894 |
| LR    | 73.044045 | 72.484533 |
| SVC   | 100.0     | 97.622924 |
| RF    | 100.0     | 94.887659 |

Perbandingan akurasi model menunjukkan bahwa Support Vector Classifier (SVC) mencapai akurasi tertinggi baik pada set pelatihan maupun uji.

![model-comparison](https://github.com/naufalmuafi/bank-customer-churn-prediction/assets/72964378/1e82fe2c-8559-4acd-85e4-748ec01f22fe)


## Model Prediction

Untuk menguji model, prediksi dihasilkan menggunakan data sampel. Algoritma SVC secara konsisten memberikan hasil terbaik.

|      | y_true | prediction_KNN | prediction_LR | prediction_SVC | prediction_RF |
|------|--------|-----------------|----------------|-----------------|----------------|
| 5709 | 0      | 1               | 0              | 0               | 0              |
| 3207 | 0      | 1               | 1              | 0               | 0              |
| 8843 | 0      | 0               | 0              | 0               | 0              |
| 2171 | 0      | 0               | 1              | 0               | 0              |
| 1854 | 1      | 1               | 1              | 1               | 1              |


### Sample Predictions:

1. Pelanggan dengan fitur (CreditScore: 815, Geography: Spain, Gender: Female, Age: 39, ...):
   - Diprediksi: Pelanggan lebih cenderung Bertahan


## Conclusion

Support Vector Classifier (SVC) menunjukkan kinerja unggul dalam memprediksi keluaran pelanggan bank dibandingkan dengan model lain. Model mencapai akurasi sebesar 97,62%, menjadikannya pilihan yang dapat diandalkan untuk diimplementasikan.


## References

[^1]: [Xie, Y., Li, X., Ngai, E. W. T., & Ying, W. (2009). Customer churn prediction using improved balanced random forests. Expert Systems with Applications, 36(3), 5445-5449.](https://www.sciencedirect.com/science/article/pii/S0957417408004326)
[^2]: [Tran, H., Le, N., & Nguyen, V. H. (2023). CUSTOMER CHURN PREDICTION IN THE BANKING SECTOR USING MACHINE LEARNING-BASED CLASSIFICATION MODELS. Interdisciplinary Journal of Information, Knowledge & Management, 18.](https://www.researchgate.net/publication/368911804_Customer_Churn_Prediction_in_the_Banking_Sector_Using_Machine_Learning-Based_Classification_Models)
[^3]: [Cohen, D. A., Gan, C., Hwa, A., & Chong, E. Y. (2006). Customer satisfaction: a study of bank customer retention in New Zealand.](https://researcharchive.lincoln.ac.nz/items/cecd1d6f-5d98-4522-a730-9b65e0c7adad)
[^over-sampling-1]: [Mohammed, R., Rawashdeh, J., & Abdullah, M. (2020, April). Machine learning with oversampling and undersampling techniques: overview study and experimental results. In 2020 11th international conference on information and communication systems (ICICS) (pp. 243-248). IEEE.](https://ieeexplore.ieee.org/abstract/document/9078901?casa_token=zwQWkVHTTbYAAAAA:Sr0rIrgCaloLp83pnimWRu2Tx8C0E_2u1Pw6whfmiv3GQW7_9bbm2ennh4JAjxzwGSmXYkFeVi1_)
[^knn1]: [Ghunimat, D.M., Alzoubi, A.E., Alzboon, A.A., & Hanandeh, S. (2022). Prediction of concrete compressive strength with GGBFS and fly ash using multilayer perceptron algorithm, random forest regression and k-nearest neighbor regression. Asian Journal of Civil Engineering, 24, 169-177.](https://www.semanticscholar.org/paper/Prediction-of-concrete-compressive-strength-with-Ghunimat-Alzoubi/2bb0f80d2914eeeccdbf156c2d0c12fd1a0b2b5b)
[^knn2]: [Adeshina, A.M. (2023). Prediction of Diabetes Mellitus using Machine Learning Algorithms: Comparative Analysis of K-Nearest Neighbor, Random Forest and Logistic Regression. SLU Journal of Science and Technology.](https://www.semanticscholar.org/paper/Prediction-of-Diabetes-Mellitus-using-Machine-of-Adeshina/79fe14595faeaab90b6fe242d86f49e118e9d750)
[^lr1]: [Adeshina, A.M. (2023). Prediction of Diabetes Mellitus using Machine Learning Algorithms: Comparative Analysis of K-Nearest Neighbor, Random Forest and Logistic Regression. SLU Journal of Science and Technology.](https://www.semanticscholar.org/paper/Prediction-of-Diabetes-Mellitus-using-Machine-of-Adeshina/79fe14595faeaab90b6fe242d86f49e118e9d750)
[^lr2]: [Das, S., Bhattacharyya, K., & Sarkar, S. (2023). Performance Analysis of Logistic Regression, Naive Bayes, KNN, Decision Tree, Random Forest and SVM on Hate Speech Detection from Twitter. International Research Journal of Innovations in Engineering and Technology.](https://www.semanticscholar.org/paper/Performance-Analysis-of-Logistic-Regression%2C-Naive-Das-Bhattacharyya/43b6d317c5ecf72d4bf6bd46e182b2b5fc97d43b)
[^svc1]: [Afrianto, M.A., & Wasesa, M. (2020). Booking Prediction Models for Peer-to-peer Accommodation Listings using Logistics Regression, Decision Tree, K-Nearest Neighbor, and Random Forest Classifiers. Journal of Information Systems Engineering and Business Intelligence.](https://www.semanticscholar.org/paper/Booking-Prediction-Models-for-Peer-to-peer-Listings-Afrianto-Wasesa/4dd5ae54caac18ee0efe38dd4a704e9e2e8c4cf2)
[^rf1]: [Afrianto, M.A., & Wasesa, M. (2020). Booking Prediction Models for Peer-to-peer Accommodation Listings using Logistics Regression, Decision Tree, K-Nearest Neighbor, and Random Forest Classifiers. Journal of Information Systems Engineering and Business Intelligence.](https://www.semanticscholar.org/paper/Booking-Prediction-Models-for-Peer-to-peer-Listings-Afrianto-Wasesa/4dd5ae54caac18ee0efe38dd4a704e9e2e8c4cf2)
[^rf2]: [Mohsin, M.A., & Hamad, A.H. (2022). Performance Evaluation of SDN DDoS Attack Detection and Mitigation Based Random Forest and K-Nearest Neighbors Machine Learning Algorithms. Revue d'Intelligence Artificielle.](https://www.semanticscholar.org/paper/Performance-Evaluation-of-SDN-DDoS-Attack-Detection-Mohsin-Hamad/9b5b52f3e6a80328a227898695bd6f02c4ddb39e)
