## Membangun Sistem Machine Learning untuk Memprediksi Risiko Stroke

Proyek ini bertujuan untuk membangun sistem machine learning yang dapat memprediksi risiko stroke berdasarkan data medis dan demografis. Sistem dikembangkan dengan menerapkan prinsip MLOps agar setiap tahap mulai dari pengolahan data hingga pemantauan model bersifat modular, terstruktur, dan siap diterapkan di lingkungan produksi.

### Pengolahan Data
Tahap eksplorasi dan pembersihan data dilakukan di repositori [Eksperimen SML Fitria Anggraini](https://github.com/ftriaa/Eksperimen_SML_Fitria-Anggraini.git). Proses ini mencakup:
- Penanganan data kosong
- Encoding fitur
- Penyeimbangan data
- Seleksi fitur dan analisis korelasi

Langkah ini memastikan data yang digunakan untuk pelatihan model dalam kondisi bersih dan informatif.

### Pengembangan Model
Beberapa model machine learning dibangun dan dievaluasi pada repositori [Membangun Model](https://github.com/ftriaa/SMSML_Stroke-Prediction/tree/2ad64a41cba6ab111c8e0b97c5d617b37a6af32d/Membangun%20Model). Aktivitas yang dilakukan antara lain:
- Pembuatan model Naive Bayes, SVM, LightGBM, Random Forest dan XGBoost
- Tuning hyperparameter
- Evaluasi menggunakan metrik seperti akurasi, presisi, recall, dan F1 score
- Menyimpan model dengan performa terbaik

### Automasi Workflow CI
Untuk memastikan alur kerja yang otomatis dan konsisten, pipeline Continuous Integration dikembangkan di repositori [Workflow CI](https://github.com/ftriaa/Workflow-CI.git). Tahapan dalam alur kerja ini meliputi:
- Validasi dan pengecekan kode
- Pengujian otomatis
- Pelatihan dan penyimpanan model yang dipicu setiap kali ada pembaruan

Langkah ini membantu menjaga kualitas dan keandalan sistem dalam setiap iterasi.

### Pemantauan dan Logging
Setelah model diterapkan, performanya dipantau secara real time melalui fitur yang dikembangkan di bagian [Monitoring dan Logging](https://github.com/ftriaa/SMSML_Stroke-Prediction/tree/2ad64a41cba6ab111c8e0b97c5d617b37a6af32d/Monitoring%20%26%20Logging). Fitur ini mencakup:
- Pencatatan hasil prediksi dan error log
- Visualisasi performa model dari waktu ke waktu
- Deteksi potensi masalah atau perubahan distribusi data

