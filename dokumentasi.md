# Submission 1: Nama Proyek Anda
Nama:

Username dicoding:

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) |
| Masalah | Masalah yang diangkat adalah identifikasi dan prediksi kemungkinan depresi pada pelajar berdasarkan berbagai faktor psikososial dan akademik. Dengan meningkatnya kasus depresi pada pelajar, diperlukan sistem berbasis machine learning untuk melakukan prediksi awal. |
| Solusi machine learning | Model machine learning dibuat dapat mengklasifikasikan pelajar ke dalam kategori depresi atau tidak berdasarkan fitur-fitur yang tersedia. Solusi ini membantu dalam memberikan wawasan awal terhadap kondisi mental pelajar. |
| Metode pengolahan | Data diproses dan diolah menggunakan TensorFlow Transform, yang melakukan proses tranformasi nama fitur, pemisaham fitur kategorikal dan numerik, normalisasi untuk fitur numerik, dan encoding untuk fitur kategorical. |
| Arsitektur model | Arsitektur model yang digunakan adalah Neural Network dengan beberapa hidden layers, menggunakan ReLU sebagai aktivasi dan sigmoid pada output untuk klasifikasi biner. Model dilatih dengan binary_crossentropy loss dan optimizer Adam. |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi performa model diantaranya ExampleCount, AUC, MeanPrediction, Precision, Recall, FalsePositives, TruePositives, FalseNegatives, TrueNegatives, dan BinaryAccuracy. |
| Performa model | Performa dari model mencapai akurasi di atas 80% pada data latih dan validasi, sehingga model dapat menunjukkan kemampuan yang cukup baik dalam mengklasifikasikan sentimen ulasan pengguna terhadap aplikasi Dana. |
| Opsi deployment | Model dideploy menggunakan TensorFlow Serving  dan docker melalui platform Railway. |
| Web app | oryshion depression [serving_model](https://oryshion-depression-pipeline-production.up.railway.app/v1/models/serving_model/metadata)|
| Monitoring | Model yang telah di deploy berhasil di monitoring dan menggunakan prometheus |