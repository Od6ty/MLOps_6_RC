# CNN Flowers Classification

## 📌 Project Overview
Proyek ini merupakan implementasi _end-to-end_ MLOps untuk klasifikasi citra bunga menggunakan _Convolutional Neural Network_ (CNN) dengan arsitektur _EfficientNetB0_ berbasis _TensorFlow_ dan _Keras_. Model yang dibangun mampu mengklasifikasikan citra ke dalam lima jenis bunga, yaitu Orchid, Tulip, Lily, Sunflower, dan Lotus. Proyek ini dikembangkan untuk memenuhi kriteria _submission_, dengan menggunakan dataset minimal 1000 citra, pembagian data ke dalam _train_, _validation_, dan _test_, serta penerapan model _Sequential_. Model yang dihasilkan disimpan dalam berbagai format untuk kebutuhan deployment, yaitu _TensorFlow SavedModel_, _TensorFlow Lite_ (TFLite), dan _TensorFlow.js_ (TFJS), sehingga mendukung implementasi lintas _platform_ dan integrasi ke dalam _pipeline_ MLOps.

## 📊 Dataset
**Nama Dataset**  : 5 Flower Types Classification Dataset

**Sumber Data**   : Kaggle

**URL Dataset**   : https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset

**Kategori Kelas**
- Orchid

- Tulip
  
- Lily
  
- Sunflower
  
- Lotus

**Jumlah Data**
- Orchid: 1.000 citra
  
- Tulip: 1.000 citra
  
- Lily: 999 citra

- Sunflower: 1.000 citra

- Lotus: 1.000 citra

Total keseluruhan: 4.999 citra (memenuhi kriteria minimal ≥ 1.000 citra)

**Karakteristik Citra**
- Resolusi citra bervariasi pada data asli

- Seluruh citra diproses dengan resize ke 180 × 180 piksel saat tahap pemuatan dataset

- Normalisasi dilakukan untuk menyesuaikan input model CNN

## 🏗️ Architecture Overview
```
+--------------------+
|   Flower Image     |
|     Dataset        |
| (≥ 1000 Images)    |
+----------+---------+
           |
           v
+--------------------+
| Data Preprocessing |
| - Resize           |
| - Normalization    |
| - Augmentation     |
+----------+---------+
           |
           v
+--------------------+
| Data Splitting     |
| Train / Val / Test |
+----------+---------+
           |
           v
+-----------------------------+
| Model Training              |
| CNN - EfficientNetB0        |
| (TensorFlow & Keras)        |
+----------+------------------+
           |
           v
+-----------------------------+
| Model Evaluation            |
| - Accuracy                  |
| - Validation Metrics        |
+----------+------------------+
           |
           v
+-----------------------------+
| Model Export & Versioning   |
| - SavedModel                |
| - TFLite                    |
| - TFJS                      |
+----------+------------------+
           |
           v
+-----------------------------+
| Deployment & Integration    |
| - API / Application         |
| - Multi-platform Support   |
+----------+------------------+
           |
           v
+-----------------------------+
| MLOps Pipeline              |
| - Automation                |
| - Reproducibility           |
| - Continuous Improvement   |
+-----------------------------+
```
