# MLOps Flowers Classification

## 👥 Tim

**Kelompok 6 MLOps RC**
- Rani Puspita Sari
- Ratu Keisha
- Anwar Muslim
- Yohana Manik

Proyek ini merupakan implementasi **end-to-end MLOps pipeline** untuk klasifikasi citra bunga menggunakan teknik machine learning tradisional. Proyek ini mengklasifikasikan citra ke dalam lima jenis bunga: **Orchid**, **Tulip**, **Lily**, **Sunflower**, dan **Lotus**.

### Alur Kerja Proyek:

1. **Data Preprocessing**: Memuat ~5000 citra bunga, melakukan resize ke 128×128 piksel, normalisasi, dan ekstraksi fitur (flatten)
2. **Model Training**: Melatih dua model machine learning:
   - **SVM (Support Vector Machine)** dengan kernel RBF
   - **Naive Bayes** (Gaussian)
3. **Model Registry**: Sistem manajemen model dengan konsep canary deployment (model canary untuk testing, model production untuk serving)
4. **API Serving**: REST API menggunakan FastAPI untuk inference real-time
5. **Monitoring**: Tracking prediksi, metrik drift, dan performa model

### Fitur Utama:

- **Canary Deployment**: Model baru diuji sebagai canary sebelum dipromosikan ke production
- **Shadow Testing**: Model canary dievaluasi secara paralel dengan production (10% traffic)
- **Model Monitoring**: Tracking distribusi kelas, confidence, dan indikasi data drift
- **CLI Interface**: Command-line interface untuk training, registry, serving, dan monitoring

## 📊 Dataset

**Nama Dataset**: 5 Flower Types Classification Dataset

**Sumber**: [Kaggle](https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)

**Kategori Kelas**:
- Orchid: 1.000 citra
- Tulip: 1.000 citra
- Lily: 999 citra
- Sunflower: 1.000 citra
- Lotus: 1.000 citra

**Total**: 4.999 citra

**Karakteristik**:
- Resolusi citra bervariasi pada data asli
- Diproses dengan resize ke 128 × 128 piksel
- Normalisasi pixel values ke range [0, 1]
- Flatten menjadi array 1D (49,152 fitur per gambar)

## 🛠️ Tools yang Digunakan

### 1. **NumPy** (`numpy`)

**Kegunaan**:
- Operasi array dan matriks untuk data numerik
- Stacking gambar menjadi array multidimensi
- Operasi matematika pada data gambar (normalisasi, reshape)

**Digunakan di**:
- `src/features/preprocessing.py`: Stacking gambar, operasi array
- `src/serving/inference.py`: Preprocessing gambar untuk inference
- `src/monitoring/drift.py`: Perhitungan statistik metrik

**Jika Tidak Ada**:
- ❌ Tidak bisa melakukan operasi array multidimensi
- ❌ Tidak bisa stack gambar menjadi dataset
- ❌ Operasi matematika pada data numerik menjadi sangat lambat
- ❌ Proyek tidak bisa berjalan sama sekali

**Alternatif**:
- **PyTorch Tensor**: Bisa digunakan untuk operasi array, tapi lebih berat
- **JAX**: Alternatif modern untuk NumPy dengan GPU support
- **CuPy**: NumPy-compatible untuk GPU (lebih kompleks setup)
- **Manual Python List**: Tidak praktis untuk dataset besar (sangat lambat)

---

### 2. **SciPy** (`scipy`)

**Kegunaan**:
- Operasi scientific computing tambahan
- Dependensi dari scikit-learn (tidak langsung digunakan di kode)

**Digunakan di**:
- Dependency dari scikit-learn (tidak digunakan langsung)

**Jika Tidak Ada**:
- ⚠️ scikit-learn tidak bisa diinstall atau error saat import
- ⚠️ Beberapa fungsi scikit-learn mungkin tidak berfungsi

**Alternatif**:
- **NumPy saja**: Untuk operasi dasar, tapi scikit-learn tetap butuh SciPy
- **Tidak ada alternatif praktis**: SciPy adalah dependency wajib untuk scikit-learn

---

### 3. **Pandas** (`pandas`)

**Kegunaan**:
- Data manipulation dan analisis
- Struktur DataFrame untuk data tabular

**Digunakan di**:
- Tidak langsung digunakan di kode (opsional untuk analisis data)

**Jika Tidak Ada**:
- ✅ Proyek tetap bisa berjalan (tidak digunakan langsung)
- ⚠️ Jika ingin melakukan analisis data tambahan, perlu install

**Alternatif**:
- **Polars**: Lebih cepat untuk dataset besar
- **Dask**: Untuk data yang lebih besar dari memori
- **PyArrow**: Untuk operasi data yang lebih efisien
- **Manual Python dict/list**: Untuk data kecil, tapi kurang praktis

---

### 4. **Scikit-learn** (`scikit-learn`)

**Kegunaan**:
- Machine learning algorithms (SVM, Naive Bayes)
- Data preprocessing (StandardScaler, train_test_split)
- Model evaluation (accuracy_score, classification_report)

**Digunakan di**:
- `src/models/train.py`: Training SVM dan Naive Bayes
- `src/features/preprocessing.py`: Data splitting dan scaling
- `src/monitoring/drift.py`: Perhitungan accuracy

**Jika Tidak Ada**:
- ❌ Tidak bisa training model (SVM, Naive Bayes)
- ❌ Tidak bisa preprocessing data (scaling, splitting)
- ❌ Tidak bisa evaluasi model
- ❌ Proyek tidak bisa berjalan sama sekali

**Alternatif**:
- **XGBoost / LightGBM**: Untuk model tree-based (perlu modifikasi kode)
- **TensorFlow / PyTorch**: Untuk deep learning (perlu rewrite besar)
- **Statsmodels**: Untuk model statistik (tidak cocok untuk image classification)
- **Manual Implementation**: Tidak praktis (SVM sangat kompleks)

---

### 5. **Pillow (PIL)** (`Pillow`)

**Kegunaan**:
- Membaca dan memproses file gambar
- Resize gambar ke ukuran yang diinginkan
- Convert format gambar (RGB, grayscale, dll)

**Digunakan di**:
- `src/features/preprocessing.py`: Load dan resize gambar saat preprocessing
- `src/serving/inference.py`: Load dan resize gambar saat inference

**Jika Tidak Ada**:
- ❌ Tidak bisa membaca file gambar (.jpg, .png)
- ❌ Tidak bisa resize gambar
- ❌ Tidak bisa convert format gambar
- ❌ Proyek tidak bisa berjalan sama sekali

**Alternatif**:
- **OpenCV (cv2)**: Lebih powerful untuk image processing, tapi lebih kompleks
- **imageio**: Lebih sederhana, tapi fitur lebih terbatas
- **scikit-image**: Untuk image processing scientific (lebih berat)
- **Wand (ImageMagick binding)**: Sangat powerful, tapi lebih kompleks

---

### 6. **FastAPI** (`fastapi`)

**Kegunaan**:
- Web framework untuk membuat REST API
- Automatic API documentation (Swagger UI)
- Request validation dengan Pydantic
- Async support untuk performa tinggi

**Digunakan di**:
- `src/serving/api.py`: Endpoint API untuk prediction dan monitoring

**Jika Tidak Ada**:
- ❌ Tidak bisa membuat REST API
- ❌ Tidak bisa serve model untuk inference
- ❌ Tidak bisa deployment model ke production
- ⚠️ Proyek bisa training, tapi tidak bisa serving

**Alternatif**:
- **Flask**: Framework web tradisional (sync, lebih lambat)
- **Django**: Framework web full-stack (overkill untuk API saja)
- **Starlette**: Framework yang lebih rendah level (FastAPI dibangun di atasnya)
- **Tornado**: Async web framework (kurang modern)
- **Sanic**: Async framework yang cepat (kurang populer)

---

### 7. **Uvicorn** (`uvicorn[standard]`)

**Kegunaan**:
- ASGI server untuk menjalankan FastAPI
- HTTP server yang cepat dan async
- Support untuk WebSocket dan HTTP/2

**Digunakan di**:
- `cli/main.py`: Menjalankan FastAPI server via uvicorn

**Jika Tidak Ada**:
- ❌ Tidak bisa menjalankan FastAPI server
- ❌ Tidak bisa serve API untuk inference
- ⚠️ FastAPI tidak bisa dijalankan (butuh ASGI server)

**Alternatif**:
- **Gunicorn + Uvicorn workers**: Untuk production (lebih robust)
- **Hypercorn**: ASGI server alternatif (kurang populer)
- **Daphne**: ASGI server dari Django (kurang optimal untuk FastAPI)
- **Manual HTTP server**: Tidak praktis (harus implement dari scratch)

---

### 8. **Joblib** (`joblib`)

**Kegunaan**:
- Serialisasi dan deserialisasi model machine learning
- Efficient untuk object Python besar (numpy arrays)
- Parallel processing support

**Digunakan di**:
- `src/models/train.py`: Menyimpan model setelah training
- `src/serving/inference.py`: Memuat model saat startup API

**Jika Tidak Ada**:
- ❌ Tidak bisa menyimpan model ke disk
- ❌ Tidak bisa memuat model untuk inference
- ❌ Model harus di-train ulang setiap kali (tidak praktis)
- ❌ Proyek tidak bisa berjalan (model persistence critical)

**Alternatif**:
- **Pickle**: Standard library Python (lebih lambat, kurang aman)
- **CloudPickle**: Extended pickle untuk cloud (lebih kompleks)
- **HDF5 (h5py)**: Untuk data numerik besar (perlu modifikasi kode)
- **ONNX**: Format universal untuk model (perlu converter)
- **TensorFlow SavedModel / PyTorch**: Untuk model deep learning (tidak cocok untuk scikit-learn)

---

### 9. **Python-multipart** (`python-multipart`)

**Kegunaan**:
- Parsing form data dengan file upload
- Required untuk FastAPI saat menerima file upload

**Digunakan di**:
- `src/serving/api.py`: Endpoint `/predict` yang menerima file upload

**Jika Tidak Ada**:
- ❌ FastAPI tidak bisa menerima file upload
- ❌ Endpoint `/predict` akan error saat menerima gambar
- ⚠️ API tidak bisa digunakan untuk inference

**Alternatif**:
- **Tidak ada alternatif praktis**: FastAPI membutuhkan ini untuk file upload
- **Base64 encoding**: Bisa encode gambar ke string, tapi kurang efisien
- **Direct file path**: Tidak praktis untuk API (security risk)

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐
│  Raw Images     │
│  (data/raw/)    │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  Preprocessing   │
│  - Resize        │
│  - Normalize     │
│  - Flatten       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Training  │
│  - SVM           │
│  - Naive Bayes   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Model Registry  │
│  - Canary        │
│  - Production    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FastAPI Server  │
│  - /predict      │
│  - /health       │
│  - /monitoring   │
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│  Monitoring        │
│  - Drift Detection │
│  - Metrics Log     │
└────────────────────┘
```

## 📁 Struktur Proyek

```
MLOps_6_RC/
├── cli/
│   └── main.py              # CLI interface
├── data/
│   ├── raw/                 # Dataset mentah
│   └── processed/           # Data terproses (kosong, preprocessing di memori)
├── models/
│   ├── canary/              # Model canary (testing)
│   ├── production/          # Model production (serving)
│   └── registry_meta.json   # Metadata registry
├── src/
│   ├── features/
│   │   └── preprocessing.py # Data preprocessing
│   ├── models/
│   │   ├── train.py         # Training logic
│   │   └── registry.py      # Model registry
│   ├── serving/
│   │   ├── api.py           # FastAPI endpoints
│   │   ├── inference.py     # Inference logic
│   │   └── schemas.py       # Pydantic schemas
│   └── monitoring/
│       └── drift.py         # Monitoring & drift detection
├── requirements.txt         # Dependencies
├── README.md                # Dokumentasi ini
└── GETTING_STARTED.md       # Panduan menjalankan proyek
```

## 🚀 Quick Start

Lihat [GETTING_STARTED.md](GETTING_STARTED.md) untuk panduan lengkap menjalankan proyek.

**Ringkasan**:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Training model
python cli/main.py train new

# 3. Promote ke production
python cli/main.py registry promote-canary

# 4. Serve API
python cli/main.py serve start

# 5. Test API
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"
```

---

## 🔁 CI/CD dengan GitHub Actions & Docker

Proyek ini menggunakan **GitHub Actions** dan **Docker** untuk menerapkan CI sederhana (build & check) dan siap dikembangkan menjadi pipeline CI/CD penuh.

### CI (Continuous Integration) dengan GitHub Actions

Workflow CI berada di `.github/workflows/ci.yml` dan akan berjalan pada setiap `push` dan `pull_request`. Secara garis besar, workflow melakukan:

- Checkout kode repository.
- Setup Python 3.11.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- (Slot untuk test) Menjalankan `pytest` ketika test suite sudah tersedia.
- Build Docker image menggunakan `docker/Dockerfile`:
  ```bash
  docker build -t flower-classification -f docker/Dockerfile .
  ```

Ini memastikan bahwa:
- Dependency valid dan dapat di-install.
- Source code selalu bisa dibangun menjadi Docker image.

### CD (Continuous Deployment) dengan Docker (Konseptual)

Untuk deployment, image yang sudah ter-build dapat:

- Didorong ke Docker registry (Docker Hub / GitHub Container Registry).
- Dijalankan di server dengan:
  ```bash
  docker run -d -p 8000:8000 \
    -e CANARY_RATIO=0.1 \
    -v /path/to/models:/app/models \
    flower-classification:latest
  ```

Strategi **canary** tetap dapat dikendalikan melalui environment variable `CANARY_RATIO`, sehingga porsi request yang juga dievaluasi oleh model canary (Naive Bayes) bisa diatur berbeda antara staging dan production.

---

## 🚀 Implementasi MLOps

Bagian ini merangkum sejauh mana konsep-konsep MLOps diintegrasikan dalam proyek, beserta tools yang digunakan.

### 1. Modeling & Perbandingan Model
- **Tujuan**: Membandingkan performa lebih dari satu model pada dataset yang sama.
- **Implementasi**:
  - Menggunakan dua model klasik dari **scikit-learn**:
    - `SVC` (SVM dengan kernel RBF) sebagai kandidat **production model**.
    - `GaussianNB` sebagai **canary/shadow model**.
  - Hasil evaluasi disimpan sebagai:
    - `report_svm.txt`
    - `report_nb.txt`
  - Artefak model disimpan dengan **joblib**:
    - `svm_canary.joblib` dan `nb_canary.joblib` + `scaler` dan `classes`.
- **Tools utama**: `scikit-learn`, `joblib`, `numpy`, `Pillow`.

### 2. Training Pipeline
- **Tujuan**: Menyatukan preprocessing + training dalam pipeline yang konsisten dan mudah dijalankan ulang.
- **Implementasi**:
  - Pipeline training dipicu melalui CLI:
    ```bash
    python cli/main.py train new
    ```
  - `src/features/preprocessing.py`:
    - Resize gambar ke 128×128, normalisasi \([0,1]\), flatten.
    - Train/test split dengan `train_test_split` (stratified, `random_state=42`).
    - Scaling menggunakan `StandardScaler`.
  - `src/models/train.py`:
    - Melatih SVM dan Naive Bayes.
    - Menyimpan scaler, classes, dan kedua model ke `models/canary/`.
- **Tools utama**: `scikit-learn` (SVC, GaussianNB, StandardScaler, train_test_split), `Pillow`, `numpy`, CLI dengan `argparse`.

### 3. Model Registry & Canary Deployment
- **Tujuan**: Memisahkan model yang sedang **diuji** (canary) dan yang **melayani user** (production).
- **Implementasi**:
  - Struktur folder model:
    - `models/canary/` → hasil training terbaru (SVM + NB).
    - `models/production/` → model yang aktif untuk serving.
  - Registry sederhana di `src/models/registry.py`:
    - `registry_meta.json` menyimpan info versi canary/production.
    - `promote_canary_to_production()` menyalin isi `models/canary/` ke `models/production/`.
  - Di API:
    - SVM dari `models/production/` digunakan sebagai **production model** (100% request).
    - NB dari `models/canary/` digunakan sebagai **canary/shadow** (subset request, dikontrol `CANARY_RATIO`).
- **Tools utama**: filesystem (`Path`, `shutil`, `json`), struktur folder `models/`, environment variable `CANARY_RATIO`.

### 4. Model Serving & API
- **Tujuan**: Menyediakan endpoint HTTP untuk inference dan monitoring model.
- **Implementasi**:
  - API dibangun dengan **FastAPI** (`src/serving/api.py`):
    - `GET /health` → health check.
    - `POST /predict` → menerima file gambar (multipart), mengembalikan label dan confidence SVM.
    - `GET /monitoring/drift` → mengembalikan metric monitoring terbaru.
  - Serving dijalankan dengan **Uvicorn**:
    ```bash
    python cli/main.py serve start
    ```
  - Schema response didefinisikan dengan **Pydantic** di `src/serving/schemas.py`.
  - Inference konsisten dengan pipeline training melalui `src/serving/inference.py`.
- **Tools utama**: `FastAPI`, `Uvicorn`, `Pydantic`, `joblib`, `numpy`, `Pillow`.

### 5. Monitoring & Deteksi Drift
- **Tujuan**: Memantau output model di production dan mendeteksi indikasi drift secara sederhana.
- **Implementasi**:
  - `log_prediction()` di `src/monitoring/drift.py`:
    - Dipanggil dari `/predict` untuk setiap prediksi.
    - Menyimpan log JSON lines ke `predictions_log.jsonl`:
      - `model_name`, `y_pred`, `y_true` (opsional), `confidence`.
  - `compute_drift()`:
    - Membaca window terakhir dari `predictions_log.jsonl`.
    - Menghitung:
      - Distribusi kelas prediksi.
      - Rata-rata confidence.
      - (Jika ada label) accuracy sederhana.
      - Perbedaan distribusi dan penurunan confidence terhadap `reference`.
    - Menyimpan hasil ke `metrics_store.json`.
  - Endpoint `GET /monitoring/drift` di FastAPI memanggil `compute_drift()` dan mengembalikan metric.
- **Tools utama**: `numpy`, `scikit-learn.metrics` (accuracy_score), `FastAPI`, file JSON/JSONL.

### 6. CI dengan GitHub Actions
- **Tujuan**: Memastikan kode selalu bisa di-build dan siap dijalankan di environment bersih.
- **Implementasi**:
  - Workflow di `.github/workflows/ci.yml`:
    - Trigger: `push` dan `pull_request`.
    - Setup Python 3.11.
    - Install dependency dari `requirements.txt`.
    - Slot untuk menjalankan `pytest` (bisa diaktifkan ketika test tersedia).
    - Build Docker image menggunakan `docker/Dockerfile`.
- **Tools utama**: **GitHub Actions**, `python`, `pip`, `docker`.

### 7. Containerization dengan Docker
- **Tujuan**: Menyediakan cara deploy yang konsisten di berbagai environment.
- **Implementasi**:
  - Dockerfile di `docker/Dockerfile`:
    - Base image: `python:3.10-slim`.
    - Install dependency sistem minimal.
    - Copy `requirements.txt` dan install Python dependencies.
    - Copy `src` dan `models`.
    - Menjalankan Uvicorn dengan `src.serving.api:app` pada port 8000.
  - Perintah lokal:
    ```bash
    docker build -t flower-classification -f docker/Dockerfile .
    docker run -p 8000:8000 -v ./models:/app/models flower-classification
    ```
- **Tools utama**: **Docker**, `docker/Dockerfile`.

### 8. Ringkasan Implementasi MLOps
- **Data & Preprocessing**: `numpy`, `Pillow`, `scikit-learn.preprocessing`.
- **Modeling & Training**: `scikit-learn` (SVM, GaussianNB), `joblib`, CLI (`argparse`).
- **Model Registry & Canary**: struktur folder `models/`, registry berbasis file (`registry_meta.json`), canary ratio dengan env var.
- **Serving**: `FastAPI`, `Uvicorn`, `Pydantic`, Docker.
- **Monitoring**: logging prediksi & drift sederhana dengan file JSON/JSONL.
- **CI**: GitHub Actions workflow untuk build dan check.
- **Containerization**: Docker image untuk konsistensi environment.

---

## 📚 Referensi

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Dataset Source](https://www.kaggle.com/datasets/kausthubkannan/5-flower-types-classification-dataset)