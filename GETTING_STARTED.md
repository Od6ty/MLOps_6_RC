# Getting Started - Panduan Menjalankan Proyek

Dokumentasi ini menjelaskan langkah-langkah untuk menjalankan proyek MLOps klasifikasi bunga dari awal hingga deployment.

## 📋 Prasyarat

Sebelum memulai, pastikan Anda memiliki:

- **Python 3.10 atau lebih tinggi**
  ```bash
  python --version
  ```

- **Virtual environment** (disarankan)
  - Windows: `python -m venv venv`
  - Linux/Mac: `python3 -m venv venv`

- **Dataset** sudah tersedia di `data/raw/`
  - Struktur: `data/raw/Orchid/`, `data/raw/Tulip/`, `data/raw/Lily/`, `data/raw/Sunflower/`, `data/raw/Lotus/`

## 🚀 Setup Awal

### 1. Aktivasi Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies

Install semua package yang diperlukan dari `requirements.txt`:

```bash
pip install -r requirements.txt
```

Dependencies yang akan terinstall:
- numpy, scipy, pandas
- scikit-learn
- Pillow (image processing)
- fastapi, uvicorn (web API)
- python-multipart (file upload)
- joblib (model persistence)

### 3. Verifikasi Struktur Direktori

Pastikan struktur proyek sudah lengkap:

```
MLOps_6_RC/
├── cli/
│   └── main.py
├── data/
│   ├── raw/
│   │   ├── Orchid/
│   │   ├── Tulip/
│   │   ├── Lily/
│   │   ├── Sunflower/
│   │   └── Lotus/
│   └── processed/
├── models/
│   ├── canary/
│   └── production/
├── src/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── serving/
├── requirements.txt
└── README.md
```

## 🔄 Workflow Utama

Proyek ini mengikuti alur kerja MLOps standar: **Training → Registry → Serving → Monitoring**

### 1. Training Model

Training model baru menggunakan dataset di `data/raw/`. Model akan disimpan di `models/canary/`.

```bash
python cli/main.py train new
```

**Apa yang terjadi:**
- Data preprocessing (resize, normalisasi)
- Split data menjadi train/test
- Training dua model: **SVM** dan **Naive Bayes**
- Model disimpan di `models/canary/`:
  - `scaler.joblib` - StandardScaler untuk preprocessing
  - `classes.joblib` - Daftar kelas label
  - `svm_canary.joblib` - Model SVM
  - `nb_canary.joblib` - Model Naive Bayes
  - `report_svm.txt` - Classification report SVM
  - `report_nb.txt` - Classification report Naive Bayes

**Output:**
- Accuracy untuk setiap model akan ditampilkan di console
- Classification report disimpan sebagai file teks

### 2. Model Registry

Setelah training, promosikan model canary ke production.

#### Cek Status Registry

```bash
python cli/main.py registry status
```

#### Promote Canary ke Production

```bash
python cli/main.py registry promote-canary
```

**Apa yang terjadi:**
- Semua file dari `models/canary/` di-copy ke `models/production/`
- Metadata registry diupdate di `models/registry_meta.json`

**Catatan:** Model production harus ada sebelum menjalankan API server.

### 3. Serving API

Jalankan FastAPI server untuk inference.

#### Start Server (Default Port 8000)

```bash
python cli/main.py serve start
```

#### Start Server dengan Port Custom

```bash
python cli/main.py serve start --port 9000
```

#### Start Server dengan Host Custom

```bash
python cli/main.py serve start --host 127.0.0.1 --port 8000
```

**Server akan berjalan di:**
- Default: `http://0.0.0.0:8000`
- Docs: `http://localhost:8000/docs` (Swagger UI)
- Alternative docs: `http://localhost:8000/redoc`

**Endpoints yang tersedia:**
- `GET /health` - Health check
- `POST /predict` - Prediksi klasifikasi bunga
- `GET /monitoring/drift` - Metrik drift monitoring

### 4. Monitoring

Pantau performa model dan deteksi drift.

#### Tampilkan Metrik Terakhir

```bash
python cli/main.py monitor show
```

#### Hitung Drift Metrics

```bash
python cli/main.py monitor compute-drift
```

**File monitoring:**
- `src/monitoring/metrics_store.json` - Store metrik
- `src/monitoring/predictions_log.jsonl` - Log prediksi

## 🧪 Testing API

### 1. Health Check

Pastikan server berjalan dengan baik:

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok"
}
```

### 2. Prediction Endpoint

#### Menggunakan cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/flower/image.jpg"
```

#### Menggunakan Python requests

```python
import requests

url = "http://localhost:8000/predict"
with open("path/to/flower/image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    print(response.json())
```

**Response:**
```json
{
  "model_used": "svm_prod",
  "label": "Orchid",
  "confidence": 0.85
}
```

#### Menggunakan Swagger UI

1. Buka browser: `http://localhost:8000/docs`
2. Klik endpoint `POST /predict`
3. Klik "Try it out"
4. Upload file gambar
5. Klik "Execute"

### 3. Monitoring Drift Endpoint

```bash
curl http://localhost:8000/monitoring/drift
```

**Response:**
```json
{
  "reference": {
    "class_distribution": {},
    "mean_confidence": null
  },
  "production": {
    "last_window_size": 10,
    "class_distribution": {
      "Orchid": 0.3,
      "Tulip": 0.2,
      "Lily": 0.2,
      "Sunflower": 0.2,
      "Lotus": 0.1
    },
    "mean_confidence": 0.82,
    "accuracy": null
  },
  "drift": {
    "class_dist_diff": null,
    "confidence_drop": null
  }
}
```

## 🐳 Docker (Opsional)

### Build Docker Image

```bash
docker build -t flower-classification -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 flower-classification
```

**Catatan:** Pastikan model sudah ada di `models/production/` sebelum build Docker, atau mount volume saat run:

```bash
docker run -p 8000:8000 -v ./models:/app/models flower-classification
```

## 🔧 Troubleshooting

### Masalah: ModuleNotFoundError

**Gejala:**
```
ModuleNotFoundError: No module named 'src'
```

**Solusi:**
- Pastikan file `__init__.py` ada di semua subdirektori `src/`
- Pastikan virtual environment sudah diaktifkan
- Install ulang dependencies: `pip install -r requirements.txt`

### Masalah: Production model not loaded

**Gejala:**
```
{"detail": "Production model not loaded"}
```

**Solusi:**
1. Pastikan sudah training model: `python cli/main.py train new`
2. Promote ke production: `python cli/main.py registry promote-canary`
3. Verifikasi file ada di `models/production/`:
   - `scaler.joblib`
   - `classes.joblib`
   - `svm_canary.joblib`
   - `nb_canary.joblib`

### Masalah: Port sudah digunakan

**Gejala:**
```
ERROR:    [Errno 48] Address already in use
```

**Solusi:**
- Gunakan port lain: `python cli/main.py serve start --port 9000`
- Atau hentikan proses yang menggunakan port 8000

### Masalah: Dataset tidak ditemukan

**Gejala:**
```
FileNotFoundError: data/raw/...
```

**Solusi:**
- Pastikan dataset sudah di-download dan diletakkan di `data/raw/`
- Struktur harus sesuai: `data/raw/Orchid/`, `data/raw/Tulip/`, dll.
- Setiap folder harus berisi file gambar `.jpg`

### Masalah: Training sangat lambat

**Solusi:**
- Dataset besar (5000 gambar) membutuhkan waktu processing
- Pastikan RAM cukup (minimal 4GB)
- Pertimbangkan mengurangi ukuran dataset untuk testing

## 📝 Catatan Penting

1. **Urutan Eksekusi:**
   - Training harus dilakukan **sebelum** serving
   - Model harus di-promote ke production **sebelum** serving

2. **Canary Deployment:**
   - Model canary (Naive Bayes) akan dievaluasi secara shadow pada 10% request (default)
   - Ratio dapat diubah dengan environment variable: `CANARY_RATIO=0.2`

3. **Model Format:**
   - Model disimpan dalam format `joblib` (scikit-learn)
   - Tidak menggunakan TensorFlow/Keras seperti yang disebutkan di README.md
   - Implementasi menggunakan SVM dan Naive Bayes

4. **Monitoring:**
   - Log prediksi disimpan di `src/monitoring/predictions_log.jsonl`
   - Metrik drift dihitung berdasarkan window terakhir (default: 200 prediksi)

## 🎯 Quick Start Summary

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Training
python cli/main.py train new

# 3. Registry
python cli/main.py registry promote-canary

# 4. Serving
python cli/main.py serve start

# 5. Testing (di terminal lain)
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"

# 6. Monitoring
python cli/main.py monitor show
```

## 📚 Referensi

- Dokumentasi proyek: [README.md](README.md)
- CLI commands: `python cli/main.py --help`
- API documentation: `http://localhost:8000/docs` (setelah server running)

---

**Kelompok 6 MLOps RC**
- Ranippu
- Ratu
- Anwar
- Yohana

