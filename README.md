# 🤟 Project Sign Language Detection using Deep Learning

# Nama : Tresna Hidayah
# Kelas : TI 3B
# NIM : 2023573010118

Sistem penerjemah Bahasa Isyarat Indonesia (BISINDO) berbasis **Deep Learning** yang mampu mengenali **huruf (A-Z)**, **angka (0-9)**, serta **kata statis dan dinamis** secara **real-time** menggunakan webcam.

Project ini menggunakan kombinasi **MediaPipe Hands** sebagai ekstraksi landmark tangan dan beberapa arsitektur Deep Learning:

* **MLP (Multi Layer Perceptron)** untuk klasifikasi huruf dan angka.
* **BiGRU (Bidirectional Gated Recurrent Unit)** untuk klasifikasi kata statis maupun dinamis.

---

# 📖 Fitur

* ✅ Deteksi Huruf A-Z
* ✅ Deteksi Angka 0-9
* ✅ Deteksi Kata BISINDO secara Real-Time
* ✅ Mendukung Gesture Static
* ✅ Mendukung Gesture Dynamic
* ✅ Landmark Visualization
* ✅ Sentence Builder
* ✅ Confidence Score
* ✅ REST API menggunakan Flask
* ✅ Frontend ReactJS

---

# 📂 Dataset

Dataset dikumpulkan sendiri menggunakan webcam dan MediaPipe Hands.

## Total Kelas

### Huruf

* A-Z (26 kelas)

### Angka

* 0-9 (10 kelas)

### Kata Static

* benar
* bertemu
* kamu
* kapan
* makan
* minum
* motor
* sama-sama
* terima kasih
* bis
* mobil

### Kata Dynamic

* Selamat Pagi
* Selamat Siang
* Selamat Sore
* Selamat Malam

Total kelas kata:

15 kelas

---

## Jumlah Sample

Setiap kelas terdiri dari:

* ±300 sample

Data disimpan dalam format:

```
.npy
```

---

# 🏗 Arsitektur Sistem

```
Webcam
   │
   ▼
React Frontend
   │
   ▼
Flask REST API
   │
   ▼
MediaPipe Hands
   │
   ▼
Landmark Extraction
   │
   ├─────────────► MLP
   │                 │
   │                 ├── Huruf
   │                 └── Angka
   │
   ▼
Sequence Buffer (30 Frame)
   │
   ▼
BiGRU
   │
   ▼
Prediksi Kata
```

---

# 🛠 Teknologi

## Frontend

* ReactJS
* TailwindCSS
* Framer Motion
* Axios

## Backend

* Flask
* TensorFlow
* OpenCV
* MediaPipe
* NumPy
* Scikit-Learn

---

# 📁 Struktur Project

```
project/

│
├── backend/
│   ├── app.py
│   ├── landmark_pipeline.py
│   ├── utils/
│   ├── model/
│   ├── model_bigru/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── dataset/
│
├── README.md
│
└── LICENSE
```

---

# 📦 Model

## 1. Huruf

Model:

```
MLP
```

Input:

```
126 Landmark
```

Output:

```
26 kelas
```

---

## 2. Angka

Model:

```
MLP
```

Input:

```
126 Landmark
```

Output

```
10 kelas
```

---

## 3. Kata

Model:

```
Bidirectional GRU
```

Input

```
30 Frame
```

Setiap frame berisi

```
254 fitur
```

yang terdiri dari

* Landmark yang telah dinormalisasi
* Hand Presence Flag
* Delta Motion

Output

```
15 kelas
```

---

# 📈 Hasil Evaluasi Model

## Bidirectional GRU

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **97.35%** |
| Test Loss | 0.0786     |

Model BiGRU memberikan performa terbaik dibandingkan model lainnya.

---

## Perbandingan Model

| Model     | Accuracy   |
| --------- | ---------- |
| GRU       | 92.04%     |
| LSTM      | 97.35%     |
| **BiGRU** | **97.35%** |

Walaupun LSTM menghasilkan akurasi yang sama pada data uji, BiGRU dipilih karena mampu mempelajari informasi temporal dari arah maju dan mundur sehingga lebih baik dalam mengenali gesture dinamis.

---

# 🚀 Instalasi

## Clone Repository

```bash
git clone https://github.com/username/sign-language-detection.git
```

---

## Backend

Masuk ke folder backend

```bash
cd backend
```

Install dependency

```bash
pip install -r requirements.txt
```

Jalankan server

```bash
python app.py
```

Server berjalan di

```
http://127.0.0.1:8000
```

---

## Frontend

Masuk ke folder frontend

```bash
cd frontend
```

Install dependency

```bash
npm install
```

Jalankan aplikasi

```bash
npm run dev
```

---

# 🔄 Cara Kerja

## Huruf dan Angka

```
Camera

↓

MediaPipe

↓

126 Landmark

↓

Scaler

↓

MLP

↓

Prediction
```

---

## Kata

```
Camera

↓

MediaPipe

↓

126 Landmark

↓

30 Frame Buffer

↓

Landmark Pipeline

↓

Scaler

↓

BiGRU

↓

Prediction
```

---

# 🎯 Fitur Website

* Real-Time Detection
* Camera Control
* Confidence Score
* Landmark Visualization
* Kalimat Builder
* Reset Sequence
* Dynamic Word Detection
* Static Word Detection
* Responsive UI

---

# 📷 Tampilan Aplikasi

Halaman utama terdiri dari beberapa menu:

* Home
* About
* Detection
* Gesture Dictionary
* Contact

Menu Detection menampilkan:

* Live Camera
* Landmark Tangan
* Hasil Prediksi
* Confidence
* Kalimat yang Terbentuk

---

# 🔬 Metode Penelitian

1. Pengumpulan Dataset
2. Ekstraksi Landmark menggunakan MediaPipe Hands
3. Preprocessing Landmark
4. Normalisasi Landmark
5. Perhitungan Delta Motion
6. Training Model Deep Learning
7. Evaluasi Model
8. Deploy Model ke Website

---

# 📚 Library

Backend

* TensorFlow
* MediaPipe
* OpenCV
* NumPy
* Flask
* Pandas
* Scikit-Learn
* Joblib

Frontend

* ReactJS
* Axios
* Framer Motion
* TailwindCSS
* Lucide React

---

# 👨‍💻 Author

**Tres Nas**

Mahasiswa Teknologi Informasi dan Komunikasi

Universitas Samudra

---

# 📄 License

Project ini dibuat untuk keperluan penelitian dan pengembangan sistem penerjemah Bahasa Isyarat Indonesia (BISINDO).

Silakan gunakan dan kembangkan dengan tetap mencantumkan atribusi kepada penulis.
