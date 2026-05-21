# Voice CS System — UAS Praktikum NLP 2025/2026 Genap

> **Sistem multilingual Speech-to-Speech end-to-end** yang menerima ujaran _code-switching_ Bahasa Indonesia, Inggris, dan Arab, memprosesnya melalui pipeline STT → LLM → TTS, lalu menghasilkan respons suara kembali.

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Persyaratan Sistem](#persyaratan-sistem)
3. [Setup Lingkungan](#setup-lingkungan)
4. [Instalasi Komponen](#instalasi-komponen)
   - [whisper.cpp (STT)](#1-whispercpp-stt)
   - [Gemini API (LLM)](#2-gemini-api-llm)
   - [Coqui TTS](#3-coqui-tts)
5. [Implementasi Kode](#implementasi-kode)
   - [stt.py](#sttpy)
   - [llm.py](#llmpy)
   - [tts.py](#ttspy)
   - [main.py](#mainpy)
6. [Menjalankan Sistem](#menjalankan-sistem)
7. [Korpus Audio (Code-Switching)](#korpus-audio-code-switching)
8. [Pipeline Analisis](#pipeline-analisis)
9. [Evaluasi](#evaluasi)
10. [Pengumpulan (Checkpoint 2)](#pengumpulan-checkpoint-2)
11. [Tips dan Catatan Penting](#tips-dan-catatan-penting)

---

## Struktur Proyek

```
voice-cs-system/
├── app/
│   ├── main.py                  # Entry point FastAPI, endpoint /voice-chat
│   ├── stt.py                   # Speech-to-Text via whisper.cpp
│   ├── llm.py                   # LLM via Gemini API
│   ├── tts.py                   # Text-to-Speech via Coqui TTS
│   ├── utils.py                 # Helper: normalisasi teks, G2P, language tagging
│   ├── whisper.cpp/             # Repo whisper.cpp (clone di sini)
│   │   ├── build/bin/whisper-cli
│   │   └── models/
│   │       └── ggml-large-v3-turbo.bin  (atau model lain)
│   └── coqui_tts/               # Asset model TTS (download manual)
│       ├── checkpoint_1260000-inference.pth
│       ├── config.json
│       └── speakers.pth
├── data/
│   ├── corpus/
│   │   ├── audio/               # File .wav rekaman code-switching
│   │   └── transcripts/         # Hasil transkripsi STT
│   └── manifests/               # Metadata corpus
├── log/                         # Log hasil tiap tahap pipeline (opsional)
├── models/                      # Alternatif lokasi model whisper
├── gradio_app/
│   └── app.py                   # Frontend Gradio (opsional, nilai tambah)
├── analisis_pipeline.py         # Script batch-run seluruh corpus audio
├── .env                         # API Key (JANGAN di-commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Persyaratan Sistem

| Komponen | Versi                                       |
| -------- | ------------------------------------------- |
| Python   | 3.9+ (disarankan 3.11 untuk semua komponen) |
| CMake    | 3.x (untuk build whisper.cpp)               |
| Git      | Terbaru                                     |
| RAM      | Min. 4 GB (8 GB+ untuk model large)         |

---

## Setup Lingkungan

### 1. Buat Virtual Environment

```bash
python3 -m venv env

# Linux / macOS
source env/bin/activate

# Windows
env\Scripts\activate
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
pip install -U google-genai
```

> **PENTING:** Gunakan venv karena versi library proyek ini cukup spesifik dan bisa konflik dengan package global.

### 3. Konfigurasi `.env`

Buat file `.env` di root proyek:

```env
GEMINI_API_KEY=AIzaSyD-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **JANGAN** taruh API key langsung di source code. Tambahkan `.env` ke `.gitignore`.

---

## Instalasi Komponen

### 1. whisper.cpp (STT)

**Clone dan pindahkan ke dalam `app/`:**

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
mv whisper.cpp app/whisper.cpp
cd app/whisper.cpp
```

**Download model:**

```bash
# Rekomendasi (butuh RAM ~3.9 GB):
./models/download-ggml-model.sh large-v3-turbo

# Alternatif jika CPU/RAM terbatas:
./models/download-ggml-model.sh small   # ~852 MB RAM
./models/download-ggml-model.sh base    # ~388 MB RAM
./models/download-ggml-model.sh tiny    # ~273 MB RAM
```

**Build whisper.cpp:**

```bash
# Linux / macOS
cmake -B build
cmake --build build --config Release

# Windows
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

**Test transkripsi:**

```bash
./build/bin/whisper-cli -m models/ggml-large-v3-turbo.bin -f samples/jfk.wav
```

Jika muncul transkrip di terminal → instalasi berhasil.

---

### 2. Gemini API (LLM)

1. Buka [https://aistudio.google.com](https://aistudio.google.com)
2. Klik **Get API key** → **Create API key**
3. Setujui Terms of Service
4. Salin API key → simpan di `.env` (lihat langkah setup di atas)

**Model yang direkomendasikan:**

- `models/gemma-4-26b-a4b-it`
- `models/gemma-4-31b-it`
- Atau model Gemini lain yang tersedia

> **PENTING:** Perhatikan **Rate Limit (RPM)** dan **Request Per Day (RPD)**. Implementasikan `sleep` atau `timeout` ketika RPM tercapai agar pipeline tidak gagal di tengah jalan.

Referensi SDK: [Google Gemini Cookbook Quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get%20started.ipynb)

---

### 3. Coqui TTS

**Download model Indonesian-TTS v1.2:**

Buka [https://github.com/Wikidepia/indonesian-tts](https://github.com/Wikidepia/indonesian-tts), navigasi ke **Releases → v1.2**, download tiga file:

| File                               | Keterangan        |
| ---------------------------------- | ----------------- |
| `checkpoint_1260000-inference.pth` | Model utama       |
| `config.json`                      | Konfigurasi model |
| `speakers.pth`                     | Daftar speaker    |

**Pindahkan ke folder proyek:**

```bash
mkdir -p app/coqui_tts
mv checkpoint_1260000-inference.pth app/coqui_tts/
mv config.json app/coqui_tts/
mv speakers.pth app/coqui_tts/
```

**Test sintesis manual:**

```bash
cd app/coqui_tts/
# Jalankan perintah TTS dari README repositori indonesian-tts
# (gunakan teks format fonemik sesuai contoh di repo)
```

Jika berhasil, akan terbentuk file `output.wav`.

> **Catatan:** Model ini menerima input **fonemik**. Output LLM perlu dikonversi ke fonem terlebih dahulu sebelum masuk ke TTS. Gunakan G2P (Grapheme-to-Phoneme) atau library `phonemizer`.

**Fix versi Transformer (jika ada error):**

```bash
pip install coqui-tts
pip install transformers==5.0  # sesuaikan jika ada konflik
```

---

## Implementasi Kode

### `stt.py`

Isi bagian `TODO` untuk path `whisper-cli` dan model:

```python
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WHISPER_CLI = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "whisper-cli")
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-large-v3-turbo.bin")

def transcribe_speech_to_text(audio_path: str) -> str:
    # Jalankan whisper-cli via subprocess
    # Kembalikan hasil transkripsi sebagai string
    ...
```

**Yang perlu dilengkapi:**

- Path dinamis ke `whisper-cli` binary menggunakan `os.path`
- Path ke file model `.bin`
- Perintah subprocess untuk menjalankan transkripsi
- Parsing output transkrip dari stdout

---

### `llm.py`

Isi konfigurasi Gemini API:

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai  # atau: from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inisialisasi client
# Konfigurasi system instruction
# Fungsi generate_response(transcript: str) -> str
```

**Yang perlu dilengkapi:**

- Inisialisasi client dengan API key dari `.env`
- System prompt yang eksplisit: instruksikan model untuk merespons dengan mempertahankan _code-switching_ **DAN/ATAU** merespons dalam bahasa yang dinormalisasi
- Fungsi `generate_response()` yang menerima transkrip dan mengembalikan teks respons

**Contoh system prompt:**

```
Kamu adalah asisten berbasis suara yang memahami percakapan code-switching
Bahasa Indonesia, Inggris, dan Arab. Pertahankan pola bahasa yang digunakan
pengguna dalam responmu. Jawab secara natural dan singkat.
```

---

### `tts.py`

Isi path model dan fungsi sintesis:

```python
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "coqui_tts", "checkpoint_1260000-inference.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "coqui_tts", "config.json")
SPEAKERS_PATH = os.path.join(BASE_DIR, "coqui_tts", "speakers.pth")

def text_to_speech(text: str, output_path: str) -> str:
    # Konversi teks ke fonem (G2P)
    # Jalankan Coqui TTS dengan parameter model, config, speaker
    # Kembalikan path file audio output
    ...
```

**Yang perlu dilengkapi:**

- Path dinamis ke tiga file model menggunakan `os.path`
- Konversi teks ke format fonemik sebelum sintesis
- Pemanggilan Coqui TTS (CLI atau Python API)
- Penentuan nama speaker yang digunakan

---

### `main.py`

Lengkapi endpoint `/voice-chat`:

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import text_to_speech

app = FastAPI()

@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    # 1. Simpan file audio sementara ke folder temp/
    # 2. Panggil transcribe_speech_to_text() → dapat transkrip
    # 3. Panggil generate_response() → dapat teks respons
    # 4. Panggil text_to_speech() → dapat file audio
    # 5. Return FileResponse(audio_output_path)
    # 6. Hapus file temp setelah selesai
    ...
```

**Yang perlu dilengkapi:**

- Simpan upload ke folder `temp/` menggunakan path dinamis
- Panggil ketiga fungsi dari `stt.py`, `llm.py`, `tts.py` secara berurutan
- Return `FileResponse` dengan file audio hasil TTS
- Cleanup file temp setelah proses selesai

---

## Menjalankan Sistem

### Backend (FastAPI)

```bash
# Dari root proyek, pastikan venv aktif
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend berjalan di: `http://127.0.0.1:8000`

### Frontend (Gradio) — Opsional

```bash
cd gradio_app/
python app.py
```

Frontend berjalan di: `http://127.0.0.1:7860`

> Pastikan **backend sudah berjalan** sebelum membuka frontend.

---

## Korpus Audio (Code-Switching)

### Kewajiban Rekaman

- Minimal **10 file audio** per orang
- **6 ujaran wajib** sama untuk semua peserta
- **4+ ujaran bebas** dari daftar yang disediakan

### Spesifikasi Audio

| Aspek     | Ketentuan                       |
| --------- | ------------------------------- |
| Format    | `.wav` (dianjurkan)             |
| Durasi    | ~10 detik per file              |
| Kualitas  | Ruangan tenang, pelafalan jelas |
| Perangkat | Smartphone direkomendasikan     |

### Konvensi Penamaan File

```
{id}_{utteranceid}.wav
```

**Contoh:** `2030_audio01.wav`

> `id` = dua digit awal + dua digit akhir NPM. Misal NPM `2012345678` → id = `2078`

### Upload Corpus

- Upload ke Google Drive shared storage kelas (A atau B)
- Pastikan file **dapat diakses publik** untuk diunduh
- Struktur folder: pisahkan per kelas (Kelas A / Kelas B)
- **Semua audio** (dari kedua kelas) akan digunakan bersama untuk pipeline

---

## Pipeline Analisis

File `analisis_pipeline.py` digunakan untuk batch-run seluruh corpus:

```python
# Alur analisis_pipeline.py:
# 1. Load semua file audio dari data/corpus/audio/
# 2. Jalankan STT → simpan transkrip ke data/corpus/transcripts/
# 3. Hitung WER/CER jika tersedia ground truth
# 4. Jalankan LLM → simpan respons ke log/
# 5. Jalankan TTS → simpan audio output ke log/
# 6. Catat latency tiap tahap
# 7. Export ringkasan hasil ke CSV atau JSON
```

**Pastikan:**

- Implementasikan `sleep` antar request LLM untuk menghindari RPM limit
- Simpan log error jika ada file yang gagal diproses
- Dokumentasikan persentase file yang berhasil diproses

---

## Evaluasi

| Komponen   | Metrik                                                  |
| ---------- | ------------------------------------------------------- |
| STT        | WER (Word Error Rate), CER (Character Error Rate)       |
| LLM        | Kualitas jawaban (penilaian manual), konsistensi format |
| TTS        | Naturalness (penilaian subjektif), intelligibility      |
| End-to-end | Latency total (ms), persentase keberhasilan pipeline    |

**Cara hitung WER sederhana:**

```python
from jiwer import wer
error_rate = wer(reference_transcript, hypothesis_transcript)
```

---

## Pengumpulan (Checkpoint 2)

### Yang Harus Dikumpulkan

| Item                   | Keterangan                                           |
| ---------------------- | ---------------------------------------------------- |
| **Link GitHub/Repo**   | Kode lengkap, README, requirements.txt               |
| **Final Report (PDF)** | Desain sistem, hasil eksperimen, evaluasi & analisis |
| **Video Presentasi**   | 3–5 menit, final report + demo (jika ada GUI)        |

### Isi Final Report

1. **Desain Sistem** — arsitektur pipeline, diagram alur
2. **Hasil Eksperimen** — output tiap tahap dari corpus audio
3. **Evaluasi & Analisis** — metrik, kendala, hipotesis, justifikasi
4. **Referensi**

### Failure Conditions (Yang Membuat Nilai Turun/Gagal)

- Pipeline tidak lengkap atau tidak dapat dijalankan
- Tidak ada evaluasi, penjelasan, atau justifikasi eksperimen

---

## Tips dan Catatan Penting

**Environment:**

- Gunakan satu versi Python yang sama (disarankan **Python 3.11**) untuk semua komponen
- Selalu aktifkan venv sebelum menjalankan apapun

**Keamanan:**

- Simpan semua credential di `.env`
- Tambahkan `.env` ke `.gitignore`
- Format `.gitignore` minimal:
  ```
  .env
  env/
  __pycache__/
  *.pyc
  app/whisper.cpp/models/*.bin
  app/coqui_tts/*.pth
  temp/
  ```

**Path:**

- Gunakan `os.path.join()` dan `os.path.dirname(os.path.abspath(__file__))` agar proyek portabel di Windows dan Linux
- Jangan pakai path absolut yang hardcoded

**LLM Rate Limit:**

```python
import time
# Setelah setiap N request, tambahkan:
time.sleep(60)  # tunggu 1 menit sebelum lanjut
```

**TTS Multibahasa:**

- Jika respons mengandung beberapa bahasa, pertimbangkan pemrosesan **per segmen** agar pelafalan lebih konsisten
- Gunakan G2P (Grapheme-to-Phoneme) sebelum input ke model TTS

**File Temp:**

```python
import tempfile, os
# Simpan file upload sementara
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(await audio.read())
    tmp_path = tmp.name
# ... proses ...
os.unlink(tmp_path)  # hapus setelah selesai
```

---

## Referensi

- [Template Project GitHub (dari aslab)](https://github.com/) ← _cek link di GCR_
- [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Gemini API Docs](https://ai.google.dev/docs)
- [Google Gemini Cookbook Quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get%20started.ipynb)
- [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS) _(fork aktif dari coqui-tts)_
- [Wikidepia/indonesian-tts](https://github.com/Wikidepia/indonesian-tts)
- [Referensi_dan_Panduan_Tambahan.pdf](./Referensi_dan_Panduan_Tambahan.pdf) _(hanya panduan tambahan, bukan instruksi UAS)_

---

_Disusun untuk UAS Praktikum NLP 2025/2026 Genap — Informatika Universitas Syiah Kuala_
