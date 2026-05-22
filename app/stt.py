import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Sesuaikan dengan lokasi whisper-cli.exe di Windows Anda
WHISPER_CLI = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "whisper-cli.exe")
# Kita akan menggunakan model ggml-base.bin karena lebih ringan
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-base.bin")

def transcribe_speech_to_text(audio_path: str) -> str:
    """
    Menjalankan whisper-cli via subprocess untuk transkripsi audio.
    Membutuhkan file audio berekstensi .wav dengan sample rate 16000Hz.
    """
    if not os.path.exists(WHISPER_CLI):
        raise FileNotFoundError(f"Whisper CLI tidak ditemukan di: {WHISPER_CLI}")
    
    if not os.path.exists(WHISPER_MODEL):
        raise FileNotFoundError(f"Model Whisper tidak ditemukan di: {WHISPER_MODEL}")

    # Command: whisper-cli.exe -m <model_path> -f <audio_path> -nt
    # -nt: no-timestamps (agar tidak menampilkan waktu [00:00:00] di output)
    command = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "-nt" 
    ]

    try:
        # Jalankan subprocess
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding='utf-8', 
            errors='ignore'
        )
        
        # Ambil teks transkripsinya (buang spasi kosong di awal/akhir)
        transkrip = result.stdout.strip()
        return transkrip
        
    except Exception as e:
        print(f"Error saat menjalankan STT: {e}")
        return ""
