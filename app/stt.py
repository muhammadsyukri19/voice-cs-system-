import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Sesuaikan dengan lokasi whisper-cli.exe di Windows Anda
WHISPER_CLI = os.path.join(BASE_DIR, "whisper.cpp", "build", "bin", "whisper-cli.exe")
# Kita akan menggunakan model ggml-small.bin (atau medium) untuk akurasi code-switching yang lebih baik
WHISPER_MODEL = os.path.join(BASE_DIR, "whisper.cpp", "models", "ggml-small.bin")

def transcribe_speech_to_text(audio_path: str) -> str:
    """
    Menjalankan whisper-cli via subprocess untuk transkripsi audio.
    Membutuhkan file audio berekstensi .wav dengan sample rate 16000Hz.
    """
    if not os.path.exists(WHISPER_CLI):
        raise FileNotFoundError(f"Whisper CLI tidak ditemukan di: {WHISPER_CLI}")
    
    if not os.path.exists(WHISPER_MODEL):
        raise FileNotFoundError(f"Model Whisper tidak ditemukan di: {WHISPER_MODEL}")

    # Command: whisper-cli.exe -m <model_path> -f <audio_path> -nt -l id
    # -nt: no-timestamps (agar tidak menampilkan waktu [00:00:00] di output)
    # -l id: secara eksplisit meminta hasil transkripsi awal ke bahasa Indonesia 
    # agar model base tidak memaksakan translate ke Bahasa Inggris akibat Code-Switching.
    command = [
        WHISPER_CLI,
        "-m", WHISPER_MODEL,
        "-f", audio_path,
        "-l", "id",
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
        
        # Jika transkrip kosong, mari kita lihat isi stderr untuk mencari tau error dari whisper.cpp
        if not transkrip:
            print("Peringatan: Transkripsi kosong. Ini log dari Whisper CLI (stderr):")
            print(result.stderr)
            
        return transkrip
        
    except Exception as e:
        print(f"Error saat menjalankan STT: {e}")
        return ""
