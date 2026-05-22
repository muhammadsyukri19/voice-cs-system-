import subprocess
import os

def preprocess_audio(input_path: str, output_path: str) -> str:
    """
    Mengonversi input (baik file video, rekaman audio webm/dari Gradio) 
    menjadi format audio wajib yang dibutuhkan whisper.cpp:
    - Ekstensi: .wav
    - Sample Rate: 16000 Hz
    - Channels: 1 (Mono)
    - Codec: pcm_s16le
    """
    # Mencari ffmpeg.exe di root folder proyek
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ffmpeg_path = os.path.join(base_dir, "ffmpeg.exe")
    
    # Jika tidak ada yang lokal, fallback ke sistem (PATH)
    if not os.path.exists(ffmpeg_path):
        ffmpeg_path = "ffmpeg"

    command = [
        ffmpeg_path,
        "-y",               # Timpa file jika sudah ada
        "-i", input_path,   # File input
        "-ar", "16000",     # Ubah sample rate ke 16 kHz
        "-ac", "1",         # Ubah agar channel menjadi mono
        "-c:a", "pcm_s16le",# Encode ke codec WAV PCM 16-bit
        output_path         # Lokasi output
    ]
    
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error FFmpeg: {e.stderr.decode('utf-8', errors='ignore')}")
        raise Exception("Gagal melakukan downsampling/ekstraksi. Pastikan file valid.")
    except FileNotFoundError:
        raise Exception("FFmpeg tidak ditemukan! Pastikan Anda sudah menginstal FFmpeg dan menambahkannya ke PATH Windows.")
