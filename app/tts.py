import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def text_to_speech(text: str, output_path: str) -> str:
    """
    Mengubah teks menjadi suara menggunakan Edge-TTS melalui subprocess,
    sebagai ganti Coqui TTS yang tidak kompatibel dengan Python 3.13.
    """
    # Gunakan voice multi-bahasa dari Microsoft Edge (mendukung Indonesia & English)
    command = [
        sys.executable, "-m", "edge_tts",
        "--voice", "id-ID-ArdiNeural",
        "--text", text,
        "--write-media", output_path
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error saat menjalankan Edge-TTS: {e.stderr.decode('utf-8', errors='ignore')}")
        return ""
    except Exception as e:
        print(f"Error tidak terduga pada TTS: {e}")
        return ""
