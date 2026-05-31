from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import base64
from .stt import transcribe_speech_to_text
from .llm import generate_response
from .tts import text_to_speech
from .utils import preprocess_audio, normalize_text

app = FastAPI(title="Voice CS System", description="API untuk Sistem Multilingual Speech-to-Speech")

@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    translate: bool = Form(False),
    target_lang: str = Form("Indonesian")
):
    # 1. Simpan file audio/video sementara ke folder temp/
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Path untuk file-file sementara
    input_audio_path = os.path.join(temp_dir, f"input_{audio.filename}")
    preprocessed_audio_path = os.path.join(temp_dir, f"prep_{audio.filename}.wav")
    output_audio_path = os.path.join(temp_dir, f"output_{audio.filename}.wav")

    try:
        # Simpan file yang diupload (bisa dari microphone, upload webm, maupun mkv/mp4)
        with open(input_audio_path, "wb") as f:
            f.write(await audio.read())
            
        # 1.5. Preprocessing video/audio (Ekstrak ke 16kHz WAV mono)
        print("Melakukan preprocessing (ekstraksi & downsampling)...")
        preprocess_audio(input_audio_path, preprocessed_audio_path)
        
        # 2. Panggil transcribe_speech_to_text() -> dapat transkrip
        print("Mengeksekusi STT...")
        transkrip = transcribe_speech_to_text(preprocessed_audio_path)
        print(f"Transkrip Mentah: {transkrip}")
        
        # 2.5 Normalisasi Transkrip
        transkrip_norm = normalize_text(transkrip)
        print(f"Transkrip Hasil Normalisasi: {transkrip_norm}")

        # 3. Panggil generate_response() -> dapat teks respons
        print("Mengeksekusi LLM...")
        respons_teks = generate_response(transkrip_norm, translate=translate, target_language=target_lang)
        print(f"Teks Respons: {respons_teks}")

        # 4. Panggil text_to_speech() -> dapat file audio
        print("Mengeksekusi TTS...")
        text_to_speech(respons_teks, output_audio_path)

        # 5. Baca audio menjadi base64 dan return bersama teks sebagai JSON
        with open(output_audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return JSONResponse(content={
            "transkrip": transkrip_norm,
            "respons_teks": respons_teks,
            "audio_base64": audio_base64
        })
    
    finally:
        # 6. Hapus file input temp setelah selesai (file output biarkan agar bisa dikirim sebagai respon)
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)
        if os.path.exists(preprocessed_audio_path):
            os.remove(preprocessed_audio_path)
