import os
import time
import csv
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import text_to_speech
from jiwer import wer, cer

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "corpus", "audio")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "corpus", "transcripts")
LOG_DIR = os.path.join(BASE_DIR, "log")
CSV_OUTPUT = os.path.join(LOG_DIR, "hasil_evaluasi.csv")

# Buat folder jika belum ada
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main():
    print(f"Mulai eksperimen Checkpoint 2. Mencari audio di {AUDIO_DIR}")
    
    if not os.path.exists(AUDIO_DIR):
        print("Folder audio belum ada!")
        return

    file_audios = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    if not file_audios:
        print("Tidak ada file .wav yang ditemukan di folder corpus/audio.")
        return
    
    print(f"Ditemukan {len(file_audios)} file audio. Memulai iterasi...")
    
    # Siapkan file CSV untuk export evaluasi
    with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Nama File", "STT Latency (s)", "Transkripsi STT", "LLM Latency (s)", "Respons Gemini", "TTS Latency (s)"])
        
        for index, file_name in enumerate(file_audios, 1):
            print(f"\n--- Memproses file ke-{index}: {file_name} ---")
            audio_path = os.path.join(AUDIO_DIR, file_name)
            
            try:
                # 1. Tahap STT
                stt_start = time.time()
                transkrip = transcribe_speech_to_text(audio_path)
                stt_latency = round(time.time() - stt_start, 2)
                
                # Simpan teks hasil transkripsinya
                txt_filename = file_name.replace(".wav", ".txt")
                with open(os.path.join(TRANSCRIPT_DIR, txt_filename), "w", encoding="utf-8") as f:
                    f.write(transkrip)
                
                print(f"STT Selesai (Latency: {stt_latency}s). Hasil: {transkrip}")
                
                # 2. Tahap LLM
                llm_start = time.time()
                respons_llm = generate_response(transkrip)
                llm_latency = round(time.time() - llm_start, 2)
                print(f"LLM Selesai (Latency: {llm_latency}s). Respons: {respons_llm}")
                
                # 3. Tahap TTS
                output_audio_path = os.path.join(LOG_DIR, f"resp_{file_name}")
                tts_start = time.time()
                text_to_speech(respons_llm, output_audio_path)
                tts_latency = round(time.time() - tts_start, 2)
                print(f"TTS Selesai (Latency: {tts_latency}s). Disimpan ke: {output_audio_path}")
                
                # Tulis hasil tahapan per file ke CSV
                writer.writerow([file_name, stt_latency, transkrip, llm_latency, respons_llm, tts_latency])
                
                # Jeda (sleep) untuk menghindari LLM Rate Limit (Batas RPM Gemini API gratis)
                if index < len(file_audios):
                    print("Jeda 30 detik untuk menghindari Rate Limit Gemini API...")
                    time.sleep(30)
                
            except Exception as e:
                print(f"Gagal memproses {file_name}. Error: {e}")
                
    print(f"\nEksperimen selesai! Ringkasan eval dicetak di {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
