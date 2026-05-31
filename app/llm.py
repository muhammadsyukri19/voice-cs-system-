import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inisialisasi client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Peringatan: GEMINI_API_KEY tidak ditemukan di file .env")

def generate_response(transcript: str, translate: bool = False, target_language: str = "Indonesian") -> str:
    """
    Mengirimkan transkrip STT ke LLM (Gemini) dan mengembalikan respon berupa teks.
    """
    if not transcript:
        return "Mohon maaf, transkripsi kosong."
        
    # Konfigurasi system instruction
    if translate:
        system_instruction = f"Tugasmu adalah menerjemahkan teks berikut ke dalam bahasa {target_language}. Jangan tambahkan kata-kata lain, cukup terjemahannya saja."
    else:
        system_instruction = f"Kamu adalah asisten berbahasa {target_language}. Tolong jawab pertanyaan atau tanggapi kalimat pengguna secara natural menggunakan SATU bahasa {target_language} saja. JANGAN menggunakan bahasa campuran (code-switching). Usahakan responsmu natural, singkat, padat."

    try:
        # Kita menggunakan model gemini-2.5-flash karena lebih cepat dan didukung
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction
        )
        
        response = model.generate_content(transcript)
        return response.text.strip()
        
    except Exception as e:
        print(f"Error saat memanggil LLM: {e}")
        return "Maaf, sistem mengalami kendala saat memproses bahasa."
