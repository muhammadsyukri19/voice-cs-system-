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

# Konfigurasi system instruction
system_instruction = """
Kamu adalah asisten berbasis suara yang memahami percakapan code-switching
Bahasa Indonesia, Inggris, dan Arab. Kembalikan respon dengan code-switching 
sealami mungkin. Usahakan responsmu natural, singkat, padat, dan langsung 
menjawab kebutuhan pengguna layaknya percakapan verbal.
"""

def generate_response(transcript: str) -> str:
    """
    Mengirimkan transkrip STT ke LLM (Gemini) dan mengembalikan respon berupa teks.
    """
    if not transcript:
        return "Mohon maaf, transkripsi kosong."

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
