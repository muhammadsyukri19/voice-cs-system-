import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import requests
import tempfile
import base64

# Alamat server backend FastAPI Anda
# Karena port 8000 bentrok dengan Django, kita asumsikan FastAPI dijalankan di port 8080
BACKEND_URL = "http://127.0.0.1:8080/voice-chat"

def process_voice_chat(audio_path, is_translate, target_lang):
    if audio_path is None:
        return None, "", "", "Error: Tidak ada audio yang direkam/diunggah."

    try:
        # Kirim file audio ke backend FastAPI
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"translate": is_translate, "target_lang": target_lang}
            response = requests.post(BACKEND_URL, files=files, data=data)

        # Jika sukses, backend akan mengembalikan JSON
        if response.status_code == 200:
            result_data = response.json()
            transkrip = result_data.get("transkrip", "")
            respons_teks = result_data.get("respons_teks", "")
            audio_base64 = result_data.get("audio_base64", "")

            # Buat file sementara untuk menampung audio TTS agar bisa diputar di web
            temp_dir = tempfile.gettempdir()
            output_audio_path = os.path.join(temp_dir, "hasil_respons.wav")
            
            with open(output_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            
            return output_audio_path, transkrip, respons_teks, "Proses berhasil!"
        else:
            return None, "", "", f"Error Backend: {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "", "", "Error: Tidak dapat terhubung ke server backend. Pastikan perintah 'uvicorn app.main:app' sudah dijalankan di terminal lain."
    except Exception as e:
        return None, "", "", f"Terjadi kesalahan: {str(e)}"

# Desain UI Web
with gr.Blocks(title="Voice CS System") as interface:
    gr.Markdown("# 🎙️ Sistem Multilingual Speech-to-Speech (Code-Switching)")
    gr.Markdown("Silakan rekam suara Anda, lalu sistem akan merespon dengan suara buatan artificial intelligence. Pastikan backend sudah menyala.")
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Suara Pengguna")
            is_translate = gr.Checkbox(label="Aktifkan Terjemahan", value=False)
            target_lang = gr.Dropdown(choices=["Indonesian", "English", "Arabic"], value="Indonesian", label="Pilih Bahasa Utama")
            submit_btn = gr.Button("Kirim Suara", variant="primary")
            status_text = gr.Textbox(label="Status Sistem", interactive=False)
            
        with gr.Column():
            transkrip_text = gr.Textbox(label="Transkripsi Input (Speech-to-Text)", interactive=False)
            respons_text = gr.Textbox(label="Respons Teks", interactive=False)
            output_audio = gr.Audio(label="Respons Sistem (Gemini + Coqui TTS)", type="filepath", interactive=False)
            
    submit_btn.click(
        fn=process_voice_chat,
        inputs=[input_audio, is_translate, target_lang],
        outputs=[output_audio, transkrip_text, respons_text, status_text]
    )

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)
