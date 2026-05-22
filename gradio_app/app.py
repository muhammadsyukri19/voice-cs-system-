import gradio as gr
import requests
import tempfile
import os

# Alamat server backend FastAPI Anda
BACKEND_URL = "http://127.0.0.1:8000/voice-chat"

def process_voice_chat(audio_path):
    if audio_path is None:
        return None, "Error: Tidak ada audio yang direkam/diunggah."

    try:
        # Kirim file audio ke backend FastAPI
        with open(audio_path, "rb") as f:
            files = {"audio": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(BACKEND_URL, files=files)

        # Jika sukses, backend akan mengembalikan file audio wav
        if response.status_code == 200:
            # Buat file sementara untuk menampung audio TTS agar bisa diputar di web
            temp_dir = tempfile.gettempdir()
            output_audio_path = os.path.join(temp_dir, "hasil_respons.wav")
            
            with open(output_audio_path, "wb") as f:
                f.write(response.content)
            
            return output_audio_path, "Proses berhasil!"
        else:
            return None, f"Error Backend: {response.text}"

    except requests.exceptions.ConnectionError:
        return None, "Error: Tidak dapat terhubung ke server backend. Pastikan perintah 'uvicorn app.main:app' sudah dijalankan di terminal lain."
    except Exception as e:
        return None, f"Terjadi kesalahan: {str(e)}"

# Desain UI Web
with gr.Blocks(title="Voice CS System") as interface:
    gr.Markdown("# 🎙️ Sistem Multilingual Speech-to-Speech (Code-Switching)")
    gr.Markdown("Silakan rekam suara Anda, lalu sistem akan merespon dengan suara buatan artificial intelligence. Pastikan backend sudah menyala.")
    
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Suara Pengguna")
            submit_btn = gr.Button("Kirim Suara", variant="primary")
            status_text = gr.Textbox(label="Status Sistem", interactive=False)
            
        with gr.Column():
            output_audio = gr.Audio(label="Respons Sistem (Gemini + Coqui TTS)", type="filepath", interactive=False)
            
    submit_btn.click(
        fn=process_voice_chat,
        inputs=[input_audio],
        outputs=[output_audio, status_text]
    )

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", server_port=7860)
