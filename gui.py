import tkinter as tk
from tkinter import filedialog, scrolledtext

import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import requests

# ==========================
# OLLAMA SETTINGS
# ==========================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # Change to whatever model you have pulled (e.g. "mistral", "phi3")

# ========================== 
# LOAD MODELS
# ==========================

print("Loading Whisper Model...")

whisper_model = whisper.load_model("models/base.pt")

print("Whisper Loaded!")

print("Loading NLLB Model...")

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)

nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("NLLB Loaded!")

# ==========================
# LANGUAGE MAPS
# ==========================

language_map = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Korean": "kor_Hang",
    "Japanese": "jpn_Jpan",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Telugu": "tel_Telu",
    "Tamil": "tam_Taml",
    "Malayalam": "mal_Mlym",
    "Kannada": "kan_Knda",
    "Punjabi": "pan_Guru",
    "Bengali": "ben_Beng",
    "Gujarati": "guj_Gujr",
    "Marathi": "mar_Deva",
    "Urdu": "urd_Arab",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Arabic": "arb_Arab",
    "Turkish": "tur_Latn",
    "Vietnamese": "vie_Latn",
    "Thai": "tha_Thai",
    "Indonesian": "ind_Latn"
}

source_lang_map = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "ko": "kor_Hang",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "kn": "kan_Knda"
}

# ==========================
# TRANSCRIBE FUNCTION
# ==========================

def transcribe_audio():

    file_path = filedialog.askopenfilename(
        filetypes=[("Media Files", "*.mp3 *.wav *.m4a *.mp4")]
    )

    if not file_path:
        return

    transcript_box.delete(1.0, tk.END)
    translated_box.delete(1.0, tk.END)
    transcript_box.insert(tk.END, "Transcribing...\n")
    root.update()

    result = whisper_model.transcribe(file_path, fp16=False)

    original_text = result["text"]
    detected_lang = result["language"]

    transcript_box.delete(1.0, tk.END)
    transcript_box.insert(tk.END, original_text)

    status_label.config(text=f"Detected Language: {detected_lang}")

    tokenizer.src_lang = source_lang_map.get(detected_lang, "eng_Latn")
    target_lang_code = language_map[selected_language.get()]

    inputs = tokenizer(original_text, return_tensors="pt")

    translated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code)
    )

    translated_text = tokenizer.batch_decode(
        translated_tokens, skip_special_tokens=True
    )[0]

    translated_box.delete(1.0, tk.END)
    translated_box.insert(tk.END, translated_text)

    with open("output/transcript.txt", "w", encoding="utf-8") as file:
        file.write(original_text)

    with open("output/translated.txt", "w", encoding="utf-8") as file:
        file.write(translated_text)

    status_label.config(
        text=f"Detected: {detected_lang} | Translated to: {selected_language.get()}"
    )

# ==========================
# CHATBOT FUNCTION (OLLAMA)
# ==========================

def ask_chatbot():

    user_question = chat_input.get("1.0", tk.END).strip()
    transcript = transcript_box.get("1.0", tk.END).strip()
    translation = translated_box.get("1.0", tk.END).strip()

    if not user_question:
        return

    if not transcript or transcript == "Transcribing...":
        chat_output.insert(tk.END, "⚠️ Please transcribe a file first.\n\n")
        return

    chat_output.insert(tk.END, f"You: {user_question}\n")
    chat_output.insert(tk.END, "Bot: Thinking...\n")
    root.update()

    try:
        prompt = (
            f"Here is an audio transcript:\n{transcript}\n\n"
            f"Translation:\n{translation}\n\n"
            f"Question: {user_question}"
        )

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
        )

        data = response.json()
        reply = data.get("response", "No response received.")

    except Exception as e:
        reply = f"Error: {str(e)}"

    chat_output.delete("end-2l", "end-1l")
    chat_output.insert(tk.END, f"Bot: {reply}\n\n")
    chat_input.delete("1.0", tk.END)


def clear_chat():
    chat_output.delete("1.0", tk.END)

# ==========================
# GUI
# ==========================

root = tk.Tk()
root.title("Whisper Multilingual Translator + Ollama Chatbot")
root.geometry("1000x1100")

heading = tk.Label(root, text="Whisper Multilingual Translator", font=("Arial", 18))
heading.pack(pady=10)

selected_language = tk.StringVar()
selected_language.set("Hindi")

language_label = tk.Label(root, text="Select Output Language:")
language_label.pack()

language_menu = tk.OptionMenu(root, selected_language, *language_map.keys())
language_menu.pack(pady=10)

button = tk.Button(root, text="Select Audio / Video File", command=transcribe_audio, width=30, height=2)
button.pack(pady=10)

transcript_label = tk.Label(root, text="Original Transcript")
transcript_label.pack()

transcript_box = scrolledtext.ScrolledText(root, width=110, height=5)
transcript_box.pack(pady=10)

translated_label = tk.Label(root, text="Translated Text")
translated_label.pack()

translated_box = scrolledtext.ScrolledText(root, width=110, height=5)
translated_box.pack(pady=10)

status_label = tk.Label(root, text="Ready")
status_label.pack(pady=5)

chat_section_label = tk.Label(
    root,
    text="──────────── Ask About the Transcript (Ollama) ────────────",
    font=("Arial", 12)
)
chat_section_label.pack(pady=5)

chat_output = scrolledtext.ScrolledText(root, width=110, height=10)
chat_output.pack(pady=5)

chat_input_label = tk.Label(root, text="Your Question:")
chat_input_label.pack()

chat_input = scrolledtext.ScrolledText(root, width=110, height=3)
chat_input.pack(pady=5)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

ask_button = tk.Button(btn_frame, text="Ask", command=ask_chatbot, width=20, height=2, bg="#4CAF50", fg="white")
ask_button.pack(side=tk.LEFT, padx=10)

clear_button = tk.Button(btn_frame, text="Clear Chat", command=clear_chat, width=20, height=2)
clear_button.pack(side=tk.LEFT, padx=10)

root.mainloop()