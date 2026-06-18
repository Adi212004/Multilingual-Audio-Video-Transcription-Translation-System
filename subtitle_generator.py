import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ==========================
# SETTINGS
# ==========================

AUDIO_FILE = "audio/Hindi_Version_of_Mann_Ki_Baat_May_2026.mp3"   # ← Change to your file name

TARGET_LANGUAGE = "English"

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
# LOAD MODELS
# ==========================

print("Loading Whisper Model...")
whisper_model = whisper.load_model("models/base.pt")

print("Loading NLLB Model...")

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Models Loaded Successfully!")

# ==========================
# TRANSCRIBE AUDIO
# ==========================

print(f"\nTranscribing: {AUDIO_FILE}")
print("This may take several minutes for long audio...\n")

result = whisper_model.transcribe(
    AUDIO_FILE,
    fp16=False,
    verbose=True
)

detected_lang = result["language"]

print("\nDetected Language:", detected_lang)

segments = result["segments"]

print(f"Total segments: {len(segments)}")

# Set source language for NLLB
tokenizer.src_lang = source_lang_map.get(
    detected_lang,
    "eng_Latn"
)

target_lang_code = language_map[TARGET_LANGUAGE]

# ==========================
# TIME FORMATTER
# ==========================

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# ==========================
# SAVE FULL ORIGINAL TRANSCRIPT
# ==========================

with open("output/original_transcript.txt", "w", encoding="utf-8") as file:
    file.write(result["text"].strip())

print("Original transcript saved to output/original_transcript.txt")

# ==========================
# GENERATE TRANSLATED SRT + FULL TRANSLATED TEXT
# (segment by segment - handles long audio correctly)
# ==========================

full_translated_text = []

with open("output/output.srt", "w", encoding="utf-8") as srt_file:

    for i, segment in enumerate(segments, start=1):

        original_text = segment["text"].strip()

        if not original_text:
            continue

        inputs = tokenizer(
            original_text,
            return_tensors="pt",
            truncation=True,
            max_length=400
        )

        translated_tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                target_lang_code
            ),
            max_length=400
        )

        translated_text = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        full_translated_text.append(translated_text)

        srt_file.write(f"{i}\n")
        srt_file.write(
            f"{format_time(segment['start'])} --> "
            f"{format_time(segment['end'])}\n"
        )
        srt_file.write(f"{translated_text}\n\n")

        # Progress indicator
        if i % 10 == 0:
            print(f"Translated {i}/{len(segments)} segments...")

print(f"\nSubtitle file generated: output/output.srt")

# ==========================
# SAVE FULL TRANSLATED TRANSCRIPT
# ==========================

with open("output/translated_transcript.txt", "w", encoding="utf-8") as file:
    file.write(" ".join(full_translated_text))

print(f"Full translated transcript saved: output/translated_transcript.txt")

print(f"\nAll done! Translated to {TARGET_LANGUAGE}.")
print("\nFiles created:")
print("  - output/original_transcript.txt  (full original text)")
print("  - output/translated_transcript.txt (full translated text)")
print("  - output/output.srt  (translated subtitles with timestamps)")