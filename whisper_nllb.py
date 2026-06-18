import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# ==========================
# LOAD WHISPER
# ==========================

print("Loading Whisper...")

whisper_model = whisper.load_model("models/base.pt")

print("Whisper Loaded!")

# ==========================
# TRANSCRIBE AUDIO
# ==========================

result = whisper_model.transcribe(
    "audio/claps.mp3",
    fp16=False
)

original_text = result["text"]

print("\nDetected Language:")
print(result["language"])

print("\nOriginal Transcript:")
print(original_text)

# ==========================
# LOAD NLLB
# ==========================

print("\nLoading NLLB...")

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("NLLB Loaded!")

# ==========================
# TRANSLATE TO HINDI
# ==========================

inputs = tokenizer(
    original_text,
    return_tensors="pt"
)

translated_tokens = nllb_model.generate(
    **inputs,
    forced_bos_token_id=
    tokenizer.convert_tokens_to_ids("hin_Deva")
)

translated_text = tokenizer.batch_decode(
    translated_tokens,
    skip_special_tokens=True
)

translated_text = translated_text[0]

print("\nTranslated Text:")
print(translated_text)

# ==========================
# SAVE FILES
# ==========================

with open(
    "output/transcript.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(original_text)

with open(
    "output/translated.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(translated_text)

print("\nFiles Saved Successfully!")