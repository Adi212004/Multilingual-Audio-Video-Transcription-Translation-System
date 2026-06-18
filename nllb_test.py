from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Bonjour tout le monde."

inputs = tokenizer(
    text,
    return_tensors="pt"
)

translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn")
)

translated_text = tokenizer.batch_decode(
    translated_tokens,
    skip_special_tokens=True
)

print(translated_text[0])
with open("output/translated.txt", "w", encoding="utf-8") as file:
    file.write(translated_text[0])
