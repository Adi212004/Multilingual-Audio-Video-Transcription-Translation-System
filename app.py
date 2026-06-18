import whisper

print("Loading Whisper Model...")

model = whisper.load_model("models/base.pt")

print("Model Loaded Successfully!")

result = model.transcribe("audio/english.mp4")

print("\nTranscript:")
print(repr(result["text"]))

with open("output/output.txt", "w", encoding="utf-8") as file:
    file.write(result["text"])

print("\nTranscript saved successfully!")
