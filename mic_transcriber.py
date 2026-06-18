import sounddevice as sd
import soundfile as sf
import whisper

print("Loading Whisper Model...")
model = whisper.load_model("models/base.pt")
print("Model Loaded!")

duration = 10  # seconds
samplerate = 16000

print("Speak now...")

audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1
)

sd.wait()

sf.write("audio/mic_recording.wav", audio, samplerate)

print("Recording Saved!")

result = model.transcribe(
    "audio/mic_recording.wav",
    fp16=False
)

print("\nTranscript:")
print(result["text"])