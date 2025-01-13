import sounddevice as sd
import numpy as np
from scipy.io import wavfile

duration = 5  # seconds
fs = 44100  # Sample rate

print("Recording for 5 seconds...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording complete. Playing back...")

sd.play(recording, samplerate=fs)
sd.wait()

# Save the recording to a WAV file (optional)
# Convert float32 to int16 for WAV file
recording_int = np.int16(recording * 32767)
wavfile.write("test_recording.wav", fs, recording_int)
print("Playback complete.")
