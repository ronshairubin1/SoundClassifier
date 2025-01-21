# -----------------------------
# Global settings
# -----------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 1.0  # Use fixed duration of 1 second
AUDIO_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 50      # Increased epochs 