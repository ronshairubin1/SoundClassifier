import numpy as np
import librosa
from scipy.signal import find_peaks
from .constants import SAMPLE_RATE

class SoundProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.sound_threshold = 0.1  # Threshold for sound detection
        
    def detect_sound(self, audio):
        """Detect if audio contains significant sound and find its location.
        
        Args:
            audio: Input audio signal (numpy array)
            
        Returns:
            tuple: (has_sound, sound_location)
                - has_sound: Boolean indicating if significant sound was detected
                - sound_location: Index of the highest peak if found, or None
        """
        # Calculate RMS energy
        frame_rms = np.sqrt(np.mean(audio**2))
        
        # Find peaks in the audio signal
        peaks, _ = find_peaks(np.abs(audio), height=self.sound_threshold)
        
        # Check if we have significant sound
        has_sound = frame_rms > self.sound_threshold or len(peaks) > 0
        
        # Find the location of the strongest sound
        if len(peaks) > 0:
            # Use the highest peak as the sound location
            sound_location = peaks[np.argmax(np.abs(audio)[peaks])]
        else:
            sound_location = None
            
        return has_sound, sound_location
        
    def center_audio(self, audio):
        """Center audio around the phoneme.
        
        Args:
            audio: Input audio signal (numpy array)
            
        Returns:
            Preprocessed audio centered around phoneme, normalized, and exactly 1 second long
        """
        # Use detect_sound to find the sound location
        has_sound, sound_location = self.detect_sound(audio)
        
        if not has_sound or sound_location is None:
            # If no clear sound, just use the middle
            center = len(audio) // 2
        else:
            # Use the detected sound location as center
            center = sound_location
        
        # Calculate window size (1 second of audio)
        window_size = self.sample_rate
        start_sample = max(0, center - window_size // 2)
        end_sample = min(len(audio), center + window_size // 2)
        
        # Extract phoneme with margins
        audio = audio[start_sample:end_sample]
        
        # Time-stretch to exactly 1 second
        target_length = self.sample_rate
        stretch_factor = target_length / len(audio)
        audio = librosa.effects.time_stretch(y=audio, rate=stretch_factor)
        
        # Ensure exactly 1 second
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)), 'constant')
        
        # Normalize volume to standard level
        target_rms = 0.1  # Target RMS energy level
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
            
        return audio
        
    def extract_features(self, audio):
        """Extract mel-spectrogram features from preprocessed audio.
        
        Args:
            audio: Input audio signal (numpy array), should be preprocessed
            
        Returns:
            features: Processed mel-spectrogram features
        """
        # Compute mel-spectrogram with fixed dimensions
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=64,  # Fixed number of mel bands
            n_fft=1024,
            hop_length=256)  # Reduced hop length for finer time resolution
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Remove the first mel band (energy component)
        mel_spec_db = mel_spec_db[1:, :]
        
        # Ensure fixed time dimension through interpolation if needed
        target_width = 64  # Increased time dimension due to smaller hop_length
        if mel_spec_db.shape[1] != target_width:
            mel_spec_db = np.array([np.interp(
                np.linspace(0, 100, target_width),
                np.linspace(0, 100, mel_spec_db.shape[1]),
                row
            ) for row in mel_spec_db])
        
        # Normalize each frequency band independently
        for i in range(mel_spec_db.shape[0]):
            band = mel_spec_db[i, :]
            mel_spec_db[i, :] = (band - np.mean(band)) / (np.std(band) + 1e-6)
        
        # Add channel dimension for model input (shape should be [height, width, channels])
        features = mel_spec_db[..., np.newaxis]
        
        return features
        
    def process_audio(self, audio):
        """Complete pipeline to process audio into features.
        
        Args:
            audio: Raw input audio signal (numpy array)
            
        Returns:
            features: Processed mel-spectrogram features ready for model input
        """
        # First center and normalize the audio
        preprocessed_audio = self.center_audio(audio)
        
        # Then extract features
        features = self.extract_features(preprocessed_audio)
        
        return features 