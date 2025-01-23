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
        
    def detect_sound_boundaries(self, audio):
        """Detect the start and end of a sound in the audio signal.
        
        Args:
            audio: Input audio signal (numpy array)
            
        Returns:
            start_idx: Start index of the sound
            end_idx: End index of the sound
            has_sound: Whether significant sound was detected
        """
        # Calculate RMS energy in small windows
        frame_length = int(0.02 * self.sample_rate)  # 20ms windows
        hop_length = frame_length // 2
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Interpolate RMS back to audio length
        rms_interp = np.interp(
            np.linspace(0, len(audio), len(audio)),
            np.linspace(0, len(audio), len(rms)),
            rms
        )
        
        # Find regions above threshold
        is_sound = rms_interp > (self.sound_threshold * np.max(rms_interp))
        
        if not np.any(is_sound):
            return 0, len(audio), False
            
        # Find first and last indices above threshold
        sound_indices = np.where(is_sound)[0]
        start_idx = sound_indices[0]
        end_idx = sound_indices[-1]
        
        # Add small margins (100ms) before and after
        margin = int(0.1 * self.sample_rate)
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(audio), end_idx + margin)
        
        return start_idx, end_idx, True

    def center_audio(self, audio):
        """Center audio around the actual sound and scale to exactly one second.
        
        Args:
            audio: Input audio signal (numpy array)
            
        Returns:
            Preprocessed audio containing the sound scaled to exactly one second
        """
        # Find the sound boundaries
        start_idx, end_idx, has_sound = self.detect_sound_boundaries(audio)
        
        if not has_sound:
            # If no clear sound, use the middle portion
            center = len(audio) // 2
            window_size = self.sample_rate  # 1 second
            start_idx = max(0, center - window_size // 2)
            end_idx = min(len(audio), center + window_size // 2)
        
        # Extract the sound segment
        audio = audio[start_idx:end_idx]
        
        # Normalize volume
        target_rms = 0.1  # Target RMS energy level
        current_rms = np.sqrt(np.mean(audio**2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        
        # Time-stretch to exactly 1 second, preserving pitch
        target_length = self.sample_rate
        if len(audio) > 0:
            stretch_factor = target_length / len(audio)
            audio = librosa.effects.time_stretch(y=audio, rate=stretch_factor)
        
        # Ensure exactly 1 second
        if len(audio) > self.sample_rate:
            audio = audio[:self.sample_rate]
        elif len(audio) < self.sample_rate:
            audio = np.pad(audio, (0, self.sample_rate - len(audio)), 'constant')
            
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