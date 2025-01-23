import numpy as np
import librosa
from pydub import AudioSegment
from .feature_extractor import AudioFeatureExtractor
import logging

class SoundPreprocessor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.feature_extractor = AudioFeatureExtractor(sr=sample_rate)
        
    def preprocess_sound(self, audio_data, is_file_path=False):
        """
        Preprocess audio data using consistent pipeline for both training and inference.
        
        Args:
            audio_data: Either a file path (str) or numpy array of audio samples
            is_file_path: Boolean indicating if audio_data is a file path
            
        Returns:
            features: Extracted features in the format expected by the model
        """
        try:
            # Load audio if path provided
            if is_file_path:
                audio_data, _ = librosa.load(audio_data, sr=self.sample_rate, mono=True)
            
            # Convert numpy array to AudioSegment
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=self.sample_rate,
                sample_width=2,  # 16-bit audio
                channels=1
            )
            
            # 1. Normalize audio
            audio_segment = audio_segment.normalize()
            
            # 2. Convert to mono if needed
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)
                
            # 3. Trim silence with consistent parameters
            audio_segment = audio_segment.strip_silence(
                silence_thresh=-40,  # Same threshold for both training and inference
                min_silence_len=100  # 100ms minimum silence length
            )
            
            # Convert back to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # 4. Extract features using feature extractor
            features = self.feature_extractor.extract_features(samples)
            if features is None:
                logging.error("Feature extraction failed")
                return None
                
            # 5. Format features consistently
            feature_values = []
            for name in self.feature_extractor.get_feature_names():
                if name.startswith('mfcc_'):
                    parts = name.split('_')
                    if 'delta2' in name:
                        base_type = 'mfcc_delta2'
                        idx = int(parts[2])
                    elif 'delta' in name:
                        base_type = 'mfcc_delta'
                        idx = int(parts[2])
                    else:
                        base_type = 'mfcc'
                        idx = int(parts[1])
                    
                    stat_type = parts[-1]  # mean or std
                    feature_values.append(features[f'{base_type}_{stat_type}'][idx])
                else:
                    feature_values.append(features[name])
                    
            return np.array(feature_values)
            
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            logging.exception("Full traceback:")
            return None
            
    def preprocess_file(self, file_path):
        """Convenience method for preprocessing a file."""
        return self.preprocess_sound(file_path, is_file_path=True)
        
    def preprocess_array(self, audio_array):
        """Convenience method for preprocessing a numpy array."""
        return self.preprocess_sound(audio_array, is_file_path=False) 