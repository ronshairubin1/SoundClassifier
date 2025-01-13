import librosa
import numpy as np
import logging

class AudioFeatureExtractor:
    def __init__(self, sr=22050, duration=None):
        """Initialize feature extractor.
        
        Args:
            sr (int): Sample rate for audio processing
            duration (float): Duration to load from audio file (None for full file)
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = 13  # Back to original MFCC count
        self.hop_length = 512  # For better temporal resolution
        
    def extract_features(self, audio_path):
        """Extract audio features from a file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary of features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Extract features
            features = {}
            
            # MFCCs with delta and delta-delta
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
            features['mfcc_delta_std'] = np.std(mfcc_delta, axis=1)
            features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)
            features['mfcc_delta2_std'] = np.std(mfcc_delta2, axis=1)
            
            # Add formant frequencies
            formants = librosa.effects.preemphasis(y)
            features['formant_mean'] = np.mean(formants)
            features['formant_std'] = np.std(formants)
            
            # Add pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            features['pitch_mean'] = np.mean(pitches[magnitudes > np.median(magnitudes)])
            features['pitch_std'] = np.std(pitches[magnitudes > np.median(magnitudes)])
            
            # Spectral Centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(cent)
            features['spectral_centroid_std'] = np.std(cent)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff_mean'] = np.mean(rolloff)
            features['rolloff_std'] = np.std(rolloff)
            
            return features
            
        except Exception as e:
            logging.error(f"Error extracting features from {audio_path}: {str(e)}")
            return None            
    def get_feature_names(self):
        """Get list of feature names in order."""
        feature_names = []
        
        # MFCCs and their deltas
        for i in range(self.n_mfcc):
            feature_names.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std',
                f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
                f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
            ])
            
        # Other features
        feature_names.extend([
            'formant_mean', 'formant_std',
            'pitch_mean', 'pitch_std',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std',
            'rolloff_mean', 'rolloff_std'
        ])
        
        return feature_names
