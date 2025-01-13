import os
import logging
import numpy as np
import pandas as pd
from .feature_extractor import AudioFeatureExtractor
from .model import SoundClassifier
import librosa
import soundfile as sf
from datetime import datetime
from config import Config
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
import joblib
import tempfile
from tqdm import tqdm

def preprocess_audio(filepath, temp_dir):
    """Preprocess audio to match inference preprocessing."""
    audio = AudioSegment.from_file(filepath, format="wav")
    # Normalize audio
    audio = audio.normalize()
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # Trim silence
    audio = audio.strip_silence(silence_thresh=-40)
    # Export preprocessed audio to a temporary file
    base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav')
    temp_path = os.path.join(temp_dir, base_name)
    audio.export(temp_path, format="wav")
    return temp_path

class SoundTrainer:
    def __init__(self, good_sounds_dir, model_dir='models'):
        """Initialize the trainer.
        
        Args:
            good_sounds_dir (str): Directory containing verified sound files
            model_dir (str): Directory to save/load model files
        """
        self.good_sounds_dir = good_sounds_dir
        self.feature_extractor = AudioFeatureExtractor()
        self.classifier = SoundClassifier(model_dir=model_dir)  # Create new instance
        self.scaler = StandardScaler()
        
    def augment_audio(self, y, sr):
        """Apply enhanced audio augmentation techniques."""
        augmented = []
        
        # Existing pitch shift augmentations
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))
        
        # Comment out time stretching augmentations
        # augmented.append(self.time_stretch_audio(y, sr, rate=0.9))
        # augmented.append(self.time_stretch_audio(y, sr, rate=1.1))
        
        # Additional augmentations
        
        # Add random noise
        noise_factor = 0.005
        noise = np.random.randn(len(y))
        augmented.append(y + noise_factor * noise)
        
        # Add background noise
        def add_background_noise(y, noise_level=0.02):
            noise = np.random.normal(0, 1, len(y))
            return y + noise_level * np.max(np.abs(y)) * noise

        augmented.append(add_background_noise(y))
        
        # Dynamic range compression
        compressed = librosa.effects.percussive(y)
        augmented.append(compressed)
        
        # Time shifting
        def shift_time(y, shift_max=0.2):
            shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
            return np.roll(y, shift)

        augmented.append(shift_time(y))
        
        # Comment out change speed augmentations
        # def change_speed(y, sr, speed_change=1.0):
        #     return librosa.resample(y, sr, int(sr * speed_change))

        # augmented.append(change_speed(y, sr, speed_change=0.8))
        # augmented.append(change_speed(y, sr, speed_change=1.2))
        
        # Equalization (simulate different recording devices)
        def equalize(y):
            return y * np.random.uniform(0.8, 1.2)

        augmented.append(equalize(y))
        
        # Reverberation (simulate different environments)
        def add_reverb(y, sr):
            return librosa.effects.preemphasis(y)

        augmented.append(add_reverb(y, sr))
        
        return augmented
    
    def _format_features(self, features):
        """Convert feature dictionary to list in consistent order."""
        feature_values = []
        for name in self.feature_extractor.get_feature_names():
            if name.startswith('mfcc_'):
                parts = name.split('_')
                # Extract the feature type (mfcc, mfcc_delta, or mfcc_delta2)
                if 'delta2' in name:
                    base_type = 'mfcc_delta2'
                    idx = int(parts[2])
                elif 'delta' in name:
                    base_type = 'mfcc_delta'
                    idx = int(parts[2])
                else:
                    base_type = 'mfcc'
                    idx = int(parts[1])
                
                # Add the stat type (mean or std)
                stat_type = parts[-1]  # mean or std
                feature_values.append(features[f'{base_type}_{stat_type}'][idx])
            else:
                feature_values.append(features[name])
        return feature_values
    
    def collect_training_data(self):
        """Collect and process training data from verified sounds."""
        import tempfile
        import shutil

        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()

        try:
            features_list = []
            labels = []
            
            # Process each file in the good sounds directory
            for filename in os.listdir(self.good_sounds_dir):
                if not filename.endswith('.wav'):
                    continue
                # Skip temporary files
                if filename.startswith('temp_') or filename.endswith('_preprocessed.wav'):
                    continue

                sound_type = filename.split('_')[0]
                filepath = os.path.join(self.good_sounds_dir, filename)
                # Preprocess the audio to match inference
                preprocessed_path = preprocess_audio(filepath, temp_dir)

                # Load audio for augmentation
                y, sr = librosa.load(preprocessed_path, sr=22050)

                # Get features for original audio
                features = self.feature_extractor.extract_features(preprocessed_path)

                if features is not None:
                    # Use _format_features method for consistency
                    feature_values = self._format_features(features)
                    features_list.append(feature_values)
                    labels.append(sound_type)

                    # Add augmented versions
                    augmented_audio = self.augment_audio(y, sr)
                    for aug_y in augmented_audio:
                        # Save temporarily and extract features
                        temp_path = os.path.join(temp_dir, 'temp_aug.wav')
                        try:
                            sf.write(temp_path, aug_y, sr)
                            aug_features = self.feature_extractor.extract_features(temp_path)
                        except Exception as e:
                            logging.error(f"Error writing or extracting features from {temp_path}: {e}")
                            aug_features = None
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                        if aug_features is not None:
                            aug_values = self._format_features(aug_features)
                            features_list.append(aug_values)
                            labels.append(sound_type)

            if not features_list:
                logging.error("No valid training data found")
                return None, None

            return np.array(features_list), np.array(labels)

        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
    
    def train_model(self, progress_callback=None):
        """Train the model on verified sounds."""
        try:
            # Data collection and preprocessing
            if progress_callback:
                progress_callback(10)  # Data collection started

            X, y = self.collect_training_data()
            if X is None or len(X) == 0:
                logging.error("No training data collected")
                return False

            logging.info(f"Collected training data: X shape={X.shape}, y shape={y.shape}")
            logging.info(f"Unique classes in training data: {np.unique(y)}")

            if progress_callback:
                progress_callback(50)  # Data collection completed

            # Feature scaling
            try:
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                logging.info(f"Data scaled successfully. X_scaled shape={X_scaled.shape}")
            except Exception as e:
                logging.error(f"Error during feature scaling: {str(e)}")
                raise

            # Model training
            if progress_callback:
                progress_callback(60)  # Model training started

            success = self.classifier.train(X_scaled, y)

            if progress_callback:
                progress_callback(90)  # Model training completed

            # Model saving
            if success:
                # Save dictionary info with model
                current_dict = Config.get_dictionary()
                model_info = {
                    'dictionary_name': current_dict['name'],
                    'dictionary_sounds': current_dict['sounds'],
                    'training_time': datetime.now().isoformat(),
                    'feature_names': self.feature_extractor.get_feature_names(),
                    'feature_count': X_scaled.shape[1],
                    'scaler': self.scaler  # Save the scaler
                }
                logging.info(f"Saving model with info: {model_info}")
                self.classifier.save_with_info(model_info)

                if progress_callback:
                    progress_callback(100)  # Training complete

            return success

        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            logging.error("Full traceback:", exc_info=True)
            if progress_callback:
                progress_callback(-1)  # Indicate error
            return False
    
    def evaluate_model(self, test_size=0.2):
        """Evaluate model performance using train-test split.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        X, y = self.collect_training_data()
        if X is None or len(X) == 0:
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train on training set
        self.classifier.train(X_train, y_train)
        
        # Predict on test set
        y_pred, y_proba = self.classifier.predict(X_test)
        
        # Calculate metrics
        unique_sounds = sorted(set(y))
        per_sound_metrics = {}
        
        for sound in unique_sounds:
            mask = y_test == sound
            if any(mask):
                sound_metrics = {
                    'accuracy': accuracy_score(y_test[mask], y_pred[mask]),
                    'samples': sum(mask),
                    'correct': sum((y_test == y_pred) & mask),
                    'confidence': np.mean(np.max(y_proba[mask], axis=1))
                }
                per_sound_metrics[sound] = sound_metrics
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'per_sound': per_sound_metrics,
            'class_names': unique_sounds
        }
        
        return metrics

    # def time_stretch_audio(self, y, sr, rate=1.0):
    #     """Time-stretch an audio signal using resampling."""
    #     # Calculate the target sample rate
    #     target_sr = int(sr / rate)
    #     # Resample the audio signal
    #     y_stretched = librosa.resample(y, sr, target_sr)
    #     return y_stretched
