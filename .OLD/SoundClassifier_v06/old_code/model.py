from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
import numpy as np
import pandas as pd
from config import Config
import librosa
import tensorflow as tf
from .cnn_classifier import (
    N_MFCC, SAMPLE_RATE,
    N_SPECTRAL, N_ROLLOFF, N_RMS,
    N_BASE_FEATURES, TOTAL_FEATURES
)

class SoundClassifier:
    def __init__(self, model_dir='models'):
        """Initialize the classifier.
        
        Args:
            model_dir (str): Directory to save/load model files
        """
        self.model_dir = model_dir
        self.model = None
        self.classes_ = None
        self.feature_names = None
        self.feature_count = None
        self.scaler = None  # Initialize scaler
        os.makedirs(model_dir, exist_ok=True)
        
    def train(self, X, y, n_estimators=100):
        """Train the random forest classifier.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target labels
            n_estimators (int): Number of trees in forest
        """
        try:
            self.model = RandomForestClassifier(
                n_estimators=200,  # More trees
                max_depth=None,    # Allow deep trees
                min_samples_split=4,  # Require more samples per split
                min_samples_leaf=2,   # Require more samples per leaf
                class_weight='balanced',  # Handle unbalanced classes
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
            # Store feature count for validation
            self.feature_count = X.shape[1]
            logging.info(f"Model trained successfully with {len(self.classes_)} classes")
            return True
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, X):
        """Predict class for features.
        
        Args:
            X (array-like): Feature matrix
            
        Returns:
            tuple: (predictions, probabilities)
        """
        if self.model is None:
            logging.error("Model not trained or loaded")
            return None, None
            
        try:
            X = np.array(X)  # Convert from Python list to NumPy array

            # Validate feature count
            if X.shape[1] != self.feature_count:
                logging.error(f"Feature count mismatch. Expected {self.feature_count}, got {X.shape[1]}")
                return None, None
            
            logging.info(f"Making prediction with input shape: {X.shape}")
            logging.info(f"Model classes: {self.classes_}")
            predictions = self.model.predict(X)
            logging.info(f"Raw predictions: {predictions}")
            probabilities = self.model.predict_proba(X)
            logging.info(f"Raw probabilities: {probabilities}")
            return predictions, probabilities
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            logging.exception("Full traceback:")
            return None, None
    
    def get_top_predictions(self, X, top_n=3):
        """Get top N predictions with probabilities."""
        # Ensure X is a 2D array
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = self.model.predict_proba(X)
        top_indices = np.argsort(probabilities, axis=1)[:, -top_n:][:, ::-1]
        top_probabilities = np.take_along_axis(probabilities, top_indices, axis=1)
        top_labels = self.model.classes_[top_indices]
        results = []
        for labels, probs in zip(top_labels, top_probabilities):
            predictions = []
            for label, prob in zip(labels, probs):
                predictions.append({"sound": label, "probability": float(prob)})
            results.append(predictions)
        return results
    
    def save_with_info(self, model_info, filename='sound_classifier.joblib'):
        """Save model to file."""
        if self.model is None:
            logging.error("No model to save")
            return False
            
        try:
            path = os.path.join(self.model_dir, filename)
            model_data = {
                'model': self.model,
                'model_info': model_info,
                'classes': self.classes_,
                'scaler': self.scaler
            }
            joblib.dump(model_data, path)
            logging.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, filename='sound_classifier.joblib'):
        """Load model from file."""
        try:
            path = os.path.join(self.model_dir, filename)
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.classes_ = model_data['classes']
            self.scaler = model_data.get('scaler', None)
            self.feature_names = model_data['model_info']['feature_names']
            self.feature_count = model_data['model_info']['feature_count']
            logging.info(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            logging.error(f"Error loading model from {path}: {str(e)}")
            logging.exception("Full traceback:")
            return False 

    def augment_audio(self, y, sr):
        """Apply enhanced audio augmentation techniques."""
        augmented = []

        # Existing augmentations
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1))

        # Replace time stretching functions with the following
        augmented.append(self.time_stretch_audio(y, rate=0.9))
        augmented.append(self.time_stretch_audio(y, rate=1.1))

        # Additional augmentations

        # Add random noise
        noise_factor = 0.005
        noise = np.random.randn(len(y))
        augmented.append(y + noise_factor * noise)

        # Add background noise
        def add_background_noise(y, noise_level=0.02):
            noise = np.random.normal(0, 1, len(y))
            return y + noise_level * np.abs(y).max() * noise

        augmented.append(add_background_noise(y))

        # Dynamic range compression
        compressed = librosa.effects.percussive(y)
        augmented.append(compressed)

        # Time shifting
        def shift_time(y, shift_max=0.2):
            shift = np.random.randint(int(sr * -shift_max), int(sr * shift_max))
            return np.roll(y, shift)

        augmented.append(shift_time(y))

        # Change speed without changing pitch
        augmented.append(self.time_stretch_audio(y, rate=0.8))
        augmented.append(self.time_stretch_audio(y, rate=1.2))

        # Equalization (simulate different recording devices)
        def equalize(y):
            return y * np.random.uniform(0.8, 1.2)

        augmented.append(equalize(y))

        # Reverberation (simulate different environments)
        def add_reverb(y, sr):
            return librosa.effects.preemphasis(y)

        augmented.append(add_reverb(y, sr))

        return augmented 

    def time_stretch_audio(self, y, rate=1.0):
        """Time-stretch an audio signal."""
        import librosa

        # Compute STFT
        stft = librosa.stft(y)
        # Apply time stretching
        stretched_stft = librosa.phase_vocoder(stft, rate)
        # Convert back to time domain
        y_stretched = librosa.istft(stretched_stft)
        return y_stretched 

    def build_model(self, input_shape, num_classes):
        # Ensure input shape has channel dimension
        if len(input_shape) == 2:
            input_shape = (*input_shape, 1)
        
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            # ... rest of model architecture
        ]) 