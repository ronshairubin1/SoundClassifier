import os
import numpy as np
import random
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import current_app
from config import Config
import logging
import json
from io import StringIO
from .audio_processing import SoundProcessor
from .constants import SAMPLE_RATE, N_MFCC, AUDIO_DURATION, AUDIO_LENGTH, BATCH_SIZE, EPOCHS

# -----------------------------
# 1. Global settings
# -----------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 1.0  # Use fixed duration of 1 second
AUDIO_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 50      # Increased epochs

# -----------------------------
# 2. Data augmentation helpers
# -----------------------------
def add_noise(audio, noise_factor=0.005):
    """Add random white noise to an audio signal."""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented

def time_shift(audio, shift_max=0.2):
    """Shift the audio by a random amount."""
    shift = np.random.randint(int(SAMPLE_RATE * -shift_max), int(SAMPLE_RATE * shift_max))
    augmented = np.roll(audio, shift)
    return augmented

def change_pitch(audio, sr=SAMPLE_RATE, pitch_range=2.0):
    """Pitch shift by small random amount."""
    n_steps = np.random.uniform(-pitch_range, pitch_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def change_speed(audio, speed_range=0.1):
    """Time stretch by small random amount."""
    speed_factor = np.random.uniform(1 - speed_range, 1 + speed_range)
    return librosa.effects.time_stretch(y=audio, rate=speed_factor)

def add_colored_noise(audio, noise_type='white', noise_factor=0.005):
    """Add different types of noise to an audio signal.
    
    Args:
        noise_type: 'white', 'pink', or 'brown'
        noise_factor: scaling factor for noise
    """
    if noise_type == 'white':
        noise = np.random.randn(len(audio))
    elif noise_type == 'pink':
        # Generate pink noise using 1/f spectrum
        f = np.fft.fftfreq(len(audio))
        f = np.abs(f)
        f[0] = 1e-6  # Avoid divide by zero
        pink = np.random.randn(len(audio)) / np.sqrt(f)
        noise = np.fft.ifft(pink).real
    elif noise_type == 'brown':
        # Generate brownian noise by cumulative sum of white noise
        noise = np.cumsum(np.random.randn(len(audio)))
        noise = noise / np.max(np.abs(noise))  # Normalize
        
    return audio + noise_factor * noise

# -----------------------------
# 3. Feature extraction
# -----------------------------
# Removed duplicate center_audio function since we use SoundProcessor

# -----------------------------
# 4. Building the dataset
# -----------------------------
def build_dataset(sound_folder):
    """Build training dataset from sound files."""
    X = []
    y = []
    total_samples = 0
    
    # Initialize sound processor
    sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
    
    # Get active dictionary
    config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
    logging.info(f"Looking for config file at: {config_file}")
    with open(config_file, 'r') as f:
        active_dict = json.load(f)
    class_names = active_dict['sounds']
    logging.info(f"Found class names: {class_names}")
    
    # Map class indices
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    logging.info(f"Class indices mapping: {class_indices}")
    
    # Define augmentation parameters
    # More subtle pitch shifts around center
    pitch_shifts_center = np.linspace(-1.0, 1.0, 9)  # 9 subtle shifts
    pitch_shifts_outer = np.array([-3.0, -2.0, 2.0, 3.0])  # 4 larger shifts
    pitch_shifts = np.concatenate([pitch_shifts_outer, pitch_shifts_center])
    
    # Different noise types and levels
    noise_types = ['white', 'pink', 'brown']
    noise_levels = np.linspace(0.001, 0.01, 5)  # 5 levels per type
    
    for class_name in class_names:
        class_path = os.path.join(sound_folder, class_name)
        logging.info(f"Processing class directory: {class_path}")
        if not os.path.exists(class_path):
            logging.warning(f"Directory {class_path} does not exist.")
            continue
        
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        logging.info(f"Processing {len(files)} files for class {class_name}")
        
        for file_name in files:
            file_path = os.path.join(class_path, file_name)
            try:
                logging.info(f"Loading file: {file_path}")
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Process original audio
                features = sound_processor.process_audio(audio)
                X.append(features)
                y.append(class_indices[class_name])
                total_samples += 1
                
                # Extensive augmentation with combinations
                for pitch_shift in pitch_shifts:
                    # Pitch-shifted version
                    audio_pitch = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=pitch_shift)
                    features = sound_processor.process_audio(audio_pitch)
                    X.append(features)
                    y.append(class_indices[class_name])
                    total_samples += 1
                    
                    # Add different noise types
                    for noise_type in noise_types:
                        for noise_factor in noise_levels:
                            # Pitch-shift + noise combination
                            audio_combined = add_colored_noise(audio_pitch, noise_type, noise_factor)
                            features = sound_processor.process_audio(audio_combined)
                            X.append(features)
                            y.append(class_indices[class_name])
                            total_samples += 1
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue
    
    if not X:
        logging.error("No valid samples were processed")
        return None, None, None, None
    
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"Total samples after augmentation: {total_samples}")
    logging.info(f"Dataset shapes: X={X.shape}, y={y.shape}")
    stats = {
        'original_counts': {},
        'augmented_counts': {}
    }
    return X, y, class_names, stats

# -----------------------------
# 5. Define a small CNN model
# -----------------------------
def build_model(input_shape, num_classes):
    """Build a CNN model optimized for phoneme classification."""
    inputs = layers.Input(shape=input_shape)
    
    # First Conv Block
    x = layers.Conv2D(16, (3, 3), padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second Conv Block
    x = layers.Conv2D(32, (3, 3), padding='same',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Reduced learning rate
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Capture the model summary
    model_summary_io = StringIO()
    model.summary(print_fn=lambda x: model_summary_io.write(x + '\n'))
    model_summary = model_summary_io.getvalue()
    
    return model, model_summary

# -----------------------------
# 6. Main training logic
# -----------------------------
if __name__ == "__main__":
    data_path = "data"  # adjust to your data folder
    
    # Build the dataset
    X, y, class_names, _ = build_dataset(data_path)
    
    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Train/validation split
    val_split = 0.2
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Build model
    input_shape = X_train.shape[1:]  # Update input shape based on features
    model, model_summary = build_model(input_shape, num_classes=len(class_names))
    print(model_summary)  # Print model summary
    
    # Set up early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    
    # Save model
    model.save("audio_classifier.h5")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc*100:.2f}%")

class TrainingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_app.training_progress = int((epoch + 1) / EPOCHS * 100)
        current_app.training_status = (
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {logs['loss']:.4f} - "
            f"accuracy: {logs['accuracy']:.4f} - "
            f"val_loss: {logs['val_loss']:.4f} - "
            f"val_accuracy: {logs['val_accuracy']:.4f}"
        )
        print(current_app.training_status)
