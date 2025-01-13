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
from scipy.signal import find_peaks

# -----------------------------
# 1. Global settings
# -----------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
TARGET_DURATION = 0.5  # Target duration in seconds
AUDIO_LENGTH = int(SAMPLE_RATE * TARGET_DURATION)
BATCH_SIZE = 32
EPOCHS = 50

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
    return librosa.effects.time_stretch(audio, speed_factor)

# -----------------------------
# 3. Audio processing functions
# -----------------------------
def center_audio(audio):
    """Center the audio around the phoneme by trimming silence."""
    # Energy thresholding to find active speech
    energy = librosa.feature.rms(y=audio)[0]
    frames = np.nonzero(energy > np.max(energy) * 0.1)[0]
    
    if frames.size:
        # Compute start and end positions
        start_frame = frames[0]
        end_frame = frames[-1]
        start_sample = max(0, start_frame * 512)
        end_sample = min(len(audio), end_frame * 512)
        audio = audio[start_sample:end_sample]
    else:
        # If no frames detected, return the original audio
        pass
    
    return audio

def stretch_audio(audio, target_duration, sr=SAMPLE_RATE):
    """Stretch or compress audio to a fixed duration."""
    # Calculate the current duration
    current_duration = librosa.get_duration(y=audio, sr=sr)
    # Calculate the stretch factor
    if current_duration == 0:
        stretch_factor = 1.0  # Avoid division by zero
    else:
        stretch_factor = target_duration / current_duration
    # Apply time stretching
    audio_stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
    return audio_stretched

def process_audio_to_features(audio):
    """Process audio into features with consistent dimensions."""
    feature_stats = {}
    
    # Include 30 ms of additional context before and after the audio
    context_duration = 0.03  # 30 ms in seconds
    context_samples = int(context_duration * SAMPLE_RATE)
    
    # Pad the audio with context (silence)
    audio = np.pad(audio, (context_samples, context_samples), 'constant')
    
    # Center the audio around the phoneme (after adding context)
    audio = center_audio(audio)
    
    # Stretch or compress audio to fixed duration
    audio = stretch_audio(audio, target_duration=TARGET_DURATION, sr=SAMPLE_RATE)
    
    # Normalize audio amplitude
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    # Ensure the audio length is exactly AUDIO_LENGTH samples
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    elif len(audio) < AUDIO_LENGTH:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)), 'constant')
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=128,
        n_fft=1024,
        hop_length=256)
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Compute MFCCs
    mfcc = librosa.feature.mfcc(
        S=mel_spec_db,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC)
    
    # Stack features
    features = np.vstack([mel_spec_db, mfcc])
    
    # Ensure consistent time frames by padding or truncating
    target_frames = 32  # Define a fixed number of time frames
    if features.shape[1] > target_frames:
        features = features[:, :target_frames]
    elif features.shape[1] < target_frames:
        pad_width = target_frames - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    
    # Standardize features
    features = (features - np.mean(features)) / (np.std(features) + 1e-6)
    
    # Add channel dimension for CNN
    features = features[..., np.newaxis]
    
    return features, feature_stats

# -----------------------------
# 4. Building the dataset
# -----------------------------
def build_dataset(sound_folder):
    """Build training dataset from sound files."""
    X = []
    y = []
    total_samples = 0
    
    # Get active dictionary
    config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
    with open(config_file, 'r') as f:
        active_dict = json.load(f)
    class_names = active_dict['sounds']
    
    # Map class indices
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    # Determine the expected feature shape
    test_audio = np.zeros(AUDIO_LENGTH)
    test_features, _ = process_audio_to_features(test_audio)
    expected_shape = test_features.shape

    for class_name in class_names:
        class_path = os.path.join(sound_folder, class_name)
        if not os.path.exists(class_path):
            logging.warning(f"Directory {class_path} does not exist.")
            continue
        
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        logging.info(f"Processing {len(files)} files for class {class_name}")
        
        for file_name in files:
            file_path = os.path.join(class_path, file_name)
            try:
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Original audio
                features, _ = process_audio_to_features(audio)

                if features.shape != expected_shape:
                    logging.warning(f"Skipping {file_path} due to shape mismatch. Expected {expected_shape}, got {features.shape}")
                    continue

                X.append(features)
                y.append(class_indices[class_name])
                total_samples += 1
                
                # Data augmentations
                augmentations = []

                # 1. Add noise
                if random.random() < 0.5:
                    audio_aug = add_noise(audio)
                    augmentations.append(audio_aug)

                # 2. Time shift
                if random.random() < 0.5:
                    audio_aug = time_shift(audio)
                    augmentations.append(audio_aug)

                # 3. Change pitch
                if random.random() < 0.3:
                    audio_aug = change_pitch(audio)
                    augmentations.append(audio_aug)

                # 4. Change speed
                if random.random() < 0.3:
                    audio_aug = change_speed(audio)
                    augmentations.append(audio_aug)

                for aug_audio in augmentations:
                    features_aug, _ = process_audio_to_features(aug_audio)

                    if features_aug.shape != expected_shape:
                        logging.warning(f"Skipping augmented audio due to shape mismatch. Expected {expected_shape}, got {features_aug.shape}")
                        continue

                    X.append(features_aug)
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
    logging.info(f"Final dataset shape: X={X.shape}, y={y.shape}")
    return X, y, class_names, None  # Keeping the return structure the same

# -----------------------------
# 5. Define a small CNN model
# -----------------------------
def build_model(input_shape, num_classes):
    """Build a CNN model optimized for phoneme classification."""
    model = models.Sequential([
        # First Conv Block
        layers.Conv2D(16, (3, 3), padding='same', input_shape=input_shape,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Second Conv Block
        layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third Conv Block
        layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Capture the model summary in a string (existing functionality)
    model_summary_io = StringIO()
    model.summary(print_fn=lambda x: model_summary_io.write(x + '\n'))
    model_summary_str = model_summary_io.getvalue()
    
    return model, model_summary_str

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
