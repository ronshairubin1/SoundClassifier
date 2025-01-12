import os
import numpy as np
import random
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import current_app
from config import Config  # Use absolute import
import logging
import json
from io import StringIO
from scipy.signal import find_peaks

# -----------------------------
# 1. Global settings
# -----------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 1.0  # seconds for each sample
AUDIO_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)
BATCH_SIZE = 16
EPOCHS = 20

# -----------------------------
# 2. Data augmentation helpers
# -----------------------------
def add_noise(audio, noise_factor=0.005):
    """Add random white noise to an audio signal."""
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented.astype(np.float32)

# -----------------------------
# 3. Audio loading + preprocessing
# -----------------------------
def load_audio(filepath):
    """Load audio file, return signal at fixed SAMPLE_RATE."""
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    # Pad or truncate to fixed length
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))), mode='constant')
    return audio

def extract_mfcc(audio):
    """
    Extract MFCC features from the audio (length = AUDIO_LENGTH).
    Returns a 2D array [N_MFCC x time frames].
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # mfcc shape -> (n_mfcc, time_frames)
    # Optionally, you might do mean normalization across time
    # mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
    return mfcc

def augment_and_extract(filepath):
    """
    Load audio, apply random augmentation, then extract MFCC features.
    """
    audio = load_audio(filepath)

    # Randomly choose one or more augmentations
    if random.random() < 0.3:
        audio = add_noise(audio, noise_factor=0.005)
    
    if random.random() < 0.3:
        try:
            # Try newer librosa version syntax
            stretch_rate = random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, stretch_rate)
        except TypeError:
            try:
                # Try older librosa version syntax
                audio = librosa.effects.time_stretch(rate=stretch_rate, y=audio)
            except:
                # Skip time stretching if both methods fail
                print("Warning: time_stretch failed, skipping this augmentation")
        
        # re-pad if time-stretch shortens or lengthens the clip
        if len(audio) > AUDIO_LENGTH:
            audio = audio[:AUDIO_LENGTH]
        else:
            audio = np.pad(audio, (0, max(0, AUDIO_LENGTH - len(audio))), mode='constant')
    
    if random.random() < 0.3:
        semitones = random.uniform(-2, 2)
        try:
            # Try newer librosa version syntax
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=semitones)
        except TypeError:
            # Try older librosa version syntax
            audio = librosa.effects.pitch_shift(y=audio, sr=SAMPLE_RATE, n_steps=semitones)

    mfcc = extract_mfcc(audio)
    return mfcc

# -----------------------------
# 4. Building the dataset
# -----------------------------
def build_dataset(sound_folder):
    """Build training dataset from sound files."""
    X = []
    y = []
    total_samples = 0
    feature_stats_list = []  # To collect stats from all samples
    
    # Get active dictionary by reading the JSON file directly
    config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
    with open(config_file, 'r') as f:
        active_dict = json.load(f)
    class_names = active_dict['sounds']
    
    logging.info(f"Building dataset for classes: {class_names}")
    
    # Initialize statistics dictionary with all required fields
    stats = {
        'original_counts': {class_name: 0 for class_name in class_names},
        'augmented_counts': {class_name: 0 for class_name in class_names},
        'total_samples': 0,
        'input_shape': None,
        'input_range': None,
        'feature_stats': None,
        'energy_comparison': {
            'eh': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'oh': {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        }
    }
    
    if not class_names:
        logging.error("No sound classes found in active dictionary")
        return None, None, None, None

    def analyze_audio(audio, label):
        """Analyze audio characteristics"""
        # Get fundamental frequency
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=600)
        f0 = f0[voiced_flag]
        if len(f0) > 0:
            mean_f0 = np.mean(f0)
            logging.info(f"  Fundamental frequency ({label}): {mean_f0:.1f} Hz")
        
        # Get formants (rough approximation using spectral peaks)
        spec = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE)
        # Calculate mean spectrum
        mean_spec = np.mean(spec, axis=1)
        # Find peaks using the new peak_pick syntax
        peaks, _ = find_peaks(mean_spec, height=np.mean(mean_spec), distance=10)
        formant_freqs = freqs[peaks][:3]  # First three formants
        logging.info(f"  Formant frequencies ({label}): {formant_freqs}")
    
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(sound_folder, class_name)
        if not os.path.exists(class_path):
            logging.error(f"Directory not found for class: {class_name}")
            continue
        
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        logging.info(f"Found {len(wav_files)} files for class {class_name}")
        stats['original_counts'][class_name] = len(wav_files)
        
        for wav_file in wav_files:
            file_path = os.path.join(class_path, wav_file)
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            analyze_audio(audio, class_name)
            rms = np.sqrt(np.mean(audio**2))
            duration = len(audio) / SAMPLE_RATE
            logging.info(f"File: {wav_file}, Class: {class_name}")
            logging.info(f"  Duration: {duration:.3f}s")
            logging.info(f"  RMS amplitude: {rms:.5f}")
            
            try:
                # Original sample
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                mfcc, feature_stats = process_audio_to_mfcc(audio)
                feature_stats_list.append(feature_stats)  # Save stats
                X.append(mfcc)
                y.append(class_idx)
                total_samples += 1
                
                # Multiple augmentations per sample
                # 1. Noise augmentation
                noisy_audio = add_noise(audio.copy(), noise_factor=0.005)
                mfcc_noise, _ = process_audio_to_mfcc(noisy_audio)
                X.append(mfcc_noise)
                y.append(class_idx)
                
                # 2. Time stretching
                stretch_rate = random.uniform(0.8, 1.2)
                stretched_audio = librosa.effects.time_stretch(audio.copy(), rate=stretch_rate)
                mfcc_stretch, _ = process_audio_to_mfcc(stretched_audio)
                X.append(mfcc_stretch)
                y.append(class_idx)
                
                # 3. Pitch shifting
                pitch_shift = random.uniform(-2, 2)
                shifted_audio = librosa.effects.pitch_shift(audio.copy(), sr=SAMPLE_RATE, n_steps=pitch_shift)
                mfcc_shift, _ = process_audio_to_mfcc(shifted_audio)
                X.append(mfcc_shift)
                y.append(class_idx)
                
                stats['augmented_counts'][class_name] += 3  # Count all augmentations
                
            except Exception as e:
                logging.error(f"Error processing file {wav_file}: {str(e)}")
                continue

    if len(X) == 0:
        logging.error("No valid samples were processed")
        return None, None, None, None

    # Convert to numpy arrays with explicit shape
    X = np.array(X)
    y = np.array(y)
    
    logging.info(f"Final dataset shapes:")
    logging.info(f"X shape: {X.shape}")  # Should be (num_samples, N_MFCC, time_frames, 1)
    logging.info(f"y shape: {y.shape}")  # Should be (num_samples,)
    logging.info(f"Class distribution: {np.bincount(y)}")
    
    # Check class balance
    class_counts = np.bincount(y)
    logging.info("Class distribution:")
    for i, count in enumerate(class_counts):
        logging.info(f"  {class_names[i]}: {count} samples")
    
    if np.std(class_counts) > np.mean(class_counts) * 0.1:  # More than 10% deviation
        logging.warning("Significant class imbalance detected!")
        for i, count in enumerate(class_counts):
            ratio = count / np.max(class_counts)
            logging.warning(f"  {class_names[i]} ratio: {ratio:.2f}")
    
    # Add feature stats to the returned statistics
    if feature_stats_list:
        stats['feature_stats'] = feature_stats_list[0]  # Use stats from first sample as representative
    
    # Add feature analysis logging before returning
    feature_analysis = {}
    if len(X) > 0:
        # Analyze first MFCC coefficient (energy)
        first_mfcc = X[:, 0, :, 0]  # Shape: (samples, time)
        feature_analysis['first_mfcc'] = {
            'mean': float(np.mean(first_mfcc)),
            'std': float(np.std(first_mfcc)),
            'min': float(np.min(first_mfcc)),
            'max': float(np.max(first_mfcc))
        }
        
        # Analyze other MFCC coefficients
        other_mfcc = X[:, 1:13, :, 0]  # Shape: (samples, 12, time)
        feature_analysis['other_mfcc'] = {
            'mean': float(np.mean(other_mfcc)),
            'std': float(np.std(other_mfcc)),
            'min': float(np.min(other_mfcc)),
            'max': float(np.max(other_mfcc))
        }
        
        # Add feature analysis to stats
        stats['feature_analysis'] = feature_analysis
        
        logging.info("Feature Analysis:")
        logging.info(f"First MFCC (Energy) - Mean: {feature_analysis['first_mfcc']['mean']:.4f}, "
                    f"Std: {feature_analysis['first_mfcc']['std']:.4f}")
        logging.info(f"Other MFCCs - Mean: {feature_analysis['other_mfcc']['mean']:.4f}, "
                    f"Std: {feature_analysis['other_mfcc']['std']:.4f}")
    
    # Add analysis comparing eh vs oh energy coefficients
    eh_samples = X[y == 0]  # Assuming 'eh' is index 0
    oh_samples = X[y == 1]  # Assuming 'oh' is index 1

    eh_energy = eh_samples[:, 0, :, 0]  # First MFCC of eh samples
    oh_energy = oh_samples[:, 0, :, 0]  # First MFCC of oh samples

    stats['energy_comparison'] = {
        'eh': {
            'mean': float(np.mean(eh_energy)),
            'std': float(np.std(eh_energy)),
            'min': float(np.min(eh_energy)),
            'max': float(np.max(eh_energy))
        },
        'oh': {
            'mean': float(np.mean(oh_energy)),
            'std': float(np.std(oh_energy)),
            'min': float(np.min(oh_energy)),
            'max': float(np.max(oh_energy))
        }
    }

    logging.info("\nEnergy Coefficient Comparison (First MFCC):")
    logging.info(f"'eh' energy - Mean: {stats['energy_comparison']['eh']['mean']:.4f}, "
                f"Std: {stats['energy_comparison']['eh']['std']:.4f}")
    logging.info(f"'oh' energy - Mean: {stats['energy_comparison']['oh']['mean']:.4f}, "
                f"Std: {stats['energy_comparison']['oh']['std']:.4f}")
    
    # Compare energy coefficient with other features
    for feature_idx in range(1, 13):  # Other MFCC coefficients
        feature_data = X[:, feature_idx, :, 0]
        eh_feature = eh_samples[:, feature_idx, :, 0]
        oh_feature = oh_samples[:, feature_idx, :, 0]
        
        stats[f'mfcc_{feature_idx}_comparison'] = {
            'eh': {
                'mean': float(np.mean(eh_feature)),
                'std': float(np.std(eh_feature)),
                'min': float(np.min(eh_feature)),
                'max': float(np.max(eh_feature))
            },
            'oh': {
                'mean': float(np.mean(oh_feature)),
                'std': float(np.std(oh_feature)),
                'min': float(np.min(oh_feature)),
                'max': float(np.max(oh_feature))
            }
        }
        
        logging.info(f"\nMFCC_{feature_idx} Comparison:")
        logging.info(f"'eh' - Mean: {stats[f'mfcc_{feature_idx}_comparison']['eh']['mean']:.4f}, "
                    f"Std: {stats[f'mfcc_{feature_idx}_comparison']['eh']['std']:.4f}")
        logging.info(f"'oh' - Mean: {stats[f'mfcc_{feature_idx}_comparison']['oh']['mean']:.4f}, "
                    f"Std: {stats[f'mfcc_{feature_idx}_comparison']['oh']['std']:.4f}")
    
    # Add right before return
    logging.info("Stats dictionary contents:")
    logging.info(json.dumps(stats, indent=2, default=str))
    
    return X, y, class_names, stats

def process_audio_to_mfcc(audio):
    """Helper function to process audio to MFCC with consistent shape"""
    # Initialize statistics dictionary
    feature_stats = {}
    
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)), mode='constant')
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Store first MFCC coefficient stats
    feature_stats['first_mfcc'] = {
        'mean': float(np.mean(mfcc[0])),
        'std': float(np.std(mfcc[0])),
        'min': float(np.min(mfcc[0])),
        'max': float(np.max(mfcc[0]))
    }
    
    # Store other MFCC coefficients stats
    feature_stats['other_mfcc'] = {
        'mean': float(np.mean(mfcc[1:])),
        'std': float(np.std(mfcc[1:])),
        'min': float(np.min(mfcc[1:])),
        'max': float(np.max(mfcc[1:]))
    }
    
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Additional spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=audio).reshape(1, -1)
    
    # Store stats for each feature type
    feature_types = {
        'delta': delta,
        'delta2': delta2,
        'centroid': centroid,
        'rolloff': rolloff,
        'rms': rms
    }
    
    for name, feature in feature_types.items():
        feature_stats[name] = {
            'shape': feature.shape,
            'mean': float(np.mean(feature)),
            'std': float(np.std(feature)),
            'min': float(np.min(feature)),
            'max': float(np.max(feature))
        }
    
    # Combine features
    features = np.concatenate([
        mfcc,           # (13, time_frames)
        delta,          # (13, time_frames)
        delta2,         # (13, time_frames)
        centroid,       # (1, time_frames)
        rolloff,        # (1, time_frames)
        rms             # (1, time_frames)
    ])
    
    # Store pre-normalization stats
    feature_stats['pre_normalization'] = {
        'shape': features.shape,
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features))
    }
    
    # Ensure consistent time dimension
    target_width = 32
    if features.shape[1] > target_width:
        features = features[:, :target_width]
    else:
        pad_width = target_width - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    
    # Normalize features
    features = (features - np.mean(features)) / (np.std(features) + 1e-5)
    
    # Store post-normalization stats
    feature_stats['post_normalization'] = {
        'mean': float(np.mean(features)),
        'std': float(np.std(features)),
        'min': float(np.min(features)),
        'max': float(np.max(features))
    }
    
    return features[..., np.newaxis], feature_stats

# -----------------------------
# 5. Define a small CNN model
# -----------------------------
def build_model(input_shape, num_classes):
    # Update input shape to account for all features (42 features total)
    input_shape = (42, input_shape[1], input_shape[2])
    model = models.Sequential()
    
    # First Conv Block
    model.add(layers.Conv2D(16, (3, 3), padding='same', 
                           input_shape=input_shape,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # Increased from 0.001
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(16, (3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))  # Increased from 0.3
    
    # Second Conv Block
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))  # Increased from 0.4

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Capture the model summary in a string
    model_summary_io = StringIO()
    model.summary(print_fn=lambda x: model_summary_io.write(x + '\n'))
    model_summary_str = model_summary_io.getvalue()
    
    return model, model_summary_str

# -----------------------------
# 6. Main training logic
# -----------------------------
if __name__ == "__main__":
    data_path = "data"  # adjust to your folder

    # Build the dataset
    X, y, class_names, stats = build_dataset(data_path, max_files_per_class=20, augment_ratio=2)
    print("Data shape:", X.shape, "Labels shape:", y.shape)
    print("Classes:", class_names)
    
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
    input_shape = (N_MFCC, X.shape[2], 1)  # (13, time_frames, 1)
    model, model_summary = build_model(input_shape, num_classes=len(class_names))
    model.summary()

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
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
