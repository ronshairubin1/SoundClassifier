import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import threading
import time
from .cnn_classifier import N_MFCC, AUDIO_LENGTH, SAMPLE_RATE
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

print(sd.query_devices())

SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 1.0  # seconds for each sample
AUDIO_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)

class SoundDetector:
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.audio_queue = []
        self.is_recording = False
        self.predictions = []
        self.callback = None
        self.stream = None
        self.thread = None
        self.baseline_rms = None  # Baseline noise level
        self.last_prediction_time = None  # For debouncing
        self.buffer = bytearray()
        self.sample_width = 2  # 16-bit audio
        self.frame_duration_ms = 50   # Frame duration in milliseconds
        self.frame_size = int(SAMPLE_RATE * self.frame_duration_ms / 1000)
        self.speech_buffer = bytearray()
        self.speech_detected = False
        self.silence_duration = 0
        self.silence_threshold_ms = 200  # Shorter duration for short words
        self.auto_stop = False

        # Reduce minimum duration for short words
        self.min_speech_duration_ms = 150  # Adjusted for shorter sounds
        self.speech_duration = 0

        # Record background noise
        self.measure_background_noise()
        
        self.audio_queue_lock = threading.Lock()
        
        logging.info(f"SoundDetector initialized with classes: {class_names}")
        
        # Print available audio devices for debugging
        devices = sd.query_devices()
        logging.info("Available audio devices:")
        for i, device in enumerate(devices):
            logging.info(f"[{i}] {device['name']} (inputs: {device['max_input_channels']})")

        # Lower confidence threshold
        self.confidence_threshold = 0.60  # Lower threshold for more predictions

        # Add amplitude threshold
        self.amplitude_threshold = 0.01  # Minimum amplitude to consider sound
        
        # Add minimum prediction threshold
        self.min_prediction_threshold = 0.4  # Minimum probability to make any prediction

        # Add circular buffer for pre-recording
        self.pre_buffer_duration_ms = 50  # 50ms pre-recording
        self.pre_buffer_size = int(SAMPLE_RATE * self.pre_buffer_duration_ms / 1000)
        self.circular_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)
        self.buffer_index = 0

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to capture audio and use amplitude thresholding."""
        try:
            if status:
                logging.warning(f"Audio stream status: {status}")
                return

            # Convert float32 audio data to float64 for processing
            audio_data = indata.astype(np.float64)
            
            # Update circular buffer
            samples_to_write = min(len(audio_data), self.pre_buffer_size)
            start_idx = self.buffer_index
            end_idx = (start_idx + samples_to_write) % self.pre_buffer_size
            
            if end_idx > start_idx:
                self.circular_buffer[start_idx:end_idx] = audio_data[:samples_to_write].flatten()
            else:
                # Handle wrap-around
                first_part = self.pre_buffer_size - start_idx
                self.circular_buffer[start_idx:] = audio_data[:first_part].flatten()
                self.circular_buffer[:end_idx] = audio_data[first_part:samples_to_write].flatten()
            
            self.buffer_index = end_idx

            # Calculate RMS amplitude
            rms = np.sqrt(np.mean(audio_data**2))

            # Use absolute threshold instead of relative to baseline
            threshold = self.amplitude_threshold

            logging.debug(f"Current RMS: {rms:.5f}, Threshold: {threshold:.5f}")

            if rms > threshold:
                # Additional check for speech-like characteristics
                # Calculate zero-crossing rate
                zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_data))))
                zero_crossing_rate = zero_crossings / len(audio_data)
                
                # Only process if it looks like speech
                if 0.01 < zero_crossing_rate < 0.2:  # Typical range for speech
                    logging.debug(f"Speech detected - RMS: {rms:.5f}, ZCR: {zero_crossing_rate:.3f}")
                    # When sound detected, get pre-buffer data
                    pre_audio = np.roll(self.circular_buffer, -self.buffer_index)
                    full_audio = np.concatenate([pre_audio, audio_data.flatten()])
                    self.speech_buffer.extend(full_audio.astype(np.float32).tobytes())
                    self.speech_detected = True
                else:
                    logging.debug(f"Non-speech sound detected - ZCR: {zero_crossing_rate:.3f}")
            else:
                logging.debug("Silence detected")
                if self.speech_detected:
                    self.silence_duration += self.frame_duration_ms
                    if self.silence_duration > self.silence_threshold_ms:
                        # Only process if we have enough speech
                        if self.speech_duration >= self.min_speech_duration_ms:
                            logging.info(f"Processing sound segment of {self.speech_duration}ms")
                            with self.audio_queue_lock:
                                self.audio_queue.append(bytes(self.speech_buffer))
                        else:
                            logging.info(f"Sound segment too short ({self.speech_duration}ms < {self.min_speech_duration_ms}ms)")
                        # Reset speech detection
                        self.speech_buffer.clear()
                        self.speech_detected = False
                        self.silence_duration = 0
                        self.speech_duration = 0
                # Optional: Limit the size of the speech buffer to prevent excessive memory usage
                max_buffer_size = SAMPLE_RATE * 5  # 5 seconds max
                if len(self.speech_buffer) > max_buffer_size:
                    logging.warning("Speech buffer exceeded maximum size, resetting.")
                    self.speech_buffer.clear()
                    self.speech_detected = False
                    self.silence_duration = 0
                    self.speech_duration = 0

            # Add spectral centroid to distinguish between 'eh' and 'oh'
            if self.speech_detected:
                spec_cent = librosa.feature.spectral_centroid(y=audio_data.flatten(), sr=SAMPLE_RATE)
                logging.debug(f"Spectral centroid: {np.mean(spec_cent)}")

        except Exception as e:
            logging.error(f"Error in audio callback: {e}")
            logging.error(f"Input data shape: {indata.shape}")
            logging.error(f"Input data type: {indata.dtype}")
            return

    def measure_background_noise(self, duration=1.0):
        """Measure the baseline noise level over a specified duration."""
        logging.info("Measuring background noise...")
        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, blocking=True)
        sd.wait()
        recording = recording.flatten()
        self.baseline_rms = np.sqrt(np.mean(recording**2))
        logging.info(f"Baseline RMS amplitude: {self.baseline_rms:.5f}")

    def process_audio(self):
        """Process buffered audio segments and make predictions."""
        while self.is_recording:
            if len(self.audio_queue) > 0:
                with self.audio_queue_lock:
                    audio_data = self.audio_queue.pop(0)
                try:
                    # Process the audio data
                    duration_ms = len(audio_data) / SAMPLE_RATE * 1000
                    if duration_ms < self.min_speech_duration_ms:
                        logging.info(f"Sound segment too short ({duration_ms:.0f}ms < {self.min_speech_duration_ms}ms)")
                        continue

                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    logging.info(f"Audio array shape: {audio_array.shape}")
                    
                    # Normalize if needed
                    if np.max(np.abs(audio_array)) > 1.0:
                        audio_array = audio_array / 32768.0  # Normalize 16-bit audio
                    
                    # Check if the sound is loud enough to be speech
                    rms = np.sqrt(np.mean(audio_array**2))
                    if rms < self.amplitude_threshold:
                        logging.info(f"Sound too quiet (RMS: {rms:.5f} < {self.amplitude_threshold})")
                        continue
                    
                    features = preprocess_audio(audio_array)
                    predictions = self.model.predict(features, verbose=0)
                    
                    # Get probabilities for both classes
                    eh_prob = float(predictions[0][self.class_names.index('eh')])
                    oh_prob = float(predictions[0][self.class_names.index('oh')])
                    
                    # Check if either prediction is strong enough
                    if max(eh_prob, oh_prob) < self.min_prediction_threshold:
                        logging.info(f"No clear prediction (eh: {eh_prob:.2%}, oh: {oh_prob:.2%})")
                        continue
                    
                    # Only make prediction if one class is significantly more likely
                    prediction_diff = abs(eh_prob - oh_prob)
                    if prediction_diff < 0.2:  # Require 20% difference between predictions
                        logging.info(f"Predictions too close (diff: {prediction_diff:.2%})")
                        continue
                    
                    predicted_label = 'eh' if eh_prob > oh_prob else 'oh'
                    confidence = max(eh_prob, oh_prob)

                    if confidence > self.confidence_threshold:
                        prediction = {
                            'class': predicted_label,
                            'confidence': confidence
                        }
                        if self.callback:
                            self.callback(prediction)
                    else:
                        logging.info(f"Low confidence prediction ignored: {predicted_label} ({confidence:.1%})")

                except Exception as e:
                    logging.error(f"Error processing audio: {e}")
                    logging.error(f"Audio data type: {type(audio_data)}")
                    logging.error(f"Audio data length: {len(audio_data)}")
                    continue
            else:
                time.sleep(0.1)

    def start_listening(self, callback=None, auto_stop=False):
        """Start listening for sounds."""
        if self.is_recording:
            return False
        
        try:
            self.is_recording = True
            self.callback = callback
            self.auto_stop = auto_stop
            self.predictions = []
            
            # Clear any existing buffers
            self.buffer = bytearray()
            self.speech_buffer = bytearray()
            self.speech_detected = False
            self.silence_duration = 0
            self.speech_duration = 0
            
            logging.info(f"Starting audio stream with:")
            logging.info(f"Sample rate: {SAMPLE_RATE}")
            logging.info(f"Frame duration: {self.frame_duration_ms}ms")
            logging.info(f"Frame size: {self.frame_size} samples")
            logging.info(f"Sample width: {self.sample_width} bytes")
            
            # Configure and start the audio stream
            self.stream = sd.InputStream(
                channels=1,
                dtype=np.float32,
                samplerate=SAMPLE_RATE,
                blocksize=self.frame_size,
                callback=self.audio_callback
            )
            
            # Start the processing thread
            self.thread = threading.Thread(target=self.process_audio)
            self.thread.daemon = True
            self.thread.start()
            
            # Start the audio stream
            self.stream.start()
            return True
            
        except Exception as e:
            logging.error(f"Error starting listener: {e}")
            self.is_recording = False
            return False

    def stop_listening(self):
        """Stop listening for sounds"""
        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Check if we are not in the processing thread
            if self.thread:
                if threading.current_thread() != self.thread:
                    self.thread.join()
                self.thread = None

            logging.info("Stopped listening.")
            self.speech_buffer.clear()
            return self.predictions
        except Exception as e:
            logging.error(f"Error stopping listener: {e}")
            raise

def preprocess_audio(audio):
    """Preprocess audio data for model input."""
    # Pad or truncate the audio to AUDIO_LENGTH
    if len(audio) > AUDIO_LENGTH:
        audio = audio[:AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, AUDIO_LENGTH - len(audio)), mode='constant')

    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    # Remove first MFCC coefficient (energy)
    mfcc = mfcc[1:, :]  # Skip the first coefficient
    delta = librosa.feature.delta(mfcc)  # Delta on reduced MFCCs
    delta2 = librosa.feature.delta(mfcc, order=2)  # Delta2 on reduced MFCCs

    # Additional spectral features
    centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE)
    rms = librosa.feature.rms(y=audio).reshape(1, -1)  # Reshape to match dimensions

    # Combine all features
    features = np.concatenate([
        mfcc,           # (12, time_frames) - now without energy
        delta,          # (12, time_frames)
        delta2,         # (12, time_frames)
        centroid,       # (1, time_frames)
        rolloff,        # (1, time_frames)
        rms             # (1, time_frames)
    ])

    # Ensure consistent time dimension
    target_width = 32
    if features.shape[1] > target_width:
        features = features[:, :target_width]
    else:
        pad_width = target_width - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')

    # Normalize features
    features = (features - np.mean(features)) / (np.std(features) + 1e-5)

    # Add dimensions for model input
    features = features[..., np.newaxis]  # Shape: (40, 32, 1)
    features = np.expand_dims(features, axis=0)  # Shape: (1, 40, 32, 1)

    return features

def record_audio(duration=1.0):
    """Record audio from microphone."""
    print(f"Recording for {duration} second(s)...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, 
                      channels=1, blocking=True)
    sd.wait()  # Wait until recording is done
    return recording.flatten()

def predict_sound(model, input_source, class_names, use_microphone=False):
    """
    Predict sound class from either a file path or microphone input.
    
    Args:
        model: Loaded tensorflow model
        input_source: Either file path (str) or None if using microphone
        class_names: List of class names
        use_microphone: Boolean to indicate if we should record from mic
    
    Returns:
        predicted_label: String name of predicted class
        confidence: Float confidence score
    """
    try:
        if use_microphone:
            audio = record_audio()
        else:
            # Load and preprocess audio file
            audio, _ = librosa.load(input_source, sr=SAMPLE_RATE, mono=True)

        # Preprocess audio
        features = preprocess_audio(audio)

        # Get prediction
        predictions = model.predict(features, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_idx])
        predicted_label = class_names[predicted_idx]

        return predicted_label, confidence

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, 0.0

def run_inference_loop(model, class_names):
    """Interactive loop for continuous predictions."""
    print("\nSound Prediction Mode")
    print("--------------------")
    print("Commands:")
    print("  'mic' - Record from microphone")
    print("  'file <path>' - Predict from audio file")
    print("  'quit' - Exit the program")
    
    while True:
        try:
            command = input("\nEnter command >>> ").strip().lower()
            
            if command == 'quit':
                print("Exiting...")
                break
                
            elif command == 'mic':
                label, conf = predict_sound(model, None, class_names, use_microphone=True)
                if label:
                    print(f"Predicted: '{label}' (confidence: {conf:.3f})")
                
            elif command.startswith('file '):
                file_path = command[5:].strip()
                label, conf = predict_sound(model, file_path, class_names, use_microphone=False)
                if label:
                    print(f"Predicted: '{label}' (confidence: {conf:.3f})")
                    
            else:
                print("Unknown command. Use 'mic', 'file <path>', or 'quit'")
                
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def test_microphone():
    print("Testing microphone...")
    duration = 5  # seconds
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()  # Wait until recording is finished

    # Save recording to a file
    from scipy.io.wavfile import write
    write('test_recording.wav', SAMPLE_RATE, recording)
    print("Recording saved to test_recording.wav")

if __name__ == "__main__":
    # Load model and class names
    try:
        model = tf.keras.models.load_model("models/audio_classifier.h5")
        class_names = np.load("models/class_names.npy", allow_pickle=True)
        print(f"Loaded class names: {class_names}")
        run_inference_loop(model, class_names)
    except Exception as e:
        print(f"Error loading model or class names: {str(e)}")

    test_microphone()
