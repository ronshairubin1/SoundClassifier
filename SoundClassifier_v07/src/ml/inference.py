import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import threading
import time
from .constants import SAMPLE_RATE, AUDIO_DURATION, AUDIO_LENGTH
from .audio_processing import SoundProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

print(sd.query_devices())

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
        self.buffer = bytearray()
        self.sample_width = 2  # 16-bit audio
        self.frame_duration_ms = 30   # Reduced for finer granularity
        self.frame_size = int(SAMPLE_RATE * self.frame_duration_ms / 1000)
        self.frame_duration = self.frame_duration_ms / 1000.0  # Convert to seconds
        self.speech_buffer = bytearray()
        self.speech_detected = False
        self.silence_duration = 0
        self.silence_threshold_ms = 300  # For better word detection
        self.auto_stop = False
        self.is_speech_active = False
        self.speech_duration = 0
        
        # Initialize sound processor
        self.sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
        
        self.audio_queue_lock = threading.Lock()
        
        logging.info(f"SoundDetector initialized with classes: {class_names}")
        
        # Print available audio devices for debugging
        devices = sd.query_devices()
        logging.info("Available audio devices:")
        for i, device in enumerate(devices):
            logging.info(f"[{i}] {device['name']} (inputs: {device['max_input_channels']})")

        # Thresholds matching training settings
        self.confidence_threshold = 0.40
        self.amplitude_threshold = 0.1  # Match peak height threshold from training
        self.min_prediction_threshold = 0.3

        # Add circular buffer for pre-recording
        self.pre_buffer_duration_ms = 100
        self.pre_buffer_size = int(SAMPLE_RATE * self.pre_buffer_duration_ms / 1000)
        self.circular_buffer = np.zeros(self.pre_buffer_size, dtype=np.float32)
        self.buffer_index = 0

    def process_audio(self):
        try:
            if not self.audio_queue:
                logging.info("No audio data in queue to process")
                return

            logging.info(f"Processing audio queue of size: {len(self.audio_queue)}")
            
            # Concatenate all audio segments
            audio_data = np.concatenate(self.audio_queue)
            logging.info(f"Concatenated audio shape: {audio_data.shape}")

            try:
                # Process audio using shared sound processor
                features = self.sound_processor.process_audio(audio_data)
                if features is None:
                    return
                    
                logging.info(f"Extracted features shape: {features.shape}")

                # Add batch dimension
                features = np.expand_dims(features, axis=0)
                predictions = self.model.predict(features, verbose=0)
                logging.info(f"Raw predictions shape: {predictions.shape}, values: {predictions[0]}")

                # Get the predicted class
                predicted_class = np.argmax(predictions[0])
                confidence = predictions[0][predicted_class]
                
                if confidence > self.confidence_threshold:
                    predicted_label = self.class_names[predicted_class]
                    logging.info(f"Prediction: {predicted_label} with confidence: {confidence:.4f}")
                    self.predictions.append((predicted_label, confidence))
                else:
                    logging.info(f"Prediction confidence {confidence:.4f} below threshold {self.confidence_threshold}")

            except Exception as e:
                logging.error(f"Error in feature extraction/prediction: {str(e)}")
                logging.error(f"Audio data stats - min: {np.min(audio_data)}, max: {np.max(audio_data)}, mean: {np.mean(audio_data)}")
                return

        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")

    def audio_callback(self, indata, frames, time, status):
        """Callback for processing audio frames from the stream."""
        try:
            if status:
                logging.warning(f"Audio callback status: {status}")
                
            # Use shared sound detection
            has_sound, _ = self.sound_processor.detect_sound(indata.flatten())
            
            # Update circular buffer
            start_idx = self.buffer_index
            end_idx = start_idx + len(indata)
            if end_idx > self.pre_buffer_size:
                # Handle wrap-around
                first_part = self.pre_buffer_size - start_idx
                self.circular_buffer[start_idx:] = indata[:first_part].flatten()
                self.circular_buffer[:end_idx - self.pre_buffer_size] = indata[first_part:].flatten()
            else:
                self.circular_buffer[start_idx:end_idx] = indata.flatten()
            self.buffer_index = (self.buffer_index + len(indata)) % self.pre_buffer_size

            # Check if this frame contains significant sound
            if has_sound:
                if not self.is_speech_active:
                    logging.info("Sound detected!")
                    self.is_speech_active = True
                    self.speech_duration = 0
                    
                    # Include pre-buffer data
                    with self.audio_queue_lock:
                        # Reconstruct the buffer in chronological order
                        pre_buffer = np.concatenate([
                            self.circular_buffer[self.buffer_index:],
                            self.circular_buffer[:self.buffer_index]
                        ])
                        self.audio_queue.append(pre_buffer)
                
                # Add current frame to queue
                with self.audio_queue_lock:
                    self.audio_queue.append(indata.flatten())
                self.speech_duration += len(indata) / SAMPLE_RATE
                
                # If we've collected enough audio, process it
                if self.speech_duration >= AUDIO_DURATION:
                    self.process_audio()
                    with self.audio_queue_lock:
                        self.audio_queue.clear()
                    self.is_speech_active = False
                    
            else:
                if self.is_speech_active:
                    # Add a bit more audio after sound stops
                    with self.audio_queue_lock:
                        self.audio_queue.append(indata.flatten())
                    self.process_audio()
                    with self.audio_queue_lock:
                        self.audio_queue.clear()
                    self.is_speech_active = False
                    logging.info("Sound ended")

        except Exception as e:
            logging.error(f"Error in audio callback: {str(e)}")

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
        """Stop listening for audio input."""
        try:
            self.is_recording = False
            
            # Clear any remaining audio data
            with self.audio_queue_lock:
                self.audio_queue.clear()
            
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # Check if we are not in the processing thread
            if self.thread:
                if threading.current_thread() != self.thread:
                    self.thread.join()
                self.thread = None

            logging.info("Stopped listening successfully.")
            self.speech_buffer.clear()
            return {"status": "success", "message": "Stopped listening successfully"}
        except Exception as e:
            error_msg = f"Error stopping listener: {e}"
            logging.error(error_msg)
            return {"status": "error", "message": error_msg}

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
        use_microphone: Whether to use microphone input
    """
    try:
        # Initialize sound processor
        sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)
        
        # Get audio data
        if use_microphone:
            audio = record_audio(AUDIO_DURATION)
        else:
            audio, _ = librosa.load(input_source, sr=SAMPLE_RATE)
        
        # Process audio using shared processor
        features = sound_processor.process_audio(audio)
        
        # Make prediction
        features = np.expand_dims(features, axis=0)
        predictions = model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        predicted_label = class_names[predicted_class]
        
        return predicted_label, confidence
        
    except Exception as e:
        logging.error(f"Error in predict_sound: {str(e)}")
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
