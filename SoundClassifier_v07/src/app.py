import os
import json
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify, Response, stream_with_context
import logging
import numpy as np
from ml.cnn_classifier import build_model, build_dataset, N_MFCC, TrainingCallback
from ml.inference import predict_sound, SoundDetector
from tensorflow.keras import models
from config import Config
from datetime import datetime
import threading
import time
from flask_cors import CORS
from pydub import AudioSegment
from sound_processor import SoundProcessor
import io
import tensorflow as tf

# Get absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'templates')
STATIC_DIR = os.path.join(CURRENT_DIR, 'static')

app = Flask(__name__, 
    static_url_path='/static',
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR)
app.secret_key = 'your-secret-key'  # For session management
CORS(app, supports_credentials=True)

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add test route
@app.route('/test')
def test():
    app.logger.debug("Test route accessed")
    return "Server is working!"

class Config:
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    STATIC_DIR = STATIC_DIR
    TEMP_DIR = os.path.join(STATIC_DIR, 'temp')
    GOOD_SOUNDS_DIR = os.path.join(STATIC_DIR, 'goodsounds')
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        # Create directories with proper permissions
        for directory in [cls.STATIC_DIR, cls.TEMP_DIR, cls.GOOD_SOUNDS_DIR, cls.CONFIG_DIR]:
            os.makedirs(directory, mode=0o755, exist_ok=True)  # Only creates if doesn't exist
            app.logger.debug(f"Created directory: {directory}")
        
        # Create sound subdirectories in goodsounds (only if they don't exist)
        dictionary = cls.get_dictionary()
        for sound in dictionary['sounds']:
            sound_dir = os.path.join(cls.GOOD_SOUNDS_DIR, sound)
            os.makedirs(sound_dir, mode=0o755, exist_ok=True)  # Only creates if doesn't exist
            app.logger.debug(f"Created sound directory: {sound_dir}")

    @staticmethod
    def get_dictionary():
        try:
            config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "name": "Default",
                "sounds": ["ah", "eh", "ee", "oh", "oo"]
            }

    @classmethod
    def get_dictionaries(cls):
        try:
            with open(os.path.join(cls.CONFIG_DIR, 'dictionaries.json'), 'r') as f:
                return json.load(f)['dictionaries']
        except:
            return [{"name": "Default", "sounds": ["ah", "eh", "ee", "oh", "oo"]}]

    @classmethod
    def save_dictionaries(cls, dictionaries):
        with open(os.path.join(cls.CONFIG_DIR, 'dictionaries.json'), 'w') as f:
            json.dump({"dictionaries": dictionaries}, f, indent=4)

# Initialize directories
Config.init_directories()

# Add this to store the detector instance
sound_detector = None
detector_lock = threading.Lock()

# Global variables to store training stats
training_stats = None
training_history = None
model_summary_str = None

inference_stats = {
    'total_predictions': 0,
    'class_counts': {},
    'confidence_levels': []
}

app.latest_prediction = None

# Function to load the active dictionary
def load_active_dictionary():
    config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
    with open(config_file, 'r') as f:
        active_dict = json.load(f)
    return active_dict

# Function to get all dictionaries (if needed)
def get_all_dictionaries():
    config_file = os.path.join(Config.CONFIG_DIR, 'dictionaries.json')
    with open(config_file, 'r') as f:
        dictionaries = json.load(f)['dictionaries']
    return dictionaries

@app.route('/')
def index():
    app.logger.debug(f"Index route accessed")
    app.logger.debug(f"Static folder: {app.static_folder}")
    app.logger.debug(f"Static URL path: {app.static_url_path}")
    app.logger.debug(f"Template folder: {app.template_folder}")
    app.logger.debug(f"Templates exist: {os.path.exists(TEMPLATE_DIR)}")
    app.logger.debug(f"Login template exists: {os.path.exists(os.path.join(TEMPLATE_DIR, 'login.html'))}")
    if 'username' not in session:
        return render_template('login.html')
    return render_template('record.html', sounds=Config.get_dictionary()['sounds'])

@app.route('/login', methods=['POST'])
def login():
    if request.form.get('type') == 'admin':
        if request.form.get('password') == 'Michal':  # Admin password
            session['username'] = 'admin'
            session['is_admin'] = True
        else:
            flash('Invalid admin password')
            return render_template('login.html')
    else:
        username = request.form.get('username')
        if username:
            session['username'] = username
            session['is_admin'] = False
        else:
            flash('Username required')
            return render_template('login.html')
    
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/record', methods=['POST'])
def record():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    sound = request.form.get('sound')
    audio_data = request.files.get('audio')
    
    app.logger.debug(f"Recording attempt - Sound: {sound}")
    app.logger.debug(f"Audio data present: {audio_data is not None}")
    
    if sound and audio_data:
        # Convert WebM to WAV
        try:
            audio = AudioSegment.from_file(audio_data, format="webm")
            wav_data = io.BytesIO()
            audio.export(wav_data, format="wav")
            wav_data.seek(0)
        except Exception as e:
            app.logger.error(f"Error converting audio: {str(e)}")
            app.logger.error(f"Audio format: {audio_data.content_type}")
            # More detailed error message
            if "ffmpeg" in str(e).lower():
                return "FFmpeg not found. Please install FFmpeg.", 500
            return "Error processing audio", 500

        try:
            # Save recording temporarily
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_filename = f"{sound}_{session['username']}_{timestamp}.wav"
            temp_path = os.path.join(Config.TEMP_DIR, temp_filename)
            app.logger.debug(f"Saving to: {temp_path}")
            
            with open(temp_path, 'wb') as f:
                f.write(wav_data.read())
            
            # Process and chop the recording
            processor = SoundProcessor()
            chopped_files = processor.chop_recording(temp_path)
            
            app.logger.debug(f"Created {len(chopped_files)} chunks")
            
            # Clean up the original recording
            os.remove(temp_path)
            
            if not chopped_files:
                app.logger.error("No valid sound chunks found")
                return "No valid sound chunks found", 500
            
            return redirect(url_for('verify_chunks', timestamp=timestamp))
        except Exception as e:
            app.logger.error(f"Error processing recording: {str(e)}")
            app.logger.error(f"Stack trace:", exc_info=True)
            return "Error processing recording", 500
    
    app.logger.error(f"Missing data - Sound: {sound}, Audio: {audio_data is not None}")
    return redirect(url_for('index'))

@app.route('/verify/<timestamp>')
def verify_chunks(timestamp):
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get list of chunks for this recording
    chunks = [f for f in os.listdir(Config.TEMP_DIR) 
             if timestamp in f]
    
    if not chunks:
        flash('All chunks have been processed')
        return redirect(url_for('index'))
    
    return render_template('verify.html', chunks=chunks)

@app.route('/process_verification', methods=['POST'])
def process_verification():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    chunk_file = request.form.get('chunk_file')
    is_good = request.form.get('is_good') == 'true'
    timestamp = chunk_file.split('_')[-2]  # Get timestamp from filename
    
    if chunk_file:
        if is_good:
            # Move to goodsounds with proper naming
            sound = chunk_file.split('_')[0]
            username = session['username']
            existing_count = len([f for f in os.listdir(Config.GOOD_SOUNDS_DIR) 
                                if f.startswith(f"{sound}_{username}_")])
            new_filename = f"{sound}_{username}_{existing_count + 1}.wav"
            new_path = os.path.join(Config.GOOD_SOUNDS_DIR, new_filename)
            
            os.rename(
                os.path.join(Config.TEMP_DIR, chunk_file),
                new_path
            )
            flash(f'Chunk saved as: {new_path}')
        else:
            # Delete rejected chunk
            os.remove(os.path.join(Config.TEMP_DIR, chunk_file))
            flash('Chunk deleted')
    
    # Return to verify page with same timestamp to continue reviewing chunks
    return redirect(url_for('verify_chunks', timestamp=timestamp))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if not username:
            flash('Username is required')
            return render_template('register.html')
        
        # Check if username exists
        if username.lower() in ['admin', 'michal']:  # Protect both admin and michal usernames
            flash('This username is reserved')
            return render_template('register.html')
        
        session['username'] = username
        session['is_admin'] = False
        flash(f'Welcome {username}!')
        return redirect(url_for('index'))
    
    return render_template('register.html')

def get_sound_statistics():
    """Get counts of sound files in both system and user directories."""
    stats = {}
    dictionary = Config.get_dictionary()
    
    for sound in dictionary['sounds']:
        # Count files in goodsounds directory
        system_path = os.path.join(STATIC_DIR, 'goodsounds', sound)
        system_count = len([f for f in os.listdir(system_path)]) if os.path.exists(system_path) else 0
        
        # Count files in user recordings (if applicable)
        user_path = os.path.join(STATIC_DIR, 'recordings', sound) if 'username' in session else None
        user_count = len([f for f in os.listdir(user_path)]) if user_path and os.path.exists(user_path) else 0
        
        stats[sound] = {
            'system': system_count,
            'user': user_count
        }
    
    return stats

@app.route('/manage_dictionaries')
def manage_dictionaries():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('index'))
    
    dictionaries = Config.get_dictionaries()
    active_dict = Config.get_dictionary()
    sound_stats = get_sound_statistics()
    current_time = datetime.now().strftime("%I:%M:%S %p")
    
    return render_template('manage_dictionaries.html',
                         dictionaries=dictionaries,
                         active_dict=active_dict,
                         sound_stats=sound_stats,
                         current_time=current_time)

@app.route('/save_dictionary', methods=['POST'])
def save_dictionary():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    name = request.form.get('name')
    sounds = request.form.get('sounds').split(',')
    sounds = [s.strip() for s in sounds if s.strip()]
    
    # Create the dictionary object
    new_dict = {"name": name, "sounds": sounds}
    
    dictionaries = Config.get_dictionaries()
    
    # Update existing or add new
    found = False
    for d in dictionaries:
        if d['name'] == name:
            d['sounds'] = sounds
            found = True
            break
    
    if not found:
        dictionaries.append(new_dict)
    
    Config.save_dictionaries(dictionaries)
    
    # Also set this as the active dictionary
    with open(os.path.join(Config.CONFIG_DIR, 'active_dictionary.json'), 'w') as f:
        json.dump(new_dict, f, indent=4)
    
    flash('Dictionary saved and activated')
    return redirect(url_for('manage_dictionaries'))

@app.route('/list_recordings')
def list_recordings():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    # Get all recordings for this user (or all if admin)
    recordings = []
    if session.get('is_admin'):
        files = os.listdir(Config.GOOD_SOUNDS_DIR)
    else:
        files = [f for f in os.listdir(Config.GOOD_SOUNDS_DIR) 
                if f.split('_')[1] == session['username']]
    
    # Group by sound type
    recordings_by_sound = {}
    for file in files:
        sound = file.split('_')[0]
        if sound not in recordings_by_sound:
            recordings_by_sound[sound] = []
        recordings_by_sound[sound].append(file)
    
    return render_template('list_recordings.html', 
                         recordings=recordings_by_sound)

@app.route('/get_sound_stats')
def get_sound_stats():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    sound_stats = {}
    files = os.listdir(Config.GOOD_SOUNDS_DIR)
    current_user = session['username']
    
    # Initialize stats for all sounds in dictionary
    for sound in Config.get_dictionary()['sounds']:
        sound_stats[sound] = {
            'system_total': 0,
            'user_total': 0
        }
    
    # Count verified recordings
    for file in files:
        parts = file.split('_')
        if len(parts) >= 2:
            sound = parts[0]
            username = parts[1]
            if sound in sound_stats:
                sound_stats[sound]['system_total'] += 1
                if username == current_user:
                    sound_stats[sound]['user_total'] += 1
    
    return json.dumps(sound_stats)

@app.route('/set_active_dictionary', methods=['POST'])
def set_active_dictionary():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    name = request.form.get('name')
    if name:
        # Find the dictionary with this name
        dictionaries = Config.get_dictionaries()
        for dict in dictionaries:
            if dict['name'] == name:
                # Save as active dictionary
                with open(os.path.join(Config.CONFIG_DIR, 'active_dictionary.json'), 'w') as f:
                    json.dump(dict, f, indent=4)
                flash(f'Activated dictionary: {name}')
                break
    
    return redirect(url_for('manage_dictionaries'))

@app.route('/upload_sounds')
def upload_sounds():
    if 'username' not in session:
        return redirect(url_for('index'))
    return render_template('upload_sounds.html', sounds=Config.get_dictionary()['sounds'])

@app.route('/process_uploads', methods=['POST'])
def process_uploads():
    if 'username' not in session:
        return redirect(url_for('index'))
    
    sound = request.form.get('sound')
    files = request.files.getlist('files')
    
    if not sound or not files:
        flash('Please select both a sound type and files')
        return redirect(url_for('upload_sounds'))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processor = SoundProcessor()
    all_chunks = []
    
    for file in files:
        if file.filename.lower().endswith('.wav'):
            # Save temporarily
            temp_path = os.path.join(Config.TEMP_DIR, f"{sound}_{session['username']}_{timestamp}_temp.wav")
            file.save(temp_path)
            
            # Process and chop
            chunks = processor.chop_recording(temp_path)
            all_chunks.extend(chunks)
            
            # Clean up temp file
            os.remove(temp_path)
    
    if not all_chunks:
        flash('No valid sound chunks found in uploads')
        return redirect(url_for('upload_sounds'))
    
    return redirect(url_for('verify_chunks', timestamp=timestamp))

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if 'username' not in session or not session.get('is_admin'):
        return redirect(url_for('index'))

    if request.method == 'POST':
        try:
            # Load training data
            sound_folder = os.path.join(app.root_path, 'static', 'goodsounds')
            X_train, y_train, class_names, stats = build_dataset(sound_folder)

            # Store training statistics globally
            app.training_stats = {
                'total_samples': len(X_train),
                'class_distribution': {
                    class_names[i]: np.sum(y_train == i) 
                    for i in range(len(class_names))
                },
                'original_counts': stats['original_counts'],
                'augmented_counts': stats['augmented_counts']
            }

            # Build and train the model
            input_shape = (N_MFCC, X_train.shape[2], 1)
            model, model_summary = build_model(input_shape, num_classes=len(class_names))

            callback = TrainingCallback()
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.001,
                restore_best_weights=True
            )
            
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_delta=0.001,
                min_lr=0.00001
            )
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                batch_size=32,
                epochs=50,
                callbacks=[
                    callback, 
                    early_stopping,
                    reduce_lr,
                    tf.keras.callbacks.BackupAndRestore(
                        backup_dir='./checkpoints'
                    )
                ]
            )

            # Store the training history
            global training_history  # Declare as global
            training_history = {
                'epochs': len(history.history['accuracy']),
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            }

            # Load the active dictionary
            active_dict = load_active_dictionary()

            # Save the model with consistent naming
            safe_dict_name = active_dict['name'].replace(' ', '_')
            model_path = os.path.join('models', f"{safe_dict_name}_model.h5")
            model.save(model_path)
            app.logger.info(f"Model saved to: {model_path}")

            # Save class names
            np.save('models/class_names.npy', np.array(class_names))

            # Store statistics and model summary
            global training_stats, model_summary_str
            training_stats = stats
            model_summary_str = model_summary

            # Redirect to the model summary page
            return redirect(url_for('model_summary'))

        except Exception as e:
            app.logger.error(f"Error training model: {e}")
            flash(f'Error training model: {str(e)}')
            return redirect(url_for('train_model'))

    # GET request - show the training page
    dictionary = Config.get_dictionary()
    sounds = dictionary['sounds']
    return render_template('train_model.html', sounds=sounds)

@app.route('/model_summary')
def model_summary():
    # First check if training_stats exists and has data
    if not training_stats:
        # Return empty/default values if no training has occurred
        complete_training_stats = {
            'input_shape': '(13, 32, 1)',  # MFCC features shape
            'input_range': 'MFCC Features',
            'total_samples': 0,
            'original_counts': {},
            'augmented_counts': {},
            'feature_stats': {},
            'energy_comparison': {}
        }
    else:
        # If training_stats exists, safely get all values with defaults
        complete_training_stats = training_stats.copy()  # Use all stats directly
        
        # Debug logging to see what MFCC comparisons are available
        for i in range(13):
            key = f'mfcc_{i}_comparison' if i > 0 else 'energy_comparison'
            if key in training_stats:
                app.logger.info(f"Found comparison data for {key}")
            else:
                app.logger.info(f"Missing comparison data for {key}")

    return render_template('model_summary.html',
                         training_stats=complete_training_stats,
                         model_summary=model_summary_str,
                         training_history=training_history)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Load the active dictionary
    active_dict = load_active_dictionary()
    
    # Temporarily disable session check for testing
    if 'username' not in session:
        session['username'] = 'test_user'  # Set a default user
    
    if request.method == 'POST':
        try:
            use_microphone = request.form.get('input_type') == 'microphone'
            
            if use_microphone:
                # Handle microphone input
                model = models.load_model('models/audio_classifier.h5')
                with open('models/class_names.npy', 'rb') as f:
                    class_names = np.load(f, allow_pickle=True)
                
                predicted_class, confidence = predict_sound(
                    model, None, class_names, use_microphone=True)
                
                if predicted_class:
                    flash(f'Predicted sound: {predicted_class} (confidence: {confidence:.3f})')
                else:
                    flash('Error processing microphone input')
                    
            else:
                # Handle file upload
                if 'audio_data' not in request.files:
                    flash('No audio file provided')
                    return redirect(url_for('predict'))
    
                audio_file = request.files['audio_data']
                if audio_file.filename == '':
                    flash('No selected file')
                    return redirect(url_for('predict'))
    
                # Save the uploaded file temporarily
                temp_path = os.path.join(STATIC_DIR, 'temp', 'temp_audio.wav')
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                audio_file.save(temp_path)
    
                # Load the active dictionary
                active_dict = load_active_dictionary()
                class_names = active_dict['sounds']
                dictionary_name = active_dict['name']

                # Construct model path
                model_path = os.path.join('models', f"{dictionary_name}_model.h5")
                if not os.path.exists(model_path):
                    flash(f"Model for '{dictionary_name}' not found. Please train the model.")
                    return redirect(url_for('predict'))

                # Load the model
                model = models.load_model(model_path)

                # Use class_names from active_dict
                predicted_class, confidence = predict_sound(
                    model, temp_path, class_names, use_microphone=False)
                
                # Clean up
                os.remove(temp_path)
    
                if predicted_class:
                    flash(f'Predicted sound: {predicted_class} (confidence: {confidence:.3f})')
                else:
                    flash('Error processing audio file')
    
            return redirect(url_for('predict'))
    
        except Exception as e:
            flash(f'Error during prediction: {str(e)}')
            return redirect(url_for('predict'))
    
    return render_template('predict.html', active_dict=active_dict)

@app.route('/predict_sound', methods=['POST'])
def predict_sound_endpoint():
    try:
        # Load the active dictionary
        active_dict = load_active_dictionary()
        class_names = active_dict['sounds']
        dictionary_name = active_dict['name']

        # Construct model path
        model_path = os.path.join('models', f"{dictionary_name}_model.h5")
        if not os.path.exists(model_path):
            return jsonify({'error': f"Model for '{dictionary_name}' not found."}), 400

        # Load the model
        model = models.load_model(model_path)

        # Get audio data from the request
        audio_file = request.files['audio']
        audio_data = audio_file.read()

        # Save audio data to a temporary file if necessary or process directly
        # ...

        # Perform prediction
        predicted_class, confidence = predict_sound(
            model, audio_data, class_names, use_microphone=False)

        return jsonify({
            'predictions': [
                {'sound': predicted_class, 'probability': confidence}
            ]
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

# Add error handlers
@app.errorhandler(403)
def forbidden_error(error):
    app.logger.error(f"403 error: {error}")
    return "403 Forbidden - Access Denied", 403

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"404 error: {error}")
    return "404 Not Found", 404

@app.route('/training_status')
def training_status():
    """Return the current training status"""
    if hasattr(app, 'training_progress'):
        return jsonify({
            'progress': app.training_progress,
            'status': app.training_status
        })
    return jsonify({
        'progress': 0,
        'status': 'Not started'
    })

@app.route('/make_active', methods=['POST'])
def make_active():
    """Set a dictionary as the active one."""
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    name = request.form.get('name')
    if not name:
        flash('Dictionary name is required')
        return redirect(url_for('manage_dictionaries'))
    
    # Find the dictionary in the list
    dictionaries = Config.get_dictionaries()
    selected_dict = None
    for d in dictionaries:
        if d['name'] == name:
            selected_dict = d
            break
    
    if not selected_dict:
        flash('Dictionary not found')
        return redirect(url_for('manage_dictionaries'))
    
    # Save as active dictionary
    with open(os.path.join(Config.CONFIG_DIR, 'active_dictionary.json'), 'w') as f:
        json.dump(selected_dict, f, indent=4)
    
    # Create directories for any new sounds
    for sound in selected_dict['sounds']:
        sound_dir = os.path.join(Config.GOOD_SOUNDS_DIR, sound)
        os.makedirs(sound_dir, mode=0o755, exist_ok=True)
    
    flash(f'Dictionary "{name}" is now active')
    return redirect(url_for('manage_dictionaries'))

@app.route('/start_listening', methods=['POST'])
def start_listening():
    # Reset inference statistics
    app.inference_stats = {
        'total_predictions': 0,
        'class_counts': {},
        'confidence_levels': []
    }

    global sound_detector
    try:
        # Load the active dictionary
        active_dict = load_active_dictionary()
        class_names = active_dict['sounds']
        dictionary_name = active_dict['name']

        app.logger.info(f"Using active dictionary: {dictionary_name}")
        app.logger.info(f"Class names: {class_names}")

        # Load the model corresponding to the active dictionary
        # Replace spaces with underscores in the filename
        safe_dict_name = dictionary_name.replace(' ', '_')
        model_path = os.path.join('models', f"{safe_dict_name}_model.h5")
        
        app.logger.info(f"Looking for model at: {model_path}")
        if not os.path.exists(model_path):
            app.logger.error(f"Model not found at path: {model_path}")
            return jsonify({
                'status': 'error', 
                'message': f"Model for '{dictionary_name}' not found. Please train the model first."
            })

        app.logger.info(f"Loading model from: {model_path}")
        model = models.load_model(model_path)
        app.logger.info("Model loaded successfully")

        # Create a new SoundDetector instance with the loaded model and class names
        app.logger.info("Creating SoundDetector instance...")
        sound_detector = SoundDetector(model, class_names)
        app.logger.info("SoundDetector instance created")

        # Start listening with the prediction callback
        app.logger.info("Starting to listen...")
        sound_detector.start_listening(callback=prediction_callback)
        app.logger.info("Listener started successfully")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        app.logger.error(f"Error in start_listening: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_listening', methods=['POST'])
def stop_listening():
    # Stop listening
    global sound_detector
    try:
        if sound_detector:
            sound_detector.stop_listening()
            sound_detector = None
            app.logger.info("Listener stopped successfully")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Not currently listening'})
    except Exception as e:
        app.logger.error(f"Error in stop_listening: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/prediction_stream')
def prediction_stream():
    def generate():
        try:
            while True:
                if app.latest_prediction:
                    data = json.dumps(app.latest_prediction)
                    yield f"data: {data}\n\n"
                    app.latest_prediction = None  # Reset after sending
                else:
                    # Send a heartbeat to keep the connection alive
                    yield ": heartbeat\n\n"
                time.sleep(0.1)
        except GeneratorExit:
            app.logger.info("Client closed the stream")
        except Exception as e:
            app.logger.error(f"Error in prediction_stream: {e}")
            yield f"data: {json.dumps({'error': 'Stream error'})}\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no'  # For Nginx buffering issues
    })

@app.route('/inference_statistics')
def inference_statistics():
    # Calculate average confidence
    if app.inference_stats['confidence_levels']:
        avg_confidence = sum(app.inference_stats['confidence_levels']) / len(app.inference_stats['confidence_levels'])
    else:
        avg_confidence = 0.0

    return render_template('inference_statistics.html',
                           inference_stats=app.inference_stats,
                           avg_confidence=avg_confidence)

def prediction_callback(prediction):
    app.logger.info(f"Got prediction: {prediction}")
    app.latest_prediction = prediction

    # Update inference statistics
    app.inference_stats['total_predictions'] += 1
    class_name = prediction['class']
    confidence = prediction['confidence']

    # Update class counts
    app.inference_stats['class_counts'].setdefault(class_name, 0)
    app.inference_stats['class_counts'][class_name] += 1

    # Update confidence levels
    app.inference_stats['confidence_levels'].append(confidence)

if __name__ == '__main__':
    app.run(debug=True)