from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import json
from sound_processor import SoundProcessor
from datetime import datetime
from pydub import AudioSegment
import io
import logging

# Get absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(CURRENT_DIR, 'templates')
STATIC_DIR = os.path.join(CURRENT_DIR, 'static')

app = Flask(__name__, 
    static_url_path='/static',
    static_folder=STATIC_DIR,
    template_folder=TEMPLATE_DIR)
app.secret_key = 'your-secret-key'  # For session management

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
            os.makedirs(directory, mode=0o755, exist_ok=True)
            app.logger.debug(f"Created directory: {directory}")

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

@app.route('/manage_dictionaries')
def manage_dictionaries():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    # Get sound statistics
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
    
    return render_template('manage_dictionaries.html',
                         dictionaries=Config.get_dictionaries(),
                         active_dictionary=Config.get_dictionary(),
                         sound_stats=sound_stats)

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

# Add error handlers
@app.errorhandler(403)
def forbidden_error(error):
    app.logger.error(f"403 error: {error}")
    return "403 Forbidden - Access Denied", 403

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"404 error: {error}")
    return "404 Not Found", 404

if __name__ == '__main__':
    app.run(debug=True) 