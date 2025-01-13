from flask import Blueprint, jsonify, request, render_template, flash, redirect, url_for, session
from ml.trainer import SoundTrainer
from ml.feature_extractor import AudioFeatureExtractor
import os
import numpy as np
from config import Config
import logging
from pydub import AudioSegment
from flask_socketio import emit
from run import classifier  # Import the global classifier instance

ml_bp = Blueprint('ml', __name__)

# Initialize ML components
trainer = SoundTrainer(good_sounds_dir=Config.GOOD_SOUNDS_DIR)
feature_extractor = AudioFeatureExtractor(sr=22050)
# Load the model if it exists
model_path = os.path.join('models', 'sound_classifier.joblib')
if os.path.exists(model_path):
    logging.info(f"Loading model from {model_path}")
    success = classifier.load()
    if classifier.model is None:
        logging.error("Failed to load model")
    elif not success:
        logging.error("Model loaded but feature names not set")
    else:
        logging.info(f"Model loaded with classes: {classifier.classes_}")
        logging.info(f"Model feature names: {classifier.feature_names}")
else:
    logging.warning(f"No model found at {model_path}")

@ml_bp.route('/train', methods=['POST'])
def train_model():
    if not session.get('is_admin'):
        return jsonify({"error": "Admin access required"}), 403
    
    def progress_callback(progress):
        # Emit progress to the client
        emit('training_progress', {'progress': progress}, namespace='/train', broadcast=True)

    try:
        logging.info("Starting model training...")
        success = trainer.train_model(progress_callback=progress_callback)
        if success:
            flash("Model trained successfully!")
            logging.info("Model training completed successfully")
        else:
            error_msg = "Error training model - check logs"
            flash(error_msg)
            logging.error(error_msg)

        # Reload the model after training
        model_loaded = classifier.load()
        if model_loaded:
            logging.info("Model reloaded successfully after training")
        else:
            logging.error("Failed to reload model after training")

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        flash(error_msg)
        logging.error(error_msg)
        logging.error("Full traceback:", exc_info=True)
    
    return redirect(url_for('ml.model_status'))

@ml_bp.route('/model-status')
def model_status():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    # Check if the model is loaded
    model_trained = classifier.model is not None

    # Proceed with model evaluation or status display
    metrics = trainer.evaluate_model() if model_trained else None
    return render_template('ml/model_status.html', metrics=metrics, model_trained=model_trained)

@ml_bp.route('/predict_sound', methods=['POST'])
def predict_sound():
    logging.info("Headers: %s", dict(request.headers))
    logging.info("Files: %s", request.files.keys())
    logging.info("Form data: %s", request.form)
    
    if not classifier.model:
        logging.error("No model loaded")
        return jsonify({"error": "No model loaded"}), 500
        
    if not hasattr(classifier, 'feature_names') or classifier.feature_names is None:
        # Try reloading the model
        success = classifier.load()
        logging.info(f"Reloading model result: {success}")
        if not success or classifier.feature_names is None:
            logging.error("Model feature names not available")
            return jsonify({"error": "Model not properly initialized"}), 500
        
    # Debug model state
    logging.info(f"Model classes: {classifier.classes_}")
    logging.info(f"Model feature names: {classifier.feature_names}")
    logging.info(f"Current dictionary: {Config.get_dictionary()['name']}")
    logging.info(f"Dictionary sounds: {Config.get_dictionary()['sounds']}")
    
    logging.info("Received prediction request")
    if 'audio' not in request.files:
        logging.error("No audio file in request")
        return jsonify({"error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    logging.info(f"Received audio file of type: {audio_file.content_type}")
    
    # Save temporarily
    temp_path = os.path.join(Config.TEMP_DIR, 'predict_temp.wav')
    try:
        # Convert WebM to WAV using pydub
        audio = AudioSegment.from_file(audio_file, format="webm")
        # Normalize audio
        audio = audio.normalize()
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        # Trim silence
        audio = audio.strip_silence(
            silence_thresh=-50,   # Less aggressive trimming
            min_silence_len=100   # Require shorter silence before trimming
        )
        audio.export(temp_path, format="wav")
        logging.info(f"Saved audio to {temp_path}")
    except Exception as e:
        logging.error(f"Error converting audio: {str(e)}")
        logging.exception("Full traceback:")
        return jsonify({"error": "Error processing audio"}), 500
    
    try:
        # Extract features using the updated feature extractor
        features = feature_extractor.extract_features(temp_path)
        if features is None:
            logging.error("Failed to extract features")
            return jsonify({"error": "Error extracting features"}), 500

        logging.info("Features extracted successfully")

        # Format features
        feature_values = []
        feature_names = (
            classifier.feature_names
            if classifier.feature_names is not None
            else feature_extractor.get_feature_names()
        )
        logging.info(f"Prediction feature names: {feature_names}")

        # Ensure feature order matches during training and inference
        for name in feature_names:
            try:
                feature_values.append(features[name])
            except KeyError:
                logging.error(f"Missing feature {name} in extracted features")
                return jsonify({"error": f"Missing feature {name}"}), 500

        # Convert feature_values to NumPy array and reshape
        feature_array = np.array(feature_values).reshape(1, -1)

        # Apply scaler if available
        if classifier.scaler is not None:
            feature_array = classifier.scaler.transform(feature_array)
        else:
            logging.warning("Scaler not found. Skipping feature scaling.")

        # Get predictions
        predictions = classifier.get_top_predictions(feature_array)
        if predictions:
            predictions = predictions[0]  # Since we have only one sample

        # Return predictions as JSON
        return jsonify({"predictions": predictions})

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@ml_bp.route('/live-inference')
def live_inference():
    """Real-time inference interface."""
    if not session.get('username'):
        return redirect(url_for('index'))
    
    # Check if model exists and is loaded
    model_path = os.path.join('models', 'sound_classifier.joblib')
    if not os.path.exists(model_path):
        flash("No trained model found. Please train a model first.", "error")
        return redirect(url_for('ml.model_status'))
    
    if not classifier.model:
        classifier.load()
        if not classifier.model:
            flash("Error loading model. Please retrain.", "error")
            return redirect(url_for('ml.model_status'))
    
    return render_template('ml/live_inference.html', 
                         dictionary=Config.get_dictionary()) 

@ml_bp.route('/test_prediction')
def test_prediction():
    """Test route to verify model predictions."""
    if not classifier.model:
        return jsonify({"error": "No model loaded"})
    
    # Get a sample from the training data
    X, y = trainer.collect_training_data()
    if X is None or len(X) == 0:
        return jsonify({"error": "No training data found"})
    
    # Try prediction on first sample
    sample = X[0:1]  # Keep 2D shape
    predictions = classifier.get_top_predictions(sample)
    
    return jsonify({
        "input_shape": sample.shape,
        "feature_count": classifier.feature_count,
        "actual_label": y[0],
        "predictions": predictions,
        "model_classes": classifier.classes_.tolist()
    }) 