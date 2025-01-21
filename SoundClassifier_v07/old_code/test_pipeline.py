import os
import logging
from trainer import SoundTrainer
from model import SoundClassifier
from feature_extractor import AudioFeatureExtractor

logging.basicConfig(level=logging.INFO)

def test_pipeline():
    """Test the complete ML pipeline."""
    
    # Initialize with path to goodsounds directory
    good_sounds_dir = os.path.join('static', 'goodsounds')
    model_dir = os.path.join('models')
    
    # Create trainer
    trainer = SoundTrainer(good_sounds_dir=good_sounds_dir, model_dir=model_dir)
    
    # Test data collection
    logging.info("Testing data collection...")
    X, y = trainer.collect_training_data()
    if X is not None:
        logging.info(f"Successfully collected {len(X)} samples with {len(X[0])} features")
        logging.info(f"Unique classes: {set(y)}")
    else:
        logging.error("Failed to collect training data")
        return
    
    # Test model training
    logging.info("\nTesting model training...")
    success = trainer.train_model(n_estimators=100)
    if success:
        logging.info("Model trained successfully")
    else:
        logging.error("Failed to train model")
        return
    
    # Test model evaluation
    logging.info("\nTesting model evaluation...")
    metrics = trainer.evaluate_model(test_size=0.2)
    if metrics:
        logging.info(f"Model accuracy: {metrics['accuracy']:.2f}")
        logging.info("\nClassification Report:")
        logging.info(metrics['report'])
    else:
        logging.error("Failed to evaluate model")
        return
    
    # Test single prediction
    logging.info("\nTesting single prediction...")
    feature_extractor = AudioFeatureExtractor()
    classifier = SoundClassifier(model_dir=model_dir)
    classifier.load()
    
    # Try prediction on first file in goodsounds
    test_file = os.path.join(good_sounds_dir, os.listdir(good_sounds_dir)[0])
    features = feature_extractor.extract_features(test_file)
    if features:
        feature_values = []
        for name in feature_extractor.get_feature_names():
            if name.startswith('mfcc_'):
                idx = int(name.split('_')[1])
                if 'mean' in name:
                    feature_values.append(features['mfcc_mean'][idx])
                else:
                    feature_values.append(features['mfcc_std'][idx])
            else:
                feature_values.append(features[name])
        
        predictions = classifier.get_top_predictions([feature_values])
        logging.info(f"Top predictions for {os.path.basename(test_file)}:")
        for sound, prob in predictions:
            logging.info(f"  {sound}: {prob:.2f}")
    
    logging.info("\nPipeline test complete!")

if __name__ == "__main__":
    test_pipeline() 