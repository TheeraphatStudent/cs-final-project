import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlparse
# import tldextract
# import requests
# from datetime import datetime
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
from data_processor import URLDataProcessor
from sklearn.utils.class_weight import compute_class_weight

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger.info("GPU is available and configured")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")
else:
    logger.info("No GPU found. Using CPU")

class URLSafetyModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Use 1-gram, 2-gram, and 3-gram
            stop_words='english'
        )
        self.svm_model = None
        self.text_pipeline = None
        self.numeric_pipeline = None
        self.data_processor = URLDataProcessor()
        
        # Common TLDs for domain parsing
        self.common_tlds = [
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'io', 'co', 'uk',
            'us', 'ca', 'au', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 'in'
        ]
        
        # Download required NLTK data with better error handling
        try:
            logger.info("Downloading required NLTK resources...")
            # Download all necessary NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {str(e)}")
            raise RuntimeError("Failed to download required NLTK resources. Please check your internet connection and try again.")
        
    def preprocess_url_text(self, url):
        """Preprocess URL text for TF-IDF vectorization"""
        # Convert to lowercase
        url = url.lower()
        
        # Remove special characters and split on common URL separators
        url = re.sub(r'[^a-zA-Z0-9\s]', ' ', url)
        
        # Simple tokenization without relying on NLTK's punkt
        tokens = url.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Join tokens back into string
        return ' '.join(tokens)

    def extract_features(self, url):
        """Extract comprehensive features from URL"""
        # Text features for TF-IDF
        text_features = self.preprocess_url_text(url)
        
        # Numeric features
        numeric_features = []
        
        # Basic URL features
        numeric_features.append(len(url))
        numeric_features.append(len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url)))
        numeric_features.append(len(re.findall(r'\d', url)))
        numeric_features.append(url.count('.'))
        numeric_features.append(1 if 'https' in url else 0)
        
        # Parse URL components
        parsed_url = urlparse(url)
        domain_info = self.data_processor.extract_domain_components(url)
        
        # Domain features
        numeric_features.append(len(domain_info['domain']))
        numeric_features.append(len(domain_info['suffix']))
        numeric_features.append(1 if domain_info['subdomain'] else 0)
        
        # Path features
        numeric_features.append(len(parsed_url.path.split('/')))
        numeric_features.append(len(parsed_url.query))
        
        # Suspicious patterns
        suspicious_extensions = ['.exe', '.bat', '.cmd', '.sh', '.msi', '.dll', '.vbs', '.ps1']
        suspicious_keywords = ['login', 'signin', 'verify', 'account', 'secure', 'update', 'install']
        
        numeric_features.append(1 if any(ext in url.lower() for ext in suspicious_extensions) else 0)
        numeric_features.append(1 if any(keyword in url.lower() for keyword in suspicious_keywords) else 0)
        
        # URL encoding features
        numeric_features.append(url.count('%'))
        numeric_features.append(url.count('&'))
        
        # Protocol features
        numeric_features.append(1 if parsed_url.scheme == 'https' else 0)
        numeric_features.append(1 if parsed_url.scheme == 'http' else 0)
        
        return {
            'text': text_features,
            'numeric': np.array(numeric_features)
        }

    def build_model(self):
        """Build and compile the enhanced Keras model"""
        # Create input layer
        input_layer = layers.Input(shape=(16,))
        
        # Build the model using functional API
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        output_layer = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall(),
                    keras.metrics.AUC()]
        )
        
    def build_svm_model(self):
        """Build SVM model with text and numeric features"""
        # Create pipelines for text and numeric features
        self.text_pipeline = Pipeline([
            ('tfidf', self.tfidf),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        
        self.numeric_pipeline = Pipeline([
            ('scaler', self.scaler),
            ('svm', SVC(kernel='rbf', probability=True))
        ])
        
        # Combine pipelines
        self.svm_model = ColumnTransformer([
            ('text', self.text_pipeline, 'text'),
            ('numeric', self.numeric_pipeline, 'numeric')
        ])
        
    def load_dataset(self, file_path):
        """Load dataset from CSV or Excel file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
        # Verify required columns
        required_columns = ['url', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
            
        return df
        
    def prepare_data(self, urls, labels):
        """Prepare and preprocess data for training"""
        # Extract features
        features = [self.extract_features(url) for url in urls]
        
        # Separate text and numeric features
        text_features = [f['text'] for f in features]
        numeric_features = np.array([f['numeric'] for f in features])
        
        # Scale numeric features
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features,  # Only use numeric features for neural network
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, urls, labels, epochs=20, batch_size=32, validation_split=0.2):
        """Train both neural network and SVM models"""
        if self.model is None:
            self.build_model()
            
        if self.svm_model is None:
            self.build_svm_model()
            
        X_train, X_test, y_train, y_test = self.prepare_data(urls, labels)
        
        # Train neural network
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        class_weights = self._calculate_class_weights(y_train)
        
        # Train neural network on numeric features
        nn_history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return {'nn_history': nn_history}
    
    def predict(self, url):
        """Predict if a URL is safe or dangerous using both models"""
        if self.model is None or self.svm_model is None:
            raise ValueError("Models have not been trained yet!")
            
        # Extract features
        features = self.extract_features(url)
        
        # Get predictions from both models
        nn_features = self.scaler.transform(features['numeric'].reshape(1, -1))
        nn_prediction = self.model.predict(nn_features)[0][0]
        
        svm_prediction = self.svm_model.predict_proba({
            'text': [features['text']],
            'numeric': [features['numeric']]
        })[0][1]  # Probability of positive class
        
        # Combine predictions (ensemble)
        combined_prediction = (nn_prediction + svm_prediction) / 2
        
        # Calculate confidence
        confidence = abs(combined_prediction - 0.5) * 2
        
        return {
            'url': url,
            'safety_score': float(combined_prediction),
            'is_safe': combined_prediction > 0.5,
            'confidence': float(confidence),
            'risk_level': self._get_risk_level(combined_prediction, confidence),
            'nn_score': float(nn_prediction),
            'svm_score': float(svm_prediction)
        }
    
    def save_model(self, filepath):
        """Save all models and preprocessing objects"""
        if self.model is None or self.svm_model is None:
            raise ValueError("No models to save!")
            
        # Save neural network model
        self.model.save(filepath)
        
        # Save SVM model and preprocessing objects
        import joblib
        joblib.dump(self.svm_model, filepath.replace('.keras', '_svm.joblib'))
        joblib.dump(self.scaler, filepath.replace('.keras', '_scaler.joblib'))
        joblib.dump(self.label_encoder, filepath.replace('.keras', '_encoder.joblib'))
        joblib.dump(self.tfidf, filepath.replace('.keras', '_tfidf.joblib'))
    
    def load_model(self, filepath):
        """Load all models and preprocessing objects"""
        self.model = keras.models.load_model(filepath)
        
        import joblib
        self.svm_model = joblib.load(filepath.replace('.keras', '_svm.joblib'))
        self.scaler = joblib.load(filepath.replace('.keras', '_scaler.joblib'))
        self.label_encoder = joblib.load(filepath.replace('.keras', '_encoder.joblib'))
        self.tfidf = joblib.load(filepath.replace('.keras', '_tfidf.joblib'))

    def _calculate_class_weights(self, y):
        """Calculate class weights to handle imbalanced data"""
        try:
            # Get unique classes
            classes = np.unique(y)
            # Calculate class weights
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y
            )
            # Create dictionary mapping class indices to weights
            class_weights = dict(zip(classes, weights))
            logger.info(f"Calculated class weights: {class_weights}")
            return class_weights
        except Exception as e:
            logger.warning(f"Error calculating class weights: {str(e)}. Using default weights.")
            return {0: 1.0, 1: 1.0}  # Default weights if calculation fails