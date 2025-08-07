import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import re
import urllib.parse
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, vstack
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import keras
from __types__.name import NameManagement

# Download required NLTK data for text processing
nltk.download('punkt')
nltk.download('stopwords')


class URLClassifier:
    """
    URL Classifier using Neural Network for malicious URL detection

    
    - Uses a feed-forward neural network with dropout layers for regularization
    - Implements feature engineering for URL analysis
    - Binary classification with sigmoid activation
    - Uses early stopping to prevent overfitting
    """

    # Set of known safe top-level domains
    safe_tlds = {
        'com', 'org', 'edu', 'gov', 'net', 'io', 'co', 'me', 'info',
        'app', 'dev', 'ai', 'cloud', 'tech', 'mil', 'int'
    }

    def __init__(self):
        """
        Initialize the URL classifier with default parameters

        
        - Initializes neural network model as None (to be created during training)
        - Sets up StandardScaler for feature normalization
        - Defines feature columns for model input
        - Sets batch size for processing large datasets
        """
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'url_length', 'path_length', 'query_length',
            'num_dots', 'num_slashes', 'num_digits',
            'num_special_chars', 'has_suspicious_extension',
            'has_suspicious_keyword', 'is_legitimate_domain',
            'has_https', 'has_http'
        ]
        name_manager = NameManagement()
        self.model_path = f'{name_manager.getFileName()}.keras'
        self.scaler_path = f'{name_manager.getScalerName()}.joblib'
        self.batch_size = 1000

    def preprocess_url(self, url):
        """
        Preprocess URL for feature extraction

        
        1. URL Normalization:
           - Convert to lowercase
           - Remove protocol (http/https)
           - Remove www prefix
        2. Feature Extraction:
           - Parse URL components using urlparse
           - Extract domain features
           - Identify TLD safety
           - Analyze path and query parameters
        3. Feature Vector Creation:
           - Combine all extracted features
           - Clean and normalize feature strings
        """
        # Convert to lowercase
        url = url.lower()

        # Store original URL parts before removing protocols
        has_https = 'https://' in url
        has_http = 'http://' in url

        # Remove protocol
        url = re.sub(r'^https?://', '', url)

        # Store www presence before removing
        has_www = url.startswith('www.')

        # Remove www.
        url = re.sub(r'^www\.', '', url)

        # Extract features from URL structure
        try:
            parsed = urlparse(('http://' + url) if not (has_http or has_https)
                              else (('https://' if has_https else 'http://') + url))

            # Extract domain features
            domain = parsed.netloc
            domain_parts = domain.split('.')
            is_ip = all(part.isdigit() for part in domain_parts)

            # Common TLDs that are generally safe
            tld = domain_parts[-1] if domain_parts else ''
            is_safe_tld = tld in self.safe_tlds

            # Extract path and query features
            path_parts = [p for p in parsed.path.split('/') if p]
            query_parts = parsed.query.split('&') if parsed.query else []

            # Combine features with special markers
            features = []

            # Add protocol features
            if has_https:
                features.append('HTTPS')
            if has_http:
                features.append('HTTP')
            if has_www:
                features.append('WWW')

            # Add domain features
            features.append(f"DOMAIN_{domain}")
            for part in domain_parts:
                features.append(f"DOMAIN_PART_{part}")

            # Add TLD features
            features.append(f"TLD_{tld}")
            if is_safe_tld:
                features.append('SAFE_TLD')

            # Add path features
            for part in path_parts:
                features.append(f"PATH_{part}")

            # Add query features
            for part in query_parts:
                features.append(f"QUERY_{part}")

            # Add special features
            if is_ip:
                features.append('IS_IP_ADDRESS')
            if len(domain_parts) > 3:
                features.append('MANY_SUBDOMAINS')
            if len(path_parts) > 5:
                features.append('DEEP_PATH')
            if len(query_parts) > 3:
                features.append('MANY_QUERY_PARAMS')

            # Remove empty strings and clean features
            features = [re.sub(r'[^\w\s-]', '', f) for f in features if f]

            return ' '.join(features)
        except:
            return url

    def extract_features(self, df):
        """
        Extract features from the processed dataset

        
        1. Feature Selection:
           - Select numerical features from DataFrame
           - Apply StandardScaler for normalization
        2. Data Transformation:
           - Convert features to numpy array
           - Scale features to zero mean and unit variance
        """
        # Select numerical features
        X = df[self.feature_columns].values

        # Scale features
        X = self.scaler.fit_transform(X)

        return X

    def create_model(self, input_dim):
        """
        Create the neural network model

        Technical Architecture:
        1. Input Layer: Takes input_dim features
        2. Hidden Layer 1: 64 neurons with ReLU activation
        3. Dropout Layer 1: 30% dropout for regularization
        4. Hidden Layer 2: 32 neurons with ReLU activation
        5. Dropout Layer 2: 20% dropout for regularization
        6. Hidden Layer 3: 16 neurons with ReLU activation
        7. Output Layer: 1 neuron with sigmoid activation

        Optimization:
        - Optimizer: Adam (adaptive moment estimation)
        - Loss: Binary Cross-Entropy
        - Metrics: Accuracy
        """
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def process_batch(self, urls_batch, labels_batch, training=True):
        """
        Process a batch of URLs

        
        1. Feature Extraction:
           - Extract features from URL batch
           - Apply scaling if scaler exists
        2. Data Preparation:
           - Handle training vs inference mode
           - Apply appropriate scaling
        """
        X_batch = self.extract_features(urls_batch)

        if training and self.scaler is None:
            # Don't center the data to keep sparsity
            self.scaler = StandardScaler(with_mean=False)
            X_batch = self.scaler.fit_transform(X_batch)
        elif self.scaler is not None:
            X_batch = self.scaler.transform(X_batch)

        return X_batch, labels_batch

    def train(self, dataset_path):
        """
        Train the model on the dataset

        
        1. Data Preparation:
           - Load and preprocess data
           - Create binary labels
           - Split into train/test sets (80/20)
        2. Model Training:
           - Create and compile model
           - Apply early stopping
           - Train for maximum 50 epochs
        3. Performance Evaluation:
           - Calculate metrics (accuracy, precision, recall, F1)
           - Generate confusion matrix
           - Save training history
        """
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        # Create binary labels from url_status
        y = (df['url_status'] != 'online').astype(int)

        # Extract features
        X = self.extract_features(df)

        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train model
        self.model = self.create_model(X.shape[1])

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )

        # Calculate and save performance metrics
        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
        }

        # Save metrics
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save model and scaler
        self.save_model()

        return history.history

    def is_safe_domain(self, url):
        """
        Check if the URL belongs to a known safe domain

        
        1. URL Parsing:
           - Remove protocol and www
           - Extract domain
        2. Safety Checks:
           - Check for localhost/IP
           - Verify against safe TLDs
           - Analyze domain structure
        """
        try:
            # Remove protocol and www
            url = re.sub(r'^https?://', '', url.lower())
            url = re.sub(r'^www\.', '', url)

            # Extract domain
            domain = url.split('/')[0].split(':')[0]  # Remove path and port

            # Check if it's localhost or IP
            if domain in ['localhost', '127.0.0.1'] or domain.startswith('192.168.') or domain.startswith('10.'):
                return True

            # Check if it's a known safe domain
            if domain in self.safe_tlds:
                return True

            # Check domain parts
            parts = domain.split('.')
            if len(parts) >= 2:
                # Check if main domain is in safe list
                main_domain = '.'.join(parts[-2:])
                if main_domain in self.safe_tlds:
                    return True

                # Check TLD
                tld = parts[-1]
                # Allow up to one subdomain for safe TLDs
                if tld in self.safe_tlds and len(parts) <= 3:
                    return True

            return False
        except:
            return False

    def predict(self, url):
        """
        Predict if a URL is malicious

        
        1. Preprocessing:
           - Check for known safe domains
           - Extract URL components
           - Calculate features
        2. Feature Processing:
           - Scale features
           - Prepare input for model
        3. Prediction:
           - Get model prediction
           - Calculate confidence
           - Generate explanation
        """
        if self.model is None:
            self.load_model()

        # First check if it's a known safe domain
        if self.is_safe_domain(url):
            return 'safe', 0.95, 'Known safe domain'

        # Create a DataFrame with the URL
        df = pd.DataFrame({'url': [url]})

        # Extract URL components and features
        from data_processes import DataProcessor
        processor = DataProcessor(None, None)

        # Get URL components
        components = processor.extract_url_components(url)
        metrics = processor.calculate_url_metrics(url)
        patterns = processor.check_suspicious_patterns(url)

        # Combine features
        features = {**components, **metrics, **patterns}
        feature_df = pd.DataFrame([features])

        # Select and scale features
        X = feature_df[self.feature_columns].values
        X = self.scaler.transform(X)

        # Make prediction
        prediction = self.model.predict(X)[0][0]
        is_malicious = prediction > 0.5

        # Generate reason
        reason = self._generate_reason(features, prediction)

        return ('malicious' if is_malicious else 'safe',
                prediction if is_malicious else 1 - prediction,
                reason)

    def _generate_reason(self, features, prediction):
        """
        Generate explanation for the prediction

        
        1. Feature Analysis:
           - Check suspicious patterns
           - Analyze feature values
        2. Reason Generation:
           - Combine multiple factors
           - Provide confidence-based explanation
        """
        reasons = []

        if features['has_suspicious_extension']:
            reasons.append("Contains suspicious file extension")
        if features['has_suspicious_keyword']:
            reasons.append("Contains suspicious keywords")
        if not features['is_legitimate_domain']:
            reasons.append("Not a known legitimate domain")
        if features['num_special_chars'] > 5:
            reasons.append("Contains many special characters")
        if features['num_digits'] > 3:
            reasons.append("Contains many digits")

        if not reasons:
            if prediction > 0.7:
                reasons.append("High confidence malicious prediction")
            else:
                reasons.append("No suspicious patterns detected")

        return " | ".join(reasons)

    def save_model(self):
        """
        Save the model and scaler

        
        - Save Keras model in .keras format
        - Save StandardScaler using joblib
        """
        if self.model is not None:
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """
        Load the model and scaler

        
        - Load Keras model from .keras file
        - Load StandardScaler from joblib file
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model file not found")

    def visualize_model(self, save_path='plots/'):
        """
        Visualize the model architecture

        
        - Use keras.utils.plot_model for visualization
        - Save architecture diagram as PNG
        - Include layer shapes and connections
        """
        if self.model is None:
            raise ValueError(
                "Model not initialized. Please train or load the model first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        keras.utils.plot_model(
            self.model,
            to_file=save_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=False,
            rankdir="TB",
            expand_nested=False,
            dpi=200,
            show_layer_activations=False,
            show_trainable=False,
        )
