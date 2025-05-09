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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class URLClassifier:

    safe_tlds = {
            'com', 'org', 'edu', 'gov', 'net', 'io', 'co', 'me', 'info',
            'app', 'dev', 'ai', 'cloud', 'tech', 'mil', 'int'
        }

    def __init__(self):
        """Initialize the URL classifier"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'url_length', 'path_length', 'query_length',
            'num_dots', 'num_slashes', 'num_digits',
            'num_special_chars', 'has_suspicious_extension',
            'has_suspicious_keyword', 'is_legitimate_domain',
            'has_https', 'has_http'
        ]
        self.model_path = 'my_model.keras'
        self.scaler_path = 'scaler.joblib'
        self.batch_size = 1000  # Process 1000 URLs at a time

    def preprocess_url(self, url):
        """Preprocess URL for feature extraction"""
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
            parsed = urlparse(('http://' + url) if not (has_http or has_https) else (('https://' if has_https else 'http://') + url))
            
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
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            np.array: Feature matrix
        """
        # Select numerical features
        X = df[self.feature_columns].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X

    def create_model(self, input_dim):
        """
        Create the neural network model
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            tf.keras.Model: Compiled model
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
        """Process a batch of URLs"""
        X_batch = self.extract_features(urls_batch)
        
        if training and self.scaler is None:
            self.scaler = StandardScaler(with_mean=False)  # Don't center the data to keep sparsity
            X_batch = self.scaler.fit_transform(X_batch)
        elif self.scaler is not None:
            X_batch = self.scaler.transform(X_batch)
            
        return X_batch, labels_batch

    def train(self, dataset_path):
        """
        Train the model on the dataset
        
        Args:
            dataset_path (str): Path to the processed dataset
            
        Returns:
            dict: Training history
        """
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        # Create binary labels from url_status
        y = (df['url_status'] != 'online').astype(int)
        
        # Extract features
        X = self.extract_features(df)
        
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
        
        # Save model and scaler
        self.save_model()
        
        return history.history

    def is_safe_domain(self, url):
        """Check if the URL belongs to a known safe domain"""
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
                if tld in self.safe_tlds and len(parts) <= 3:  # Allow up to one subdomain for safe TLDs
                    return True
            
            return False
        except:
            return False

    def predict(self, url):
        """
        Predict if a URL is malicious
        
        Args:
            url (str): URL to check
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            self.load_model()
            
        # First check if it's a known safe domain
        if self.is_safe_domain(url):
            return {
                'url': url,
                'prediction': 'safe',
                'confidence': 0.95,
                'reason': 'Known safe domain'
            }
            
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
        
        return {
            'url': url,
            'prediction': 'malicious' if is_malicious else 'safe',
            'confidence': prediction if is_malicious else 1 - prediction,
            'reason': reason
        }
        
    def _generate_reason(self, features, prediction):
        """
        Generate explanation for the prediction
        
        Args:
            features (dict): URL features
            prediction (float): Model prediction
            
        Returns:
            str: Explanation for the prediction
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
        """Save the model and scaler"""
        if self.model is not None:
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        """Load the model and scaler"""
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model file not found")
