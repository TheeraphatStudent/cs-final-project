import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
import joblib
import re
import urllib.parse
from urllib.parse import urlparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from __types__.name import NameManagement

# Download required NLTK data for text processing
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class URLClassifierSVM:
    safe_tlds = {
        'com', 'org', 'edu', 'gov', 'net', 'io', 'co', 'me', 'info',
        'app', 'dev', 'ai', 'cloud', 'tech', 'mil', 'int', 'uk', 'de', 'fr'
    }

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        self.numerical_features = [
            'url_length', 'path_length', 'query_length', 'fragment_length',
            'num_dots', 'num_slashes', 'num_digits', 'num_special_chars',
            'num_underscores', 'num_hyphens', 'num_equals', 'num_ampersands',
            'domain_length', 'subdomain_count', 'path_depth', 'query_param_count',
            'has_https', 'has_www', 'is_ip_address', 'has_suspicious_extension',
            'has_suspicious_keyword', 'is_legitimate_domain', 'entropy_score'
        ]
        
        name_manager = NameManagement()
        self.model_path = f'{name_manager.getFileName()}_svm.joblib'
        self.scaler_path = f'{name_manager.getScalerName()}_svm.joblib'
        self.tfidf_path = f'{name_manager.getFileName()}_tfidf.joblib'
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Enhanced text preprocessing with NLP techniques
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        tokens = word_tokenize(text)
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def extract_enhanced_features(self, url):
        """
        Extract enhanced features from URL with NLP techniques
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Dictionary of extracted features
        """
        if pd.isna(url):
            return {feature: 0 for feature in self.numerical_features}
            
        url = str(url).lower()
        
        # Basic URL components
        has_https = 'https://' in url
        has_http = 'http://' in url
        has_www = 'www.' in url
        
        # Remove protocol and www
        clean_url = re.sub(r'^https?://', '', url)
        clean_url = re.sub(r'^www\.', '', clean_url)
        
        try:
            parsed = urlparse(('http://' + clean_url) if not (has_http or has_https) 
                             else (('https://' if has_https else 'http://') + clean_url))
            
            # Domain analysis
            domain = parsed.netloc
            domain_parts = domain.split('.')
            is_ip = all(part.isdigit() for part in domain_parts)
            
            # Enhanced metrics
            url_length = len(url)
            path_length = len(parsed.path)
            query_length = len(parsed.query)
            fragment_length = len(parsed.fragment)
            
            # Character analysis
            num_dots = url.count('.')
            num_slashes = url.count('/')
            num_digits = sum(c.isdigit() for c in url)
            num_special_chars = len(re.findall(r'[^a-zA-Z0-9./]', url))
            num_underscores = url.count('_')
            num_hyphens = url.count('-')
            num_equals = url.count('=')
            num_ampersands = url.count('&')
            
            # Domain features
            domain_length = len(domain)
            subdomain_count = len(domain_parts) - 1 if len(domain_parts) > 1 else 0
            
            # Path analysis
            path_parts = [p for p in parsed.path.split('/') if p]
            path_depth = len(path_parts)
            
            # Query analysis
            query_params = parsed.query.split('&') if parsed.query else []
            query_param_count = len(query_params)
            
            # Suspicious patterns
            suspicious_extensions = ['.exe', '.zip', '.rar', '.pdf', '.doc', '.docx', '.bat', '.cmd']
            has_suspicious_extension = any(ext in url for ext in suspicious_extensions)
            
            suspicious_keywords = ['login', 'signin', 'account', 'bank', 'secure', 'verify', 'update', 'confirm']
            has_suspicious_keyword = any(keyword in url for keyword in suspicious_keywords)
            
            legitimate_domains = ['google', 'facebook', 'amazon', 'microsoft', 'apple', 'github', 'stackoverflow']
            is_legitimate_domain = any(domain in url for domain in legitimate_domains)
            
            # Calculate entropy (measure of randomness)
            char_freq = {}
            for char in url:
                char_freq[char] = char_freq.get(char, 0) + 1
            
            entropy_score = 0
            url_len = len(url)
            if url_len > 0:
                for freq in char_freq.values():
                    p = freq / url_len
                    if p > 0:
                        entropy_score -= p * np.log2(p)
            
            return {
                'url_length': url_length,
                'path_length': path_length,
                'query_length': query_length,
                'fragment_length': fragment_length,
                'num_dots': num_dots,
                'num_slashes': num_slashes,
                'num_digits': num_digits,
                'num_special_chars': num_special_chars,
                'num_underscores': num_underscores,
                'num_hyphens': num_hyphens,
                'num_equals': num_equals,
                'num_ampersands': num_ampersands,
                'domain_length': domain_length,
                'subdomain_count': subdomain_count,
                'path_depth': path_depth,
                'query_param_count': query_param_count,
                'has_https': int(has_https),
                'has_www': int(has_www),
                'is_ip_address': int(is_ip),
                'has_suspicious_extension': int(has_suspicious_extension),
                'has_suspicious_keyword': int(has_suspicious_keyword),
                'is_legitimate_domain': int(is_legitimate_domain),
                'entropy_score': entropy_score
            }
            
        except Exception as e:
            # Return default values if parsing fails
            return {feature: 0 for feature in self.numerical_features}

    def create_text_features(self, urls):
        """
        Create text features using TF-IDF vectorization
        
        Args:
            urls (list): List of URLs
            
        Returns:
            sparse matrix: TF-IDF features
        """
        # Preprocess URLs for text analysis
        processed_urls = []
        for url in urls:
            if pd.isna(url):
                processed_urls.append("")
            else:
                # Extract meaningful text from URL
                url_text = str(url).lower()
                # Remove common URL patterns to focus on meaningful text
                url_text = re.sub(r'https?://', '', url_text)
                url_text = re.sub(r'www\.', '', url_text)
                url_text = re.sub(r'[0-9]+', 'NUM', url_text)  # Replace numbers
                url_text = re.sub(r'[^\w\s]', ' ', url_text)  # Replace special chars with spaces
                processed_urls.append(url_text)
        
        # Fit and transform TF-IDF
        tfidf_features = self.tfidf_vectorizer.fit_transform(processed_urls)
        return tfidf_features

    def create_model(self):
        """
        Create SVM model with optimized parameters
        
        Returns:
            Pipeline: SVM model pipeline
        """
        # Create SVM with RBF kernel and optimized parameters
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        return svm_model

    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize SVM hyperparameters using GridSearchCV
        
        Args:
            X_train (array): Training features
            y_train (array): Training labels
            
        Returns:
            SVC: Optimized SVM model
        """
        # Define parameter grid for optimization
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
        # Create base SVM model
        base_svm = SVC(probability=True, random_state=42, class_weight='balanced')
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Optimizing hyperparameters...")
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

    def train(self, dataset_path):
        """
        Train the SVM model on the dataset
        
        Args:
            dataset_path (str): Path to the dataset
            
        Returns:
            dict: Training history and metrics
        """
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)
        
        # Create binary labels
        y = (df['url_status'] != 'online').astype(int)
        
        # Extract numerical features
        print("Extracting numerical features...")
        numerical_features = []
        for _, row in df.iterrows():
            features = self.extract_enhanced_features(row['url'])
            numerical_features.append([features[feature] for feature in self.numerical_features])
        
        X_numerical = np.array(numerical_features)
        
        # Create text features
        print("Creating text features...")
        X_text = self.create_text_features(df['url'].tolist())
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_text.toarray()])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters
        self.model = self.optimize_hyperparameters(X_train_scaled, y_train)
        
        # Train final model
        print("Training final model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save metrics
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/svm_model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save model and components
        self.save_model()
        
        print(f"Training completed!")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        return metrics

    def predict(self, url):
        """
        Predict if a URL is malicious
        
        Args:
            url (str): URL to classify
            
        Returns:
            tuple: (prediction, confidence, reason)
        """
        if self.model is None:
            self.load_model()
        
        # Check if known safe domain
        if self.is_safe_domain(url):
            return 'safe', 0.95, 'Known safe domain'
        
        # Extract features
        numerical_features = self.extract_enhanced_features(url)
        X_numerical = np.array([[numerical_features[feature] for feature in self.numerical_features]])
        
        # Create text features
        X_text = self.tfidf_vectorizer.transform([url])
        
        # Combine features
        X_combined = np.hstack([X_numerical, X_text.toarray()])
        
        # Scale features
        X_scaled = self.scaler.transform(X_combined)
        
        # Make prediction
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        prediction = 'malicious' if prediction_proba[1] > 0.5 else 'safe'
        
        # Calculate confidence
        confidence = max(prediction_proba)
        
        # Generate reason
        reason = self._generate_reason(numerical_features, prediction, confidence)
        
        return prediction, confidence, reason

    def is_safe_domain(self, url):
        """
        Check if URL belongs to known safe domain
        
        Args:
            url (str): URL to check
            
        Returns:
            bool: True if safe domain
        """
        try:
            url = re.sub(r'^https?://', '', url.lower())
            url = re.sub(r'^www\.', '', url)
            domain = url.split('/')[0].split(':')[0]
            
            if domain in ['localhost', '127.0.0.1'] or domain.startswith('192.168.') or domain.startswith('10.'):
                return True
            
            if domain in self.safe_tlds:
                return True
            
            parts = domain.split('.')
            if len(parts) >= 2:
                main_domain = '.'.join(parts[-2:])
                if main_domain in self.safe_tlds:
                    return True
                
                tld = parts[-1]
                if tld in self.safe_tlds and len(parts) <= 3:
                    return True
            
            return False
        except:
            return False

    def _generate_reason(self, features, prediction, confidence):
        """
        Generate explanation for prediction
        
        Args:
            features (dict): Extracted features
            prediction (str): Model prediction
            confidence (float): Prediction confidence
            
        Returns:
            str: Explanation
        """
        reasons = []
        
        if features['has_suspicious_extension']:
            reasons.append("Contains suspicious file extension")
        
        if features['has_suspicious_keyword']:
            reasons.append("Contains suspicious keywords")
        
        if features['entropy_score'] > 4.0:
            reasons.append("High entropy (random character distribution)")
        
        if features['num_special_chars'] > 10:
            reasons.append("High number of special characters")
        
        if features['subdomain_count'] > 3:
            reasons.append("Multiple subdomains")
        
        if features['query_param_count'] > 5:
            reasons.append("Many query parameters")
        
        if features['is_ip_address']:
            reasons.append("Uses IP address instead of domain")
        
        if not features['has_https']:
            reasons.append("No HTTPS protocol")
        
        if not reasons:
            if prediction == 'malicious':
                reasons.append("General suspicious patterns detected")
            else:
                reasons.append("No obvious suspicious patterns")
        
        return "; ".join(reasons)

    def save_model(self):
        """Save the trained model and components"""
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.tfidf_vectorizer, self.tfidf_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the trained model and components"""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.tfidf_vectorizer = joblib.load(self.tfidf_path)
            print("Model loaded successfully")
        else:
            raise FileNotFoundError("No trained model found")

    def get_feature_importance(self):
        """
        Get feature importance for SVM model
        
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # For SVM with RBF kernel, we can't directly get feature importance
        # But we can analyze the support vectors
        if hasattr(self.model, 'support_vectors_'):
            # Calculate feature importance based on support vectors
            support_vectors = self.model.support_vectors_
            dual_coef = self.model.dual_coef_[0]
            
            # Calculate importance as weighted sum of support vectors
            importance = np.abs(np.dot(dual_coef, support_vectors))
            
            # Create feature names
            feature_names = self.numerical_features + [f'tfidf_{i}' for i in range(support_vectors.shape[1] - len(self.numerical_features))]
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importance))
            
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}

    def visualize_model(self, save_path='plots/'):
        """
        Visualize model performance and features
        
        Args:
            save_path (str): Path to save visualizations
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs(save_path, exist_ok=True)
        
        # Load metrics if available
        metrics_path = 'metrics/svm_model_metrics.json'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            # Plot confusion matrix
            cm = np.array(metrics['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('SVM Model Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'{save_path}svm_confusion_matrix.png')
            plt.close()
            
            # Plot feature importance
            importance = self.get_feature_importance()
            if importance:
                top_features = dict(list(importance.items())[:15])
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(top_features)), list(top_features.values()))
                plt.yticks(range(len(top_features)), list(top_features.keys()))
                plt.xlabel('Feature Importance')
                plt.title('Top 15 Feature Importance (SVM)')
                plt.tight_layout()
                plt.savefig(f'{save_path}svm_feature_importance.png')
                plt.close()
        
        print(f"Model visualizations saved to {save_path}") 