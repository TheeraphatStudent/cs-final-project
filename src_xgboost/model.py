import os
import numpy as np
import pandas as pd
import xgboost as xgb
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
from __types__.name import NameManagement

nltk.download('punkt')
nltk.download('stopwords')


class URLClassifier:
    safe_tlds = {
        'com', 'org', 'edu', 'gov', 'net', 'io', 'co', 'me', 'info',
        'app', 'dev', 'ai', 'cloud', 'tech', 'mil', 'int'
    }

    def __init__(self):
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
        self.model_path = f'{name_manager.getFileName()}.pkl'
        self.scaler_path = f'{name_manager.getScalerName()}.joblib'
        self.batch_size = 1000

    def preprocess_url(self, url):
        url = url.lower()
        has_https = 'https://' in url
        has_http = 'http://' in url
        url = re.sub(r'^https?://', '', url)
        has_www = url.startswith('www.')
        url = re.sub(r'^www\.', '', url)

        try:
            parsed = urlparse(('http://' + url) if not (has_http or has_https)
                              else (('https://' if has_https else 'http://') + url))

            domain = parsed.netloc
            domain_parts = domain.split('.')
            is_ip = all(part.isdigit() for part in domain_parts)

            tld = domain_parts[-1] if domain_parts else ''
            is_safe_tld = tld in self.safe_tlds

            path_parts = [p for p in parsed.path.split('/') if p]
            query_parts = parsed.query.split('&') if parsed.query else []

            features = []

            if has_https:
                features.append('HTTPS')
            if has_http:
                features.append('HTTP')
            if has_www:
                features.append('WWW')

            features.append(f"DOMAIN_{domain}")
            for part in domain_parts:
                features.append(f"DOMAIN_PART_{part}")

            features.append(f"TLD_{tld}")
            if is_safe_tld:
                features.append('SAFE_TLD')

            for part in path_parts:
                features.append(f"PATH_{part}")

            for part in query_parts:
                features.append(f"QUERY_{part}")

            if is_ip:
                features.append('IS_IP_ADDRESS')
            if len(domain_parts) > 3:
                features.append('MANY_SUBDOMAINS')
            if len(path_parts) > 5:
                features.append('DEEP_PATH')
            if len(query_parts) > 3:
                features.append('MANY_QUERY_PARAMS')

            features = [re.sub(r'[^\w\s-]', '', f) for f in features if f]

            return ' '.join(features)
        except:
            return url

    def extract_features(self, df):
        X = df[self.feature_columns].values
        X = self.scaler.fit_transform(X)
        return X

    def create_model(self):
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        return model

    def process_batch(self, urls_batch, labels_batch, training=True):
        X_batch = self.extract_features(urls_batch)

        if training and self.scaler is None:
            self.scaler = StandardScaler(with_mean=False)
            X_batch = self.scaler.fit_transform(X_batch)
        elif self.scaler is not None:
            X_batch = self.scaler.transform(X_batch)

        return X_batch, labels_batch

    def train(self, dataset_path):
        print("Loading dataset...")
        df = pd.read_csv(dataset_path)

        y = (df['url_status'] != 'online').astype(int)

        X = self.extract_features(df)

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = self.create_model()

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            # early_stopping_rounds=10,
            verbose=True
        )

        y_pred = self.model.predict(X_test)
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_history': {
                'accuracy': [float(self.model.score(X_train, y_train))],
                'val_accuracy': [float(self.model.score(X_test, y_test))]
            }
        }

        os.makedirs('metrics', exist_ok=True)
        with open('metrics/model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        self.save_model()

        return {'accuracy': [metrics['accuracy']], 'val_accuracy': [metrics['val_accuracy']]}

    def is_safe_domain(self, url):
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

    def predict(self, url):
        if self.model is None:
            self.load_model()

        if self.is_safe_domain(url):
            return 'safe', 0.95, 'Known safe domain'

        df = pd.DataFrame({'url': [url]})

        from data_processes import DataProcessor
        processor = DataProcessor(None, None)

        components = processor.extract_url_components(url)
        metrics = processor.calculate_url_metrics(url)
        patterns = processor.check_suspicious_patterns(url)

        features = {**components, **metrics, **patterns}
        feature_df = pd.DataFrame([features])

        X = feature_df[self.feature_columns].values
        X = self.scaler.transform(X)

        prediction_proba = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]

        reason = self._generate_reason(features, prediction_proba[1])

        return ('malicious' if prediction == 1 else 'safe',
                confidence,
                reason)

    def _generate_reason(self, features, prediction):
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
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            raise FileNotFoundError("Model file not found")

    def visualize_model(self, save_path='plots/'):
        if self.model is None:
            raise ValueError("Model not initialized. Please train or load the model first.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        import matplotlib.pyplot as plt
        xgb.plot_importance(self.model, max_num_features=10)
        plt.tight_layout()
        plt.savefig(save_path + 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
