import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from urllib.parse import urlparse
import hashlib

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataProcessorSVM:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # NLP init
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.feature_columns = [
            'url_length', 'path_length', 'query_length', 'fragment_length',
            'num_dots', 'num_slashes', 'num_digits', 'num_special_chars',
            'num_underscores', 'num_hyphens', 'num_equals', 'num_ampersands',
            'domain_length', 'subdomain_count', 'path_depth', 'query_param_count',
            'has_https', 'has_www', 'is_ip_address', 'has_suspicious_extension',
            'has_suspicious_keyword', 'is_legitimate_domain', 'entropy_score',
            'hash_entropy', 'avg_word_length', 'unique_char_ratio'
        ]
        
        # Suspicious patterns
        self.suspicious_extensions = [
            '.exe', '.zip', '.rar', '.pdf', '.doc', '.docx', '.bat', '.cmd',
            '.scr', '.pif', '.com', '.vbs', '.js', '.jar', '.apk', '.dmg'
        ]
        
        self.suspicious_keywords = [
            'login', 'signin', 'account', 'bank', 'secure', 'verify', 'update',
            'confirm', 'password', 'credit', 'card', 'paypal', 'ebay', 'amazon',
            'facebook', 'google', 'microsoft', 'apple', 'admin', 'root'
        ]
        
        self.legitimate_domains = [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
            'youtube.com', 'twitter.com', 'linkedin.com', 'reddit.com'
        ]

    def extract_enhanced_url_components(self, url):
        if pd.isna(url):
            return {
                'subdomain': '', 'domain': '', 'suffix': '', 'path': '',
                'query': '', 'fragment': '', 'has_https': False, 'has_http': False,
                'text_content': '', 'cleaned_text': ''
            }
            
        url = str(url).lower()
        
        # Extract protocol
        has_https = 'https://' in url
        has_http = 'http://' in url
        
        # Remove protocol
        clean_url = re.sub(r'^https?://', '', url)
        
        # Split into parts
        parts = clean_url.split('/', 1)
        domain_part = parts[0]
        path_part = parts[1] if len(parts) > 1 else ''
        
        # Split domain into components
        domain_parts = domain_part.split('.')
        if len(domain_parts) >= 2:
            suffix = domain_parts[-1]
            domain = domain_parts[-2]
            subdomain = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
        else:
            suffix = ''
            domain = domain_part
            subdomain = ''
            
        # Split path into components
        path_parts = path_part.split('?', 1)
        path = path_parts[0]
        query = path_parts[1].split('#')[0] if len(path_parts) > 1 else ''
        fragment = path_parts[1].split('#')[1] if len(path_parts) > 1 and '#' in path_parts[1] else ''
        
        # Extract text content for NLP analysis
        text_content = f"{subdomain} {domain} {suffix} {path} {query}".strip()
        cleaned_text = self.preprocess_text(text_content)
        
        return {
            'subdomain': subdomain,
            'domain': domain,
            'suffix': suffix,
            'path': path,
            'query': query,
            'fragment': fragment,
            'has_https': has_https,
            'has_http': has_http,
            'text_content': text_content,
            'cleaned_text': cleaned_text
        }

    def preprocess_text(self, text):
        """
        Advanced text preprocessing with NLP techniques
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def calculate_enhanced_metrics(self, url):
        """
        Calculate enhanced metrics for URL analysis
        
        Args:
            url (str): URL to analyze
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        if pd.isna(url):
            return {feature: 0 for feature in self.feature_columns}
            
        url = str(url)
        components = self.extract_enhanced_url_components(url)
        
        # Basic metrics
        url_length = len(url)
        path_length = len(components['path'])
        query_length = len(components['query'])
        fragment_length = len(components['fragment'])
        
        # Character analysis
        num_dots = url.count('.')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        num_special_chars = len(re.findall(r'[^a-zA-Z0-9./]', url))
        num_underscores = url.count('_')
        num_hyphens = url.count('-')
        num_equals = url.count('=')
        num_ampersands = url.count('&')
        
        # Domain analysis
        domain = f"{components['subdomain']}.{components['domain']}.{components['suffix']}" if components['subdomain'] else f"{components['domain']}.{components['suffix']}"
        domain_length = len(domain)
        subdomain_count = len(components['subdomain'].split('.')) if components['subdomain'] else 0
        
        # Path analysis
        path_parts = [p for p in components['path'].split('/') if p]
        path_depth = len(path_parts)
        
        # Query analysis
        query_params = components['query'].split('&') if components['query'] else []
        query_param_count = len(query_params)
        
        # Protocol and structure
        has_https = int(components['has_https'])
        has_www = int('www.' in url)
        is_ip_address = int(all(part.isdigit() for part in domain.split('.')) if '.' in domain else False)
        
        # Suspicious patterns
        has_suspicious_extension = int(any(ext in url for ext in self.suspicious_extensions))
        has_suspicious_keyword = int(any(keyword in url for keyword in self.suspicious_keywords))
        is_legitimate_domain = int(any(domain in url for domain in self.legitimate_domains))
        
        # Advanced metrics
        entropy_score = self.calculate_entropy(url)
        hash_entropy = self.calculate_hash_entropy(url)
        avg_word_length = self.calculate_avg_word_length(components['text_content'])
        unique_char_ratio = self.calculate_unique_char_ratio(url)
        
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
            'has_https': has_https,
            'has_www': has_www,
            'is_ip_address': is_ip_address,
            'has_suspicious_extension': has_suspicious_extension,
            'has_suspicious_keyword': has_suspicious_keyword,
            'is_legitimate_domain': is_legitimate_domain,
            'entropy_score': entropy_score,
            'hash_entropy': hash_entropy,
            'avg_word_length': avg_word_length,
            'unique_char_ratio': unique_char_ratio
        }

    def calculate_entropy(self, text):
        """
        Calculate Shannon entropy of text
        
        Args:
            text (str): Input text
            
        Returns:
            float: Entropy value
        """
        if not text:
            return 0.0
            
        char_freq = {}
        for char in text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        entropy = 0.0
        text_len = len(text)
        
        for freq in char_freq.values():
            p = freq / text_len
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def calculate_hash_entropy(self, text):
        """
        Calculate entropy of hash representation
        
        Args:
            text (str): Input text
            
        Returns:
            float: Hash entropy value
        """
        if not text:
            return 0.0
            
        # Create hash
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        return self.calculate_entropy(hash_hex)

    def calculate_avg_word_length(self, text):
        """
        Calculate average word length
        
        Args:
            text (str): Input text
            
        Returns:
            float: Average word length
        """
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        return sum(len(word) for word in words) / len(words)

    def calculate_unique_char_ratio(self, text):
        """
        Calculate ratio of unique characters
        
        Args:
            text (str): Input text
            
        Returns:
            float: Unique character ratio
        """
        if not text:
            return 0.0
            
        unique_chars = len(set(text))
        total_chars = len(text)
        
        return unique_chars / total_chars if total_chars > 0 else 0.0

    def check_enhanced_suspicious_patterns(self, url):
        """
        Enhanced suspicious pattern detection
        
        Args:
            url (str): URL to check
            
        Returns:
            dict: Dictionary of suspicious pattern flags
        """
        if pd.isna(url):
            return {
                'has_suspicious_extension': False,
                'has_suspicious_keyword': False,
                'is_legitimate_domain': False,
                'has_obfuscation': False,
                'has_redirect': False
            }
            
        url = str(url).lower()
        
        # Check for suspicious extensions
        has_suspicious_extension = any(ext in url for ext in self.suspicious_extensions)
        
        # Check for suspicious keywords
        has_suspicious_keyword = any(keyword in url for keyword in self.suspicious_keywords)
        
        # Check for legitimate domains
        is_legitimate_domain = any(domain in url for domain in self.legitimate_domains)
        
        # Check for obfuscation techniques
        has_obfuscation = any([
            '%' in url,  # URL encoding
            '\\x' in url,  # Hex encoding
            '\\u' in url,  # Unicode encoding
            len(re.findall(r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}', url)) > 0  # IP addresses
        ])
        
        # Check for redirect patterns
        has_redirect = any([
            'redirect' in url,
            'goto' in url,
            'jump' in url,
            'url=' in url
        ])
        
        return {
            'has_suspicious_extension': has_suspicious_extension,
            'has_suspicious_keyword': has_suspicious_keyword,
            'is_legitimate_domain': is_legitimate_domain,
            'has_obfuscation': has_obfuscation,
            'has_redirect': has_redirect
        }

    def process_data(self):
        """
        Enhanced data processing pipeline
        
        Returns:
            pd.DataFrame: Processed dataset
        """
        try:
            print(f"Reading CSV file: {self.input_path}")
            df = pd.read_csv(self.input_path, nrows=500)
            
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            if 'url' not in df.columns:
                raise ValueError("Column 'url' not found in dataset")
            
            components_data = []
            for _, row in df.iterrows():
                components = self.extract_enhanced_url_components(row['url'])
                components_data.append(components)
            
            components_df = pd.DataFrame(components_data)
            
            print("Calculating enhanced metrics...")
            metrics_data = []
            for _, row in df.iterrows():
                metrics = self.calculate_enhanced_metrics(row['url'])
                metrics_data.append(metrics)
            
            metrics_df = pd.DataFrame(metrics_data)
            
            print("Checking suspicious patterns...")
            suspicious_data = []
            for _, row in df.iterrows():
                suspicious = self.check_enhanced_suspicious_patterns(row['url'])
                suspicious_data.append(suspicious)
            
            suspicious_df = pd.DataFrame(suspicious_data)
            
            print("Combining features...")
            # Use the correct column names from our CSV
            base_columns = ['url']
            if 'type' in df.columns:
                base_columns.append('type')
            if 'isMalicious' in df.columns:
                base_columns.append('isMalicious')
            
            result_df = pd.concat([
                df[base_columns],
                components_df,
                metrics_df,
                suspicious_df
            ], axis=1)
            
            # Add text features for NLP
            print("Adding text features...")
            result_df['text_features'] = result_df['cleaned_text'].fillna('')
            
            # Create binary labels based on isMalicious column if available
            if 'isMalicious' in df.columns:
                # Convert string 'True'/'False' to boolean if needed
                if df['isMalicious'].dtype == 'object':
                    result_df['is_malicious'] = df['isMalicious'].map({'True': 1, 'False': 0, True: 1, False: 0})
                else:
                    result_df['is_malicious'] = df['isMalicious'].astype(int)
            else:
                # Fallback: use type column to determine maliciousness
                if 'type' in df.columns:
                    result_df['is_malicious'] = (df['type'] != 'benign').astype(int)
                else:
                    # Default to 0 if no label information available
                    result_df['is_malicious'] = 0
            
            # Save processed data
            print(f"Saving processed data to {self.output_path}...")
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            result_df.to_csv(self.output_path, index=False)
            
            print("Data processing completed successfully!")
            return result_df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return None

    def get_data_summary(self):
        """
        Get comprehensive data summary
        
        Returns:
            dict: Data summary statistics
        """
        try:
            df = pd.read_csv(self.output_path)
            
            summary = {
                'total_urls': len(df),
                'malicious_count': df['is_malicious'].sum(),
                'safe_count': len(df) - df['is_malicious'].sum(),
                'malicious_ratio': df['is_malicious'].mean(),
                'avg_url_length': df['url_length'].mean(),
                'avg_entropy': df['entropy_score'].mean(),
                'https_ratio': df['has_https'].mean(),
                'suspicious_extension_ratio': df['has_suspicious_extension'].mean(),
                'suspicious_keyword_ratio': df['has_suspicious_keyword'].mean(),
                'legitimate_domain_ratio': df['is_legitimate_domain'].mean()
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
            return None

    def validate_data(self):
        """
        Validate processed data quality
        
        Returns:
            dict: Validation results
        """
        try:
            df = pd.read_csv(self.output_path)
            
            validation = {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_urls': df['url'].duplicated().sum(),
                'invalid_urls': len(df[df['url_length'] == 0]),
                'feature_ranges': {
                    'url_length': (df['url_length'].min(), df['url_length'].max()),
                    'entropy_score': (df['entropy_score'].min(), df['entropy_score'].max()),
                    'subdomain_count': (df['subdomain_count'].min(), df['subdomain_count'].max())
                }
            }
            
            return validation
            
        except Exception as e:
            print(f"Error validating data: {str(e)}")
            return None 